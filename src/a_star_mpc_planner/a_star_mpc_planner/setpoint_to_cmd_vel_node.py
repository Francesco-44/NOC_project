"""
setpoint_to_cmd_vel_node.py

Converts MPC lookahead setpoints into body-frame /cmd_vel commands for the locomotion layer.

Subscribes:
  <pose_topic>          geometry_msgs/PoseStamped
  /mpc/next_setpoint geometry_msgs/PoseStamped

Publishes:
  /cmd_vel           geometry_msgs/Twist

Behavior:
  - Proportional XY controller in robot body frame.
  - Optional yaw controller (disabled by default to avoid spin-in-place).
  - Safety timeout: publishes zero if setpoint stream is stale.
"""

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def _wrap_to_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _clamp(value: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, value))


def _clamp_delta(target: float, current: float, max_delta: float) -> float:
    return current + _clamp(target - current, -max_delta, max_delta)


class SetpointToCmdVelNode(Node):

    def __init__(self):
        super().__init__('setpoint_to_cmd_vel_node')

        self.declare_parameter('cmd_rate_hz', 20.0)
        self.declare_parameter('cmd_kp_xy', 1.0)
        self.declare_parameter('cmd_kp_yaw', 1.5)
        self.declare_parameter('cmd_max_vx', 0.8)
        # Limite INFERIORE su vx. NaN = simmetrico (-cmd_max_vx), che era il
        # comportamento storico. Va tenuto allineato a mpc_vx_min: il YAML
        # richiede che i clamp non eccedano i limiti del modello dell'MPC,
        # altrimenti l'anello esterno chiede piu' di quanto il piano preveda.
        self.declare_parameter('cmd_min_vx', float('nan'))
        # "Gira prima, poi vai": sopra questo disallineamento di prua la
        # componente LONGITUDINALE del comando viene sfumata verso zero, cosi'
        # il robot ruota (quasi) sul posto invece di percorrere metri con la
        # prua storta. 0 disattiva.
        #
        # Serve perche' il costo dell'MPC NON puo' risolverlo: penalizzare la
        # retromarcia con un termine asimmetrico e' inefficace finche' Q_xy=200
        # domina (misurato: nessun effetto fino a R~10^3, dove pero' equivale a
        # vietarla). E l'MPC gia' ruota al massimo, |wz| saturato: una virata di
        # 180 gradi a omega_max=0.3 richiede 10.5 s, il DOPPIO dell'orizzonte di
        # 5.25 s. Dentro l'orizzonte indietreggiare insegue il riferimento
        # meglio che girarsi, e la scelta e' corretta per il problema che l'MPC
        # vede — solo che il problema e' troncato.
        self.declare_parameter('cmd_turn_first_deg', 0.0)
        # Quanto resta della velocita' longitudinale a prua completamente
        # invertita. 0 = rotazione pura sul posto; un valore piccolo lascia lo
        # sgancio all'indietro che serve a staccarsi da un muro.
        self.declare_parameter('cmd_turn_first_floor', 0.25)
        self.declare_parameter('cmd_max_vy', 0.4)
        self.declare_parameter('cmd_max_omega', 1.2)
        self.declare_parameter('cmd_stop_radius', 0.2)
        self.declare_parameter('setpoint_timeout_sec', 1.0)
        self.declare_parameter('enable_yaw_control', False)
        self.declare_parameter('cmd_smoothing_alpha', 0.35)
        self.declare_parameter('cmd_max_ax', 1.0)
        self.declare_parameter('cmd_max_ay', 0.8)
        self.declare_parameter('cmd_max_alpha', 1.5)

        self._rate_hz = float(self.get_parameter('cmd_rate_hz').value)
        self._kp_xy = float(self.get_parameter('cmd_kp_xy').value)
        self._kp_yaw = float(self.get_parameter('cmd_kp_yaw').value)
        self._max_vx = float(self.get_parameter('cmd_max_vx').value)
        _min_vx = float(self.get_parameter('cmd_min_vx').value)
        self._min_vx = -self._max_vx if _min_vx != _min_vx else _min_vx
        self._turn_first = math.radians(
            float(self.get_parameter('cmd_turn_first_deg').value))
        self._turn_floor = float(self.get_parameter('cmd_turn_first_floor').value)
        self._max_vy = float(self.get_parameter('cmd_max_vy').value)
        self._max_omega = float(self.get_parameter('cmd_max_omega').value)
        self._stop_radius = float(self.get_parameter('cmd_stop_radius').value)
        self._setpoint_timeout = float(self.get_parameter('setpoint_timeout_sec').value)
        self._enable_yaw_control = bool(self.get_parameter('enable_yaw_control').value)
        self._cmd_smoothing_alpha = float(self.get_parameter('cmd_smoothing_alpha').value)
        self._cmd_max_ax = float(self.get_parameter('cmd_max_ax').value)
        self._cmd_max_ay = float(self.get_parameter('cmd_max_ay').value)
        self._cmd_max_alpha = float(self.get_parameter('cmd_max_alpha').value)

        self._pose: PoseStamped | None = None
        self._yaw = 0.0
        self._setpoint: PoseStamped | None = None
        self._setpoint_rx_time = None
        self._last_cmd = Twist()
        self._has_last_cmd = False

        # Nome del topic della posa: parametrico, e' l'unico punto in cui
        # il robot entra in questo nodo. Il G1 pubblica su /robot_pose.
        self.declare_parameter('pose_topic', '/robot_pose')
        _pose_topic = self.get_parameter('pose_topic').value

        self.create_subscription(PoseStamped, _pose_topic, self._pose_cb, 10)
        self.create_subscription(PoseStamped, '/mpc/next_setpoint', self._setpoint_cb, 10)

        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_timer(1.0 / self._rate_hz, self._control_cb)

        self.get_logger().info(
            f'setpoint_to_cmd_vel ready: /mpc/next_setpoint + {_pose_topic} -> /cmd_vel'
        )

    def _pose_cb(self, msg: PoseStamped):
        self._pose = msg
        q = msg.pose.orientation
        self._yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)

    def _setpoint_cb(self, msg: PoseStamped):
        self._setpoint = msg
        self._setpoint_rx_time = self.get_clock().now()

    def _publish_zero(self):
        self._has_last_cmd = False
        self._last_cmd = Twist()
        self._cmd_pub.publish(Twist())

    def _apply_cmd_smoothing(self, raw_cmd: Twist) -> Twist:
        if not self._has_last_cmd:
            self._last_cmd = raw_cmd
            self._has_last_cmd = True
            return raw_cmd

        dt = max(1.0 / max(self._rate_hz, 1e-3), 1e-3)

        vx_limited = _clamp_delta(raw_cmd.linear.x, self._last_cmd.linear.x, self._cmd_max_ax * dt)
        vy_limited = _clamp_delta(raw_cmd.linear.y, self._last_cmd.linear.y, self._cmd_max_ay * dt)
        wz_limited = _clamp_delta(raw_cmd.angular.z, self._last_cmd.angular.z, self._cmd_max_alpha * dt)

        alpha = _clamp(self._cmd_smoothing_alpha, 0.0, 1.0)
        smoothed = Twist()
        smoothed.linear.x = (1.0 - alpha) * self._last_cmd.linear.x + alpha * vx_limited
        smoothed.linear.y = (1.0 - alpha) * self._last_cmd.linear.y + alpha * vy_limited
        smoothed.angular.z = (1.0 - alpha) * self._last_cmd.angular.z + alpha * wz_limited

        self._last_cmd = smoothed
        return smoothed

    def _control_cb(self):
        if self._pose is None or self._setpoint is None or self._setpoint_rx_time is None:
            self.get_logger().warn(
                f'[CMD_VEL] Waiting — pose={self._pose is not None} '
                f'setpoint={self._setpoint is not None}',
                throttle_duration_sec=5.0,
            )
            self._publish_zero()
            return

        now = self.get_clock().now()
        age_sec = (now - self._setpoint_rx_time).nanoseconds * 1e-9
        if age_sec > self._setpoint_timeout:
            self.get_logger().warn(
                f'setpoint timeout ({age_sec:.2f}s > {self._setpoint_timeout:.2f}s), zeroing /cmd_vel',
                throttle_duration_sec=1.0,
            )
            self._publish_zero()
            return

        px = float(self._pose.pose.position.x)
        py = float(self._pose.pose.position.y)
        sx = float(self._setpoint.pose.position.x)
        sy = float(self._setpoint.pose.position.y)

        dx_world = sx - px
        dy_world = sy - py
        dist = math.hypot(dx_world, dy_world)

        # World -> robot body frame (x forward, y left)
        ex = math.cos(self._yaw) * dx_world + math.sin(self._yaw) * dy_world
        ey = -math.sin(self._yaw) * dx_world + math.cos(self._yaw) * dy_world

        # Errore di prua rispetto all'ORIENTAMENTO del setpoint. Calcolato qui,
        # prima del blocco lineare, perche' serve anche alla sfumatura
        # "gira prima, poi vai"; il canale di imbardata piu' sotto lo riusa.
        yaw_sp = _quat_to_yaw(self._setpoint.pose.orientation.x,
                              self._setpoint.pose.orientation.y,
                              self._setpoint.pose.orientation.z,
                              self._setpoint.pose.orientation.w)
        yaw_err = _wrap_to_pi(yaw_sp - self._yaw)

        raw_cmd = Twist()
        if dist <= self._stop_radius:
            raw_cmd.linear.x = 0.0
            raw_cmd.linear.y = 0.0
        else:
            vx_cmd = _clamp(self._kp_xy * ex, self._min_vx, self._max_vx)
            vy_cmd = _clamp(self._kp_xy * ey, -self._max_vy, self._max_vy)

            # Sfumatura "gira prima, poi vai". Il fattore va da 1 (prua
            # allineata) al pavimento (prua invertita) con un coseno rialzato:
            # e' liscio, quindi non introduce gradini nel twist, che e' l'asse
            # su cui un umanoide perde l'equilibrio per primo.
            if self._turn_first > 0.0:
                a = abs(math.atan2(math.sin(yaw_err), math.cos(yaw_err)))
                if a > self._turn_first:
                    t = min(1.0, (a - self._turn_first)
                            / max(1e-6, math.pi - self._turn_first))
                    g = 1.0 - (1.0 - self._turn_floor) * 0.5 * (1.0 - math.cos(math.pi * t))
                    vx_cmd *= g
                    vy_cmd *= g
            raw_cmd.linear.x = vx_cmd
            raw_cmd.linear.y = vy_cmd

        if self._enable_yaw_control and dist > self._stop_radius:
            # yaw_sp / yaw_err sono gia' stati calcolati sopra.
            raw_cmd.angular.z = _clamp(self._kp_yaw * yaw_err, -self._max_omega, self._max_omega)
        else:
            raw_cmd.angular.z = 0.0

        cmd = self._apply_cmd_smoothing(raw_cmd)

        self._cmd_pub.publish(cmd)

        stopped = dist <= self._stop_radius
        self.get_logger().info(
            f'[CMD_VEL] robot=({px:.2f},{py:.2f})  setpt=({sx:.2f},{sy:.2f})  '
            f'dist={dist:.2f}m{"  STOPPED" if stopped else ""}  '
            f'vx={cmd.linear.x:+.2f}  vy={cmd.linear.y:+.2f}  wz={cmd.angular.z:+.2f}',
            throttle_duration_sec=1.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = SetpointToCmdVelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
