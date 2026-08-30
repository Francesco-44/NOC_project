"""
pose_from_tf — pubblica la posa del robot come PoseStamped leggendola da TF.

Sorgente di posa ALTERNATIVA a odom_to_pose_node: invece di ripubblicare
/odom, interroga TF a rate fisso. Utile quando la posa nasce da una catena di
trasformazioni (EKF, SLAM) e non da un singolo topic Odometry.

Default di questo stack: `odom -> base_link`, cioe' il frame di pianificazione,
perche' qui non c'e' alcuna mappa a priori (A* lavora sulla griglia gaussiana a
orizzonte mobile). Nel progetto di origine i default erano `map -> base_footprint`
perche' li' la posa veniva da slam_toolbox in modalita' localization.

RILEVAMENTO DI TF CONGELATA (tf_freeze_timeout)
-----------------------------------------------
Il lookup chiede ``Time()`` = "l'ultima disponibile", quindi quando l'odometria
smette di alimentare TF il buffer continua a restituire PER SEMPRE la stessa
trasformazione. Ripubblicarla con lo stamp corrente sarebbe indistinguibile da
una posa sana: a valle il planner continuerebbe a ripianificare da una posa che
non si muove piu', e il robot camminerebbe contro una posa stantia.

Quindi: si pubblica solo finche' lo stamp PROPRIO della trasformazione avanza.
Se resta fermo per tf_freeze_timeout si smette di pubblicare, il topic della
posa tace, e il timeout a valle ferma il robot.

Il confronto e' fra stamp consecutivi, misurando da quanto stanno fermi
sull'orologio LOCALE: non si confronta mai uno stamp remoto con l'orologio
locale, quindi regge anche quando il produttore della posa sta dall'altra parte
del collegamento e i due orologi non concordano.

Intercetta una posa CONGELATA, non una SBAGLIATA: un'odometria che diverge
mantiene gli stamp in avanzamento e supera questo controllo.

Ripreso dal progetto Unitree-G1 (policypilot/navigation/pose_from_tf.py).
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener, TransformException


class PoseFromTf(Node):
    def __init__(self):
        super().__init__('pose_from_tf')
        self.declare_parameter('map_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('rate_hz', 30.0)
        self.declare_parameter('pose_topic', '/robot_pose')
        # Stop publishing once the transform's own stamp has stood still this
        # long (see the module docstring). 0 or less disables the check.
        self.declare_parameter('tf_freeze_timeout', 0.5)

        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.freeze_timeout = float(self.get_parameter('tf_freeze_timeout').value)

        self._last_stamp_ns = None    # stamp of the last transform we saw
        self._last_change = None      # local time that stamp last advanced
        self._frozen = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(
            PoseStamped, self.get_parameter('pose_topic').value, 10)

        rate = float(self.get_parameter('rate_hz').value)
        self.create_timer(1.0 / rate, self._tick)
        self._warned = False
        self.get_logger().info(
            f"pose_from_tf: {self.map_frame}->{self.base_frame} → "
            f"{self.get_parameter('pose_topic').value} "
            f"(freeze timeout {self.freeze_timeout:.2f}s)")

    def _is_frozen(self, tf) -> bool:
        """True once the transform's own stamp has stood still longer than
        tf_freeze_timeout — i.e. TF is being re-served, not re-computed."""
        if self.freeze_timeout <= 0.0:
            return False
        stamp_ns = Time.from_msg(tf.header.stamp).nanoseconds
        now = self.get_clock().now()
        if stamp_ns != self._last_stamp_ns:
            self._last_stamp_ns = stamp_ns
            self._last_change = now
            if self._frozen:
                self._frozen = False
                self.get_logger().warn(
                    f"TF {self.map_frame}->{self.base_frame} is advancing again "
                    f"— /robot_pose resumed. The robot navigated blind for part "
                    f"of the gap: check where it actually is before re-arming.")
            return False
        if self._last_change is None:
            self._last_change = now
            return False
        held = (now - self._last_change).nanoseconds * 1e-9
        if held <= self.freeze_timeout:
            return False
        if not self._frozen:
            self._frozen = True
            self.get_logger().error(
                f"*** TF {self.map_frame}->{self.base_frame} FROZEN for "
                f"{held:.2f}s (same stamp re-served) — the odometry stopped "
                f"feeding TF. Muting /robot_pose so the MPC stops instead of "
                f"walking against a stale pose. ***")
        else:
            self.get_logger().error(
                f"    {self.map_frame}->{self.base_frame} still frozen "
                f"({held:.1f}s)", throttle_duration_sec=2.0)
        return True

    def _tick(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
        except TransformException as e:
            if not self._warned:
                self.get_logger().warn(
                    f"waiting for TF {self.map_frame}->{self.base_frame}: {e}",
                    throttle_duration_sec=3.0)
                self._warned = True
            return
        if self._is_frozen(tf):
            return
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.pose.position.x = tf.transform.translation.x
        msg.pose.position.y = tf.transform.translation.y
        msg.pose.position.z = tf.transform.translation.z
        msg.pose.orientation = tf.transform.rotation
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PoseFromTf()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
