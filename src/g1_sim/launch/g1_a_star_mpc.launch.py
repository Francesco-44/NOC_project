"""
g1_a_star_mpc.launch.py — stack completo di navigazione autonoma del G1.

Impianto MuJoCo: in Gazebo il G1 non avrebbe alcuna sorgente di moto senza
scriverne una da zero (vedi g1_sim/README.md).

Catena
------
    mujoco_sim  --/odom------------> odom_to_pose_node --/robot_pose--+
                --/livox/lidar--> lidar_filter_node                   |
                                     |                                |
                                     +--/lidar/points_filtered--------+
                                                    |                 |
                                                    v                 v
                                              a_star_node  ------> mpc_node
                                              /a_star/path      /mpc/next_setpoint
                                                                      |
                                                    setpoint_to_cmd_vel_node
                                                                      |
                                                                 /cmd_vel
                                                                      |
                                                                 mujoco_sim

Il robot entra nella catena solo attraverso il file di parametri e il nome del
topic della posa: nessun nodo dello stack algoritmico dipende dalla piattaforma.

Il goal si manda su /global_goal (PoseStamped) oppure con lo strumento
"2D Goal Pose" di RViz, che pubblica su /goal_pose (vedi argomento goal_relay).

Argomenti
---------
  params_file  : str  profilo del planner (default: planner_params_g1.yaml)
  lidar_params : str  configurazione dell'adattatore LiDAR
  sim_params   : str  configurazione del simulatore
  robot_model  : bool (default true)   pubblica /robot_description per RViz
  use_rviz     : bool (default true)
  viewer       : bool (default true)   finestra MuJoCo
  people       : str  (default '')     ostacoli dinamici: '' oppure 'default'
  nav_graph    : bool (default false)  memoria globale topologica (Dijkstra)
  goal_relay   : bool (default true)   /goal_pose -> /global_goal per RViz
  use_mission  : bool (default false)  esegue una missione a waypoint
  mission_file : str  YAML della missione (vedi robot_real_goal_manager)
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    g1_share      = get_package_share_directory("g1_sim")
    planner_share = get_package_share_directory("a_star_mpc_planner")

    default_planner = os.path.join(planner_share, "config", "planner_params_g1.yaml")
    default_overlay = os.path.join(planner_share, "config", "overlay_none.yaml")
    default_lidar   = os.path.join(g1_share, "config", "lidar_filter_g1.yaml")
    default_sim     = os.path.join(g1_share, "config", "g1_sim.yaml")
    default_rviz    = os.path.join(g1_share, "rviz", "g1_nav.rviz")

    args = [
        DeclareLaunchArgument("params_file",  default_value=default_planner),
        # Secondo file di parametri, fuso SOPRA params_file (ROS 2 applica i
        # file nell'ordine della lista e l'ultimo vince). Serve a variare
        # pochi parametri senza duplicare un profilo da 250 righe: vedi
        # config/overlay_nonconvex.yaml per i mondi concavi.
        DeclareLaunchArgument("planner_overlay", default_value=default_overlay),
        DeclareLaunchArgument("lidar_params", default_value=default_lidar),
        DeclareLaunchArgument("sim_params",   default_value=default_sim),
        DeclareLaunchArgument("rviz_config",  default_value=default_rviz),
        DeclareLaunchArgument("use_rviz",     default_value="true"),
        DeclareLaunchArgument("robot_model",  default_value="true",
                              description="pubblica /robot_description per vedere il G1 in RViz"),
        DeclareLaunchArgument("viewer",       default_value="true"),
        DeclareLaunchArgument("people",       default_value=""),
        # Geometria del mondo MuJoCo. "industrial" e' il magazzino;
        # long_wall / horseshoe / dead_end sono i mondi con ostacoli non
        # convessi (vedi g1_sim/mujoco_world.py, WORLDS). Cambiando mondo
        # cambia anche la posa di spawn, presa dal mondo stesso.
        DeclareLaunchArgument("world",        default_value="industrial"),
        DeclareLaunchArgument("nav_graph",    default_value="false"),
        DeclareLaunchArgument("goal_relay",   default_value="true"),
        DeclareLaunchArgument("use_mission",  default_value="false"),
        DeclareLaunchArgument("mission_file", default_value=""),
        # Ritardo prima che la missione pubblichi il PRIMO goal. Il default del
        # nodo e' 3 s, troppo pochi: il goal partirebbe prima che si riesca ad
        # avviare viz/record_run.sh, e /global_goal e' pubblicato UNA SOLA
        # volta — una bag che se lo perde e' inutilizzabile dagli strumenti
        # di analisi. 20 s bastano per lanciare il recorder con calma.
        DeclareLaunchArgument("mission_delay", default_value="20.0"),
    ]

    params_file  = LaunchConfiguration("params_file")
    lidar_params = LaunchConfiguration("lidar_params")
    sim_params   = LaunchConfiguration("sim_params")

    # Il simulatore E' la sorgente di /clock, quindi e' l'unico nodo che NON usa
    # use_sim_time; tutto il resto della catena lo usa.
    sim_time = {"use_sim_time": True}
    planner  = [params_file, LaunchConfiguration("planner_overlay"), sim_time]

    nodes = [
        # L'URDF ha come radice 'pelvis', mentre mujoco_sim pubblica la posa
        # della base come 'base_link'. Senza questo ponte l'albero del robot
        # resta staccato da odom: RViz non sa dove metterlo, impila i link
        # nell'origine e li disegna bianchi con lo stato in errore.
        # mujoco_sim impone la posa del giunto libero, che nel MJCF E' il
        # bacino, quindi la trasformazione e' l'identita'.
        Node(
            package="tf2_ros", executable="static_transform_publisher",
            name="base_link_to_pelvis", output="log",
            arguments=["0", "0", "0", "0", "0", "0", "base_link", "pelvis"],
            parameters=[sim_time],
            condition=IfCondition(LaunchConfiguration("robot_model")),
        ),

        # ── modello del robot: /joint_states -> TF dei 29 giunti -> RViz ──
        # mujoco_sim pubblica gia' /joint_states; robot_state_publisher li
        # trasforma nella catena TF completa e in /robot_description, che e' cio'
        # che il display RobotModel di RViz consuma. Non serve alla navigazione:
        # con robot_model:=false lo stack funziona identico, senza il modello.
        Node(
            package="robot_state_publisher", executable="robot_state_publisher",
            name="robot_state_publisher", output="log",
            parameters=[sim_time, {"robot_description": ParameterValue(
                Command(["cat ", os.path.join(g1_share, "description", "g1_29dof.urdf")]),
                value_type=str)}],
            condition=IfCondition(LaunchConfiguration("robot_model")),
        ),

        # ── impianto ────────────────────────────────────────────────────
        Node(
            package="g1_sim", executable="mujoco_sim", name="mujoco_sim",
            output="screen",
            parameters=[
                sim_params,
                {"viewer": ParameterValue(LaunchConfiguration("viewer"), value_type=bool),
                 "people": LaunchConfiguration("people"),
                 "world": LaunchConfiguration("world"),
                 "use_sim_time": False},
            ],
        ),

        # ── percezione: nuvola dal frame sensore al frame di pianificazione ──
        # L'adattatore e' completamente parametrico (topic, frame, range,
        # altezze, voxel): il robot entra solo dal YAML.

        Node(
            package="robot_real_lidar", executable="lidar_filter_node",
            name="g1_lidar_filter", output="screen",
            parameters=[lidar_params, sim_time],
        ),

        # ── posa: /odom -> PoseStamped nel frame di pianificazione ──────
        Node(
            package="a_star_mpc_planner", executable="odom_to_pose_node",
            name="odom_to_pose_node", output="screen",
            parameters=planner,
            remappings=[("/odom/raw", "/odom")],
        ),

        # ── pianificazione e controllo ──────────────────────────────────
        Node(
            package="a_star_mpc_planner", executable="a_star_node",
            name="a_star_node", output="screen", parameters=planner,
        ),
        Node(
            package="a_star_mpc_planner", executable="mpc_node",
            name="mpc_node", output="screen", parameters=planner,
        ),
        Node(
            package="a_star_mpc_planner", executable="setpoint_to_cmd_vel_node",
            name="setpoint_to_cmd_vel_node", output="screen", parameters=planner,
        ),

        # ── memoria globale topologica (opzionale) ──────────────────────
        Node(
            package="a_star_mpc_planner", executable="nav_graph_node",
            name="nav_graph_node", output="screen", parameters=planner,
            condition=IfCondition(LaunchConfiguration("nav_graph")),
        ),

        # ── goal: lo strumento "2D Goal Pose" di RViz pubblica su /goal_pose,
        #    lo stack ascolta /global_goal. Il relay e' parametrico e forza il
        #    frame, cosi' si evita una dipendenza da topic_tools ──────────
        Node(
            package="robot_real_goal_manager", executable="goal_relay_node",
            name="goal_relay_node", output="screen",
            parameters=[sim_time, {"input_topic": "/goal_pose",
                                   "output_topic": "/global_goal",
                                   "override_frame": "odom",
                                   "force_frame": True}],
            condition=IfCondition(LaunchConfiguration("goal_relay")),
        ),

        # ── missione a waypoint per prove ripetibili (opzionale) ────────
        # Serve agli esperimenti che richiedono scenari identici ripetuti:
        # vedi guides/roadmap_teorica_noc.md §6.6 (fronte di Pareto).
        Node(
            package="robot_real_goal_manager", executable="mission_runner_node",
            name="mission_runner_node", output="screen",
            parameters=[sim_time, {"mission_file": LaunchConfiguration("mission_file"),
                                   "global_goal_topic": "/global_goal",
                                   "odom_topic": "/odom",
                                   "start_delay_sec": ParameterValue(
                                       LaunchConfiguration("mission_delay"),
                                       value_type=float)}],
            condition=IfCondition(LaunchConfiguration("use_mission")),
        ),

        # ── visualizzazione ─────────────────────────────────────────────
        Node(
            package="rviz2", executable="rviz2", name="rviz2", output="log",
            arguments=["-d", LaunchConfiguration("rviz_config")],
            parameters=[sim_time],
            condition=IfCondition(LaunchConfiguration("use_rviz")),
        ),
    ]

    return LaunchDescription(args + nodes)
