"""
g1_sim.launch.py — solo l'impianto: MuJoCo + G1 + magazzino.

Serve a verificare il simulatore da solo, guidandolo a mano:

    ros2 launch g1_sim g1_sim.launch.py
    ros2 run g1_sim key_teleop

Argomenti
---------
  params_file : str  (default: config/g1_sim.yaml del pacchetto)
  viewer      : bool (default: true)  finestra MuJoCo
  people      : str  (default: '')    '' nessuno, 'default' 3 persone di prova
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("g1_sim")
    default_params = os.path.join(share, "config", "g1_sim.yaml")

    args = [
        DeclareLaunchArgument("params_file", default_value=default_params),
        DeclareLaunchArgument("viewer",      default_value="true"),
        DeclareLaunchArgument("people",      default_value=""),
    ]

    sim = Node(
        package="g1_sim",
        executable="mujoco_sim",
        name="mujoco_sim",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "viewer": ParameterValue(LaunchConfiguration("viewer"), value_type=bool),
                "people": LaunchConfiguration("people"),
                "use_sim_time": False,   # il nodo E' la sorgente di /clock
            },
        ],
    )

    return LaunchDescription(args + [sim])
