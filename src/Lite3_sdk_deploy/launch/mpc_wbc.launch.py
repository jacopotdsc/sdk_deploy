"""Run independent MPC and WBC nodes; rl_deploy connects separately."""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def find_mpx_root():
    folder = os.path.dirname(os.path.realpath(__file__))
    while True:
        for relative in (("mpc", "upstream"),
                         ("src", "Lite3_sdk_deploy", "mpc", "upstream")):
            candidate = os.path.join(folder, *relative)
            if os.path.isfile(os.path.join(candidate, "mpx", "utils", "mpc_wrapper_srbd.py")):
                return candidate
        parent = os.path.dirname(folder)
        if parent == folder:
            return None
        folder = parent


def generate_launch_description():
    mpx_root = find_mpx_root()
    defaults = {
        "mpx_root": mpx_root,
        "pipeline_namespace": "",
        "mpc_frequency": "50.0",
        "mpc_kinematics_backend": "mujoco",
        "wbc_frequency": "100.0",
        "wbc_backend": "ik",
        "wbc_targets_backend": "integration",
        "torque_limit_hip": "40.0",
        "torque_limit_thigh": "40.0",
        "torque_limit_knee": "65.0",
    }
    choices = {"wbc_backend": ["jax", "pinocchio", "ik"],
               "wbc_targets_backend": ["integration", "ik"],
               "mpc_kinematics_backend": ["mujoco", "pinocchio"]}
    arguments = [DeclareLaunchArgument(
        name, default_value=value,
        **({"choices": choices[name]} if name in choices else {}))
        for name, value in defaults.items()]

    def value(name):
        return ParameterValue(LaunchConfiguration(name), value_type=float)

    namespace = LaunchConfiguration("pipeline_namespace")
    return LaunchDescription(arguments + [
        Node(package="lite3_sdk_deploy", executable="mpc_srbd",
             namespace=namespace, output="screen",
             parameters=[{"mpc_kinematics_backend": ParameterValue(LaunchConfiguration("mpc_kinematics_backend"), value_type=str), "mpx_root": ParameterValue(LaunchConfiguration("mpx_root"), value_type=str), "mpc_frequency": value("mpc_frequency")}]),
        Node(package="lite3_sdk_deploy", executable="wbc",
             namespace=namespace, output="screen",
             parameters=[{"wbc_targets_backend": ParameterValue(LaunchConfiguration("wbc_targets_backend"), value_type=str), "mpx_root": ParameterValue(LaunchConfiguration("mpx_root"), value_type=str), "wbc_backend": ParameterValue(LaunchConfiguration("wbc_backend"), value_type=str)},
                         {name: value(name) for name in
                          ("wbc_frequency", "torque_limit_hip",
                           "torque_limit_thigh", "torque_limit_knee")}]),
    ])
