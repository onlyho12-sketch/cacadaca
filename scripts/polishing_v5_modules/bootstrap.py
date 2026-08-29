"""CLI bootstrap for polishing_v5.

SimulationApp must be created before importing modules that touch omni/isaac APIs.
"""
import argparse
import os
import sys
import traceback


def _strip_ros_paths():
    def is_external_ros_path(path):
        return "/opt/ros" in path or "/.local/ros-humble/" in path

    if "PYTHONPATH" in os.environ:
        os.environ["PYTHONPATH"] = ":".join(
            p for p in os.environ["PYTHONPATH"].split(":") if not is_external_ros_path(p)
        )
    sys.path = [p for p in sys.path if not is_external_ros_path(p)]


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("obj_name_pos", nargs="?", help="Object name, for compatibility with positional calls")
    parser.add_argument("--obj_name", type=str, default=None)
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim without opening the GUI")
    args, _ = parser.parse_known_args(argv)
    args.obj_name = (args.obj_name or args.obj_name_pos or "car").strip().lower()
    return args


def run_from_cli(argv=None):
    args = _parse_args(argv)
    _strip_ros_paths()

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({
        "headless": bool(args.headless),
        "renderer": os.environ.get("ISAAC_RENDERER", "RealTimePathTracing"),
    })
    try:
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.ros2.bridge")
        try:
            import rclpy  # noqa: F401
            from std_msgs.msg import Float64  # noqa: F401
        except ImportError:
            pass

        from .runner import main

        main(simulation_app, obj_name=args.obj_name)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
