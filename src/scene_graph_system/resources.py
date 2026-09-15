"""Locate source/install resources and keep mutable state outside the package."""

from pathlib import Path
import os
from typing import Optional


def get_package_root() -> Optional[str]:
    """Return the ROS package share directory in a checkout or catkin install."""
    override = os.environ.get("SCENE_GRAPH_SYSTEM_ROOT")
    if override:
        candidate = Path(override).expanduser().resolve()
        if not (candidate / "package.xml").is_file():
            raise ValueError("SCENE_GRAPH_SYSTEM_ROOT must contain package.xml")
        return str(candidate)
    for parent in Path(__file__).resolve().parents:
        if (parent / "package.xml").is_file() and (parent / "config").is_dir():
            return str(parent)
    try:
        import rospkg
    except ImportError:
        return None
    try:
        return rospkg.RosPack().get_path("scene_graph_system")
    except rospkg.ResourceNotFound:
        return None


def package_resource_path(relative_path: str) -> str:
    """Resolve package data, including legacy model names (existence not required)."""
    root = get_package_root()
    if root is None:
        raise RuntimeError("Cannot locate scene_graph_system; source the catkin workspace or set SCENE_GRAPH_SYSTEM_ROOT")
    relative = Path(relative_path)
    if relative.name in ("best.pt", "global_seg_best.pt") and str(relative.parent).replace('\\', '/') in ('.', 'scripts', 'resources', 'models'):
        relative = Path("models") / relative.name
    elif relative.name in ("action.py", "sorting_action.py") and str(relative.parent).replace('\\', '/') in ('.', 'scripts', 'resources'):
        relative = Path("scripts") / relative.name
    return str(Path(root) / relative)


def find_package_resource(relative_path: str) -> Optional[str]:
    path = package_resource_path(relative_path)
    return path if Path(path).exists() else None


def runtime_data_path(filename: str) -> str:
    """Use SCENE_GRAPH_RUNTIME_DIR or ROS_HOME/scene_graph_system for mutable data."""
    directory = os.environ.get("SCENE_GRAPH_RUNTIME_DIR")
    if directory:
        root = Path(directory).expanduser()
    else:
        ros_home = os.environ.get("ROS_HOME")
        root = (Path(ros_home).expanduser() if ros_home else Path.home() / ".ros") / "scene_graph_system"
    return str(root / filename)
