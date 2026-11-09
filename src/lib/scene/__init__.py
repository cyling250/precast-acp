"""场景级点云生成与面实例化（T-037，SDS v1）。"""

from .visibility import make_scan_poses, compute_visibility
from .face_labels import assign_member_faces, face_type_name
from .scene_generator import generate_scene, main

__all__ = [
    "make_scan_poses", "compute_visibility",
    "assign_member_faces", "face_type_name",
    "generate_scene", "main",
]
