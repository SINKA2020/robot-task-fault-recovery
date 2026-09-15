from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple
import math

import numpy as np

from scene_graph_system.scene_graph.relation_utils import normalize_relation_name

try:
    import open3d as o3d
    _HAS_OPEN3D = True
except Exception:
    o3d = None
    _HAS_OPEN3D = False


# ============================================================
# Utilities
# ============================================================

def _pcd_to_numpy(pcd: Any) -> Optional[np.ndarray]:
    """
    Convert point cloud-like input to numpy array of shape (N, 3).
    Supports:
      - None
      - torch.Tensor
      - numpy.ndarray
      - list-like
      - open3d.geometry.PointCloud
    """
    if pcd is None:
        return None

    # PyTorch tensor
    if hasattr(pcd, "detach") and hasattr(pcd, "cpu"):
        arr = pcd.detach().cpu().numpy()
        arr = np.asarray(arr, dtype=float)
        return _ensure_nx3(arr)

    # Open3D point cloud
    if _HAS_OPEN3D and isinstance(pcd, o3d.geometry.PointCloud):
        arr = np.asarray(pcd.points, dtype=float)
        return _ensure_nx3(arr)

    arr = np.asarray(pcd, dtype=float)
    return _ensure_nx3(arr)


def _ensure_nx3(arr: np.ndarray) -> Optional[np.ndarray]:
    if arr is None:
        return None
    if arr.size == 0:
        return np.empty((0, 3), dtype=float)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Point cloud must have shape (N, 3), got {arr.shape}")
    return arr


def _to_numpy_1d(x: Any, expected_len: Optional[int] = None) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.asarray(x, dtype=float).reshape(-1)
    if expected_len is not None and len(arr) != expected_len:
        raise ValueError(f"Expected length {expected_len}, got {len(arr)}")
    return arr


def _safe_norm(v: np.ndarray, eps: float = 1e-8) -> float:
    n = float(np.linalg.norm(v))
    return 0.0 if n < eps else n


def get_iou(box_a: Iterable[float], box_b: Iterable[float]) -> Tuple[float, float]:
    """
    2D IoU for boxes in [x1, y1, x2, y2] format.
    Returns:
        iou, intersection_area
    """
    a = np.asarray(box_a, dtype=float).reshape(4)
    b = np.asarray(box_b, dtype=float).reshape(4)

    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    iw = max(ix2 - ix1, 0.0)
    ih = max(iy2 - iy1, 0.0)
    inter = iw * ih

    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0, 0.0
    return inter / union, inter


def get_node_dist(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """
    Minimum distance between two point clouds.
    Uses Open3D if available, otherwise uses a numpy chunked fallback.
    """
    pts_a = _ensure_nx3(np.asarray(pts_a, dtype=float))
    pts_b = _ensure_nx3(np.asarray(pts_b, dtype=float))

    if pts_a is None or pts_b is None or len(pts_a) == 0 or len(pts_b) == 0:
        return float("inf")

    if _HAS_OPEN3D:
        pcd_a = o3d.geometry.PointCloud()
        pcd_a.points = o3d.utility.Vector3dVector(pts_a)
        pcd_b = o3d.geometry.PointCloud()
        pcd_b.points = o3d.utility.Vector3dVector(pts_b)
        dists = np.asarray(pcd_a.compute_point_cloud_distance(pcd_b), dtype=float)
        return float(dists.min()) if len(dists) > 0 else float("inf")

    # Numpy fallback, chunked to avoid huge memory spikes
    min_dist = float("inf")
    chunk = 512
    for i in range(0, len(pts_a), chunk):
        a_chunk = pts_a[i:i + chunk]  # (Ca, 3)
        diff = a_chunk[:, None, :] - pts_b[None, :, :]  # (Ca, Cb, 3)
        dist = np.linalg.norm(diff, axis=-1)
        local_min = float(dist.min())
        if local_min < min_dist:
            min_dist = local_min
    return min_dist


def _compute_aabb_from_points(points: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    points = _ensure_nx3(points)
    if points is None or len(points) == 0:
        return None
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    return mins, maxs


def _compute_aabb_from_corner_pts(corner_pts: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if corner_pts is None:
        return None
    arr = np.asarray(corner_pts, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"corner_pts must have shape (N, 3), got {arr.shape}")
    if len(arr) == 0:
        return None
    mins = np.min(arr, axis=0)
    maxs = np.max(arr, axis=0)
    return mins, maxs


def _aabb_volume(aabb: Tuple[np.ndarray, np.ndarray]) -> float:
    mins, maxs = aabb
    size = np.maximum(maxs - mins, 0.0)
    return float(size[0] * size[1] * size[2])


def _aabb_intersection(aabb_a: Tuple[np.ndarray, np.ndarray],
                       aabb_b: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    a_min, a_max = aabb_a
    b_min, b_max = aabb_b
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    return inter_min, inter_max


def _aabb_intersection_volume(aabb_a: Tuple[np.ndarray, np.ndarray],
                              aabb_b: Tuple[np.ndarray, np.ndarray]) -> float:
    inter_min, inter_max = _aabb_intersection(aabb_a, aabb_b)
    size = np.maximum(inter_max - inter_min, 0.0)
    return float(size[0] * size[1] * size[2])


def _horizontal_overlap_ratio_xy(aabb_upper: Tuple[np.ndarray, np.ndarray],
                                 aabb_lower: Tuple[np.ndarray, np.ndarray]) -> float:
    """
    XY plane overlap ratio normalized by upper object's XY footprint.
    Here y is vertical in many robotics conventions? No.
    For this code we follow original convention:
      x: left/right
      y: up/down
      z: front/back
    So horizontal plane is x-z plane, not x-y.
    """
    u_min, u_max = aabb_upper
    l_min, l_max = aabb_lower

    inter_x = max(min(u_max[0], l_max[0]) - max(u_min[0], l_min[0]), 0.0)
    inter_z = max(min(u_max[2], l_max[2]) - max(u_min[2], l_min[2]), 0.0)
    inter = inter_x * inter_z

    upper_area = max(u_max[0] - u_min[0], 0.0) * max(u_max[2] - u_min[2], 0.0)
    if upper_area <= 0:
        return 0.0
    return float(inter / upper_area)


def _aabb_center(aabb: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mins, maxs = aabb
    return (mins + maxs) / 2.0


def _aabb_size(aabb: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mins, maxs = aabb
    return np.maximum(maxs - mins, 0.0)


# ============================================================
# Config
# ============================================================

@dataclass
class SceneGraphConfig:
    # Distance thresholds (meters)
    contact_distance: float = 0.01
    close_distance: float = 0.06
    support_gap_tol: float = 0.025
    inside_gap_tol: float = 0.01

    # Overlap / ratio thresholds
    on_top_overlap_ratio: float = 0.35
    on_top_min_center_dy: float = 0.008
    inside_overlap_ratio: float = 0.85
    intersect_volume_ratio: float = 0.18

    # Direction thresholds
    direction_y_thresh: float = 0.75
    direction_x_thresh: float = 0.75
    direction_z_thresh: float = 0.8

    # Occlusion thresholds
    occlude_ratio_thresh: float = 0.5
    depth_front_ratio_thresh: float = 0.9

    # Numerical stability
    eps: float = 1e-8

    # Prefer stable relation priority
    relation_priority: Tuple[str, ...] = (
        "inside",
        "on",
        "intersecting",
        "adjacent_to",
        "above",
        "below",
        "on_the_right_of",
        "on_the_left_of",
        "blocking",
    )


# ============================================================
# Core data structures
# ============================================================

@dataclass(eq=False)
class Node:
    name: str
    object_id: Optional[str] = None
    pos3d: Optional[Iterable[float]] = None
    corner_pts: Optional[Any] = None
    bbox2d: Optional[Iterable[float]] = None
    pcd: Optional[Any] = None
    depth: Optional[Iterable[float]] = None
    global_node: bool = False
    states: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self._pos3d = _to_numpy_1d(self.pos3d, expected_len=3) if self.pos3d is not None else None
        self._bbox2d = _to_numpy_1d(self.bbox2d, expected_len=4) if self.bbox2d is not None else None
        self._pcd = _pcd_to_numpy(self.pcd) if self.pcd is not None else None
        self._depth = np.asarray(self.depth, dtype=float).reshape(-1) if self.depth is not None else None
        self.attributes = dict(self.attributes or {})

    @property
    def pos3d_np(self) -> Optional[np.ndarray]:
        return self._pos3d

    @property
    def bbox2d_np(self) -> Optional[np.ndarray]:
        return self._bbox2d

    @property
    def pcd_np(self) -> Optional[np.ndarray]:
        return self._pcd

    @property
    def depth_np(self) -> Optional[np.ndarray]:
        return self._depth

    def uid(self) -> str:
        return str(self.object_id) if self.object_id is not None else self.display_name()

    def display_name(self) -> str:
        if self.states:
            return f"{self.name} ({', '.join(self.states)})"
        return self.name

    def add_state(self, state: str) -> None:
        if state not in self.states:
            self.states.append(state)

    def clear_states(self) -> None:
        self.states.clear()

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def get_attribute(self, key: str, default: Any = None) -> Any:
        return self.attributes.get(key, default)

    def get_aabb(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        # Prefer pcd-derived AABB if available because it is usually more faithful.
        if self.pcd_np is not None and len(self.pcd_np) > 0:
            return _compute_aabb_from_points(self.pcd_np)
        if self.corner_pts is not None:
            return _compute_aabb_from_corner_pts(self.corner_pts)
        return None

    def get_center(self) -> Optional[np.ndarray]:
        if self.pos3d_np is not None:
            return self.pos3d_np
        aabb = self.get_aabb()
        if aabb is not None:
            return _aabb_center(aabb)
        return None

    def __hash__(self) -> int:
        return hash(self.uid())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return self.uid() == other.uid()

    def __str__(self) -> str:
        return self.uid()


@dataclass(frozen=True)
class Edge:
    start: Node
    end: Node
    edge_type: str

    def __str__(self) -> str:
        return f"{self.start.uid()}->{self.edge_type}->{self.end.uid()}"


# ============================================================
# SceneGraph
# ============================================================

class SceneGraph:
    def __init__(self,
                 event: Any = None,
                 task: Any = None,
                 config: Optional[SceneGraphConfig] = None):
        self.event = event
        self.task = task
        self.config = config or SceneGraphConfig()

        self.nodes: List[Node] = []
        self._node_map: Dict[str, Node] = {}
        self.edges: Dict[Tuple[str, str], Edge] = {}

    # ----------------------------
    # Node management
    # ----------------------------
    def add_node_wo_edge(self, node: Node) -> Node:
        self._register_node(node)
        return node

    def add_node(self, node: Node) -> Node:
        """
        Add a node and infer relations with all existing nodes.
        """
        if node.uid() in self._node_map:
            # Replace the node object but keep graph consistent
            self._replace_node(node)
            return node

        existing_nodes = list(self.nodes)
        self._register_node(node)

        for other in existing_nodes:
            self._infer_pairwise_relations(other, node)

        return node

    def remove_node(self, node_or_uid: Any) -> None:
        uid = node_or_uid if isinstance(node_or_uid, str) else node_or_uid.uid()
        if uid not in self._node_map:
            return

        self.nodes = [n for n in self.nodes if n.uid() != uid]
        self._node_map.pop(uid, None)

        to_remove = [k for k, e in self.edges.items() if e.start.uid() == uid or e.end.uid() == uid]
        for k in to_remove:
            self.edges.pop(k, None)

    def clear_edges(self) -> None:
        self.edges.clear()

    def rebuild_edges(self) -> None:
        """
        Recompute all relations from scratch.
        Useful if node geometry was updated after insertion.
        """
        self.clear_edges()
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                self._infer_pairwise_relations(self.nodes[i], self.nodes[j])

    def _register_node(self, node: Node) -> None:
        if node.uid() in self._node_map:
            raise ValueError(f"Duplicate node uid: {node.uid()}")
        self.nodes.append(node)
        self._node_map[node.uid()] = node

    def _replace_node(self, new_node: Node) -> None:
        uid = new_node.uid()

        idx = None
        for i, n in enumerate(self.nodes):
            if n.uid() == uid:
                idx = i
                break

        if idx is not None:
            self.nodes[idx] = new_node
        self._node_map[uid] = new_node

        # Update existing edges that reference old node
        new_edges: Dict[Tuple[str, str], Edge] = {}
        for key, edge in self.edges.items():
            start = new_node if edge.start.uid() == uid else edge.start
            end = new_node if edge.end.uid() == uid else edge.end
            new_edges[(start.uid(), end.uid())] = Edge(start, end, edge.edge_type)
        self.edges = new_edges

        # Rebuild to refresh geometry-based relations
        self.rebuild_edges()

    # ----------------------------
    # Edge management
    # ----------------------------
    def get_edge(self, start: Node, end: Node) -> Optional[Edge]:
        if start is None or end is None:
            return None
        return self.edges.get((start.uid(), end.uid()))

    def remove_edge(self, start: Node, end: Node) -> None:
        if start is None or end is None:
            return
        self.edges.pop((start.uid(), end.uid()), None)

    def add_edge(self, start: Node, end: Node, edge_type: Optional[str] = None) -> None:
        """
        Backward-compatible API:
        - add_edge(a, b): infer relation automatically
        - add_edge(a, b, "on"): add explicit relation
        """
        if start is None or end is None:
            return
        if start.uid() == end.uid():
            return

        if edge_type is None:
            rel_ab, rel_ba = self._classify_pair(start, end)

            if rel_ab is not None:
                self.edges[(start.uid(), end.uid())] = Edge(start, end, normalize_relation_name(rel_ab))
            else:
                self.edges.pop((start.uid(), end.uid()), None)

            if rel_ba is not None:
                self.edges[(end.uid(), start.uid())] = Edge(end, start, normalize_relation_name(rel_ba))
            else:
                self.edges.pop((end.uid(), start.uid()), None)
            return

        self.edges[(start.uid(), end.uid())] = Edge(start, end, normalize_relation_name(edge_type))

    def infer_edge(self, a: Node, b: Node) -> None:
        self.add_edge(a, b, edge_type=None)

    def _is_aligned(self, a: Node, b: Node) -> bool:
        """
        Lightweight alignment check for block-building scenes.

        Heuristic:
        - similar height (y)
        - similar depth / front-back (z)
        - not too far apart in x
        """
        cfg = self.config

        center_a = a.get_center()
        center_b = b.get_center()
        aabb_a = a.get_aabb()
        aabb_b = b.get_aabb()

        if center_a is None or center_b is None or aabb_a is None or aabb_b is None:
            return False

        dy = abs(float(center_a[1] - center_b[1]))
        dz = abs(float(center_a[2] - center_b[2]))
        dx = abs(float(center_a[0] - center_b[0]))

        size_a = aabb_a[1] - aabb_a[0]
        size_b = aabb_b[1] - aabb_b[0]

        avg_x = max(0.5 * (float(size_a[0]) + float(size_b[0])), cfg.eps)
        avg_y = max(0.5 * (float(size_a[1]) + float(size_b[1])), cfg.eps)
        avg_z = max(0.5 * (float(size_a[2]) + float(size_b[2])), cfg.eps)

        align_y_thresh = max(0.35 * avg_y, cfg.contact_distance * 1.5)
        align_z_thresh = max(0.35 * avg_z, cfg.contact_distance * 1.5)
        max_lateral_gap = max(2.5 * avg_x, cfg.contact_distance * 4.0)

        return dy <= align_y_thresh and dz <= align_z_thresh and dx <= max_lateral_gap
    # ----------------------------
    # Relation inference
    # ----------------------------
    def _infer_pairwise_relations(self, node_a: Node, node_b: Node) -> None:
        """
        Infer relation(s) for an unordered pair.
        We compute best relations in both directions where appropriate.
        """
        if node_a is None or node_b is None:
            return
        if node_a.uid() == node_b.uid():
            return

        relation_ab, relation_ba = self._classify_pair(node_a, node_b)

        if relation_ab is not None:
            self.add_edge(node_a, node_b, relation_ab)
        else:
            self.remove_edge(node_a, node_b)

        if relation_ba is not None:
            self.add_edge(node_b, node_a, relation_ba)
        else:
            self.remove_edge(node_b, node_a)

    def _classify_pair(self, a: Node, b: Node) -> Tuple[Optional[str], Optional[str]]:
        """
        Return (a->b relation, b->a relation).

        Block-building oriented priority:
            on/supports > intersecting > adjacent_to > aligned_with > directional

        Notes:
        - "inside" is intentionally weakened for block scenes because it often
        becomes noisy with AABB / point-cloud overlap.
        - stacking relation is normalized to "on".
        - reverse support relation is "supports".
        """
        cfg = self.config

        aabb_a = a.get_aabb()
        aabb_b = b.get_aabb()
        center_a = a.get_center()
        center_b = b.get_center()

        if aabb_a is None or aabb_b is None or center_a is None or center_b is None:
            return None, None

        pos_delta = center_b - center_a
        norm = _safe_norm(pos_delta, cfg.eps)
        direction = pos_delta / norm if norm > 0 else np.zeros(3, dtype=float)

        dist = self._estimate_node_distance(a, b)

        # 1) Core stacking / support relation for block construction.
        on_ab = self._is_on_top_of(a, b, dist)   # a on b
        on_ba = self._is_on_top_of(b, a, dist)   # b on a

        if on_ab and not on_ba:
            return "on", "supports"
        if on_ba and not on_ab:
            return "supports", "on"

        # 2) Strong geometric conflict / collision.
        if self._is_intersecting(a, b):
            return "intersecting", "intersecting"

        # 3) Side contact / local adjacency.
        touching = dist <= cfg.contact_distance
        if touching:
            return "adjacent_to", "adjacent_to"

        # 4) Coarse structural alignment, useful for bridge/arch/block layouts.
        if self._is_aligned(a, b):
            return "aligned_with", "aligned_with"

        # 5) Keep simple directional relations as fallback.
        directional_ab, directional_ba = self._directional_relation(a, b, direction, dist)
        if directional_ab or directional_ba:
            return directional_ab, directional_ba

        # 6) Optional weak fallback: inside only for clearly container-like cases.
        # For block-building objects, inside is usually noisy, so place it last.
        inside_ab = self._is_inside(a, b)
        inside_ba = self._is_inside(b, a)
        if inside_ab and not inside_ba:
            return "inside", None
        if inside_ba and not inside_ab:
            return None, "inside"

        return None, None

    def _estimate_node_distance(self, a: Node, b: Node) -> float:
        """
        Prefer point-cloud distance; fall back to AABB gap distance.
        """
        if a.pcd_np is not None and b.pcd_np is not None and len(a.pcd_np) > 0 and len(b.pcd_np) > 0:
            return get_node_dist(a.pcd_np, b.pcd_np)

        aabb_a = a.get_aabb()
        aabb_b = b.get_aabb()
        if aabb_a is None or aabb_b is None:
            return float("inf")

        a_min, a_max = aabb_a
        b_min, b_max = aabb_b

        gap = np.maximum(np.maximum(a_min - b_max, b_min - a_max), 0.0)
        return float(np.linalg.norm(gap))

    def _is_inside(self, inner: Node, outer: Node) -> bool:
        cfg = self.config
        aabb_inner = inner.get_aabb()
        aabb_outer = outer.get_aabb()
        if aabb_inner is None or aabb_outer is None:
            return False

        i_min, i_max = aabb_inner
        o_min, o_max = aabb_outer

        # Loose containment with tolerance
        if np.any(i_min < (o_min - cfg.inside_gap_tol)):
            return False
        if np.any(i_max > (o_max + cfg.inside_gap_tol)):
            return False

        inter_vol = _aabb_intersection_volume(aabb_inner, aabb_outer)
        inner_vol = _aabb_volume(aabb_inner)
        if inner_vol <= cfg.eps:
            return False

        ratio = inter_vol / inner_vol
        return ratio >= cfg.inside_overlap_ratio

    def _is_on_top_of(self, upper: Node, lower: Node, dist: float) -> bool:
        cfg = self.config
        aabb_u = upper.get_aabb()
        aabb_l = lower.get_aabb()
        center_u = upper.get_center()
        center_l = lower.get_center()

        if aabb_u is None or aabb_l is None:
            return False
        if center_u is None or center_l is None:
            return False

        u_min, u_max = aabb_u
        l_min, l_max = aabb_l

        # 中心高度至少应大体满足上层高于下层
        center_dy = float(center_u[1] - center_l[1])
        if center_dy < cfg.on_top_min_center_dy:
            return False

        # upper 底面 - lower 顶面
        vertical_gap = float(u_min[1] - l_max[1])

        # 放宽一点“轻微压入”的容忍度
        if vertical_gap < -(cfg.support_gap_tol * 1.25):
            return False

        # ON 的水平门：x 方向按上层宽度归一化；z 方向不再要求区间相交。
        inter_x = max(min(u_max[0], l_max[0]) - max(u_min[0], l_min[0]), 0.0)
        upper_x_extent = float(u_max[0] - u_min[0])
        if upper_x_extent <= cfg.eps:
            return False

        x_overlap_ratio = float(inter_x / upper_x_extent)
        if x_overlap_ratio < max(0.60, float(cfg.on_top_overlap_ratio)):
            return False

        # 使用检测端当前帧点云；30 点与 global detector 的默认最低点数一致。
        upper_points = upper.pcd_np
        lower_points = lower.pcd_np
        if upper_points is None or lower_points is None:
            return False
        if len(upper_points) < 30 or len(lower_points) < 30:
            return False
        if not np.all(np.isfinite(upper_points)) or not np.all(np.isfinite(lower_points)):
            return False

        # 分位区间减小单个深度异常点对 z 距离的影响。
        upper_z_low, upper_z_high = np.percentile(upper_points[:, 2], [1.0, 99.0])
        lower_z_low, lower_z_high = np.percentile(lower_points[:, 2], [1.0, 99.0])
        z_interval_gap = float(max(
            upper_z_low - lower_z_high,
            lower_z_low - upper_z_high,
            0.0,
        ))
        if z_interval_gap > cfg.support_gap_tol:
            return False

        # 明确接触/支撑
        if vertical_gap <= cfg.support_gap_tol * 1.2:
            return True

        # 几何距离仍很近，也视作 on
        if dist <= max(cfg.close_distance * 0.8, cfg.support_gap_tol * 2.0):
            return True

        return False

    def _is_intersecting(self, a: Node, b: Node) -> bool:
        cfg = self.config
        aabb_a = a.get_aabb()
        aabb_b = b.get_aabb()
        center_a = a.get_center()
        center_b = b.get_center()

        if aabb_a is None or aabb_b is None:
            return False

        inter_min, inter_max = _aabb_intersection(aabb_a, aabb_b)
        inter_size = np.maximum(inter_max - inter_min, 0.0)
        inter_vol = float(inter_size[0] * inter_size[1] * inter_size[2])
        if inter_vol <= cfg.eps:
            return False

        # 堆叠豁免：
        # 横向重叠明显、竖直相交很薄时，更像 on 的测量误差，不像真正碰撞
        if center_a is not None and center_b is not None:
            upper, lower = (a, b) if center_a[1] >= center_b[1] else (b, a)
            upper_aabb = upper.get_aabb()
            lower_aabb = lower.get_aabb()
            upper_center = upper.get_center()
            lower_center = lower.get_center()

            if upper_aabb is not None and lower_aabb is not None and upper_center is not None and lower_center is not None:
                overlap_ratio = _horizontal_overlap_ratio_xy(upper_aabb, lower_aabb)
                y_overlap = float(inter_size[1])
                center_dy = float(upper_center[1] - lower_center[1])

                if (
                    overlap_ratio >= max(0.50, cfg.on_top_overlap_ratio * 0.9)
                    and y_overlap <= max(cfg.support_gap_tol * 1.8, 0.012)
                    and center_dy >= cfg.on_top_min_center_dy
                ):
                    return False

        vol_a = _aabb_volume(aabb_a)
        vol_b = _aabb_volume(aabb_b)
        denom = min(vol_a, vol_b)
        if denom <= cfg.eps:
            return False

        return (inter_vol / denom) >= cfg.intersect_volume_ratio

    def _directional_relation(self,
                              a: Node,
                              b: Node,
                              direction_ab: np.ndarray,
                              dist: float) -> Tuple[Optional[str], Optional[str]]:
        """
        direction_ab is unit vector from a -> b.
        """
        cfg = self.config

        if dist > cfg.close_distance:
            return None, None

        # Ignore purely global nodes for weak directional relations
        if a.global_node or b.global_node:
            return None, None

        x, y, z = direction_ab

        # Vertical relations
        if abs(y) >= cfg.direction_y_thresh:
            if y > 0:
                # b above a
                return "below", "above"
            else:
                return "above", "below"

        # Left-right relations
        if abs(x) >= cfg.direction_x_thresh:
            if x > 0:
                # b on right of a
                return "on the left of", "on the right of"
            else:
                return "on the right of", "on the left of"

        return None, None

    def _blocking_relation(self,
                           a: Node,
                           b: Node,
                           direction_ab: np.ndarray) -> Tuple[Optional[str], Optional[str]]:
        """
        Approximate front-back occlusion relation using:
          - z-direction separation
          - 2D bbox overlap
          - depth ordering
        Returns:
          a->b, b->a
        Meaning:
          if returns ("blocking", None), then a blocks b
        """
        cfg = self.config

        if a.bbox2d_np is None or b.bbox2d_np is None:
            return None, None
        if a.depth_np is None or b.depth_np is None or len(a.depth_np) == 0 or len(b.depth_np) == 0:
            return None, None

        z = direction_ab[2]
        if abs(z) < cfg.direction_z_thresh:
            return None, None

        _, inter = get_iou(a.bbox2d_np, b.bbox2d_np)
        area_a = max(a.bbox2d_np[2] - a.bbox2d_np[0], 0.0) * max(a.bbox2d_np[3] - a.bbox2d_np[1], 0.0)
        area_b = max(b.bbox2d_np[2] - b.bbox2d_np[0], 0.0) * max(b.bbox2d_np[3] - b.bbox2d_np[1], 0.0)
        if area_a <= cfg.eps or area_b <= cfg.eps:
            return None, None

        occ_a_by_b = inter / area_a
        occ_b_by_a = inter / area_b

        # Use robust depth statistic instead of min-depth
        a_med = float(np.median(a.depth_np))
        b_med = float(np.median(b.depth_np))

        # Smaller depth => closer to camera
        if occ_b_by_a >= cfg.occlude_ratio_thresh and a_med < b_med:
            return "blocking", None
        if occ_a_by_b >= cfg.occlude_ratio_thresh and b_med < a_med:
            return None, "blocking"

        return None, None

    # ----------------------------
    # Debug / query helpers
    # ----------------------------
    def get_relations(self) -> List[Edge]:
        return list(self.edges.values())

    def get_node(self, node_uid: str) -> Optional[Node]:
        return self._node_map.get(node_uid)

    def get_neighbors(self, node: Node) -> List[Edge]:
        uid = node.uid()
        return [e for e in self.edges.values() if e.start.uid() == uid or e.end.uid() == uid]

    def has_relation(self, start: Node, end: Node, edge_type: Optional[str] = None) -> bool:
        edge = self.get_edge(start, end)
        if edge is None:
            return False
        if edge_type is None:
            return True
        return edge.edge_type == edge_type

    def relation_summary(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for e in self.edges.values():
            out[e.edge_type] = out.get(e.edge_type, 0) + 1
        return out

    # ----------------------------
    # Equality / display
    # ----------------------------
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SceneGraph):
            return False

        self_nodes = set(n.uid() for n in self.nodes)
        other_nodes = set(n.uid() for n in other.nodes)
        if self_nodes != other_nodes:
            return False

        self_edges = set((e.start.uid(), e.end.uid(), e.edge_type) for e in self.edges.values())
        other_edges = set((e.start.uid(), e.end.uid(), e.edge_type) for e in other.edges.values())
        return self_edges == other_edges

    def __str__(self) -> str:
        lines = ["[Nodes]"]
        for node in sorted(set(self.nodes), key=lambda n: n.uid()):
            node_line = node.uid()
            if node.display_name() != node.uid():
                node_line = f"{node.uid()}: {node.display_name()}"
            lines.append(node_line)

        lines.append("")
        lines.append("[Edges]")

        # Print stable relations first, then lexicographically
        priority = {name: i for i, name in enumerate(self.config.relation_priority)}
        edges_sorted = sorted(
            self.edges.values(),
            key=lambda e: (priority.get(e.edge_type, 999), e.start.uid(), e.end.uid(), e.edge_type)
        )
        for edge in edges_sorted:
            lines.append(str(edge))

        return "\n".join(lines)

# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    # Simple demo with AABB via corner points
    # Convention:
    #   x: left/right
    #   y: up/down
    #   z: front/back

    def make_box(center, size):
        cx, cy, cz = center
        sx, sy, sz = size
        hx, hy, hz = sx / 2, sy / 2, sz / 2
        return np.array([
            [cx - hx, cy - hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz + hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz - hz],
            [cx + hx, cy + hy, cz + hz],
        ], dtype=float)

    bottom = Node(
        name="cube",
        object_id="cube_1",
        pos3d=[0.0, 0.02, 0.0],
        corner_pts=make_box(center=[0.0, 0.02, 0.0], size=[0.04, 0.04, 0.04]),
    )

    top = Node(
        name="cube",
        object_id="cube_2",
        pos3d=[0.0, 0.06, 0.0],
        corner_pts=make_box(center=[0.0, 0.06, 0.0], size=[0.04, 0.04, 0.04]),
    )

    right = Node(
        name="cube",
        object_id="cube_3",
        pos3d=[0.08, 0.02, 0.0],
        corner_pts=make_box(center=[0.08, 0.02, 0.0], size=[0.04, 0.04, 0.04]),
    )

    sg = SceneGraph()
    sg.add_node(bottom)
    sg.add_node(top)
    sg.add_node(right)

    print(sg)
