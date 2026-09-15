#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, Iterable, List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))



from scene_graph_system.scene_graph.scene_graph import Node, SceneGraph


SCENE_GRAPH_SNAPSHOT_SCHEMA_VERSION = 2

_OMITTED_ATTRIBUTE_KEYS = {
    "points_camera",
    "points_table",
    "points_sg",
    "depth_values",
    "mask",
    "mask_points",
    "pcd",
    "pcd_np",
}


def _to_sec(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if hasattr(value, "to_sec"):
            return float(value.to_sec())
        return float(value)
    except Exception:
        return None


def _json_safe(value: Any, *, max_list_items: int = 128) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None

    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist(), max_list_items=max_list_items)
        except Exception:
            return None

    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text in _OMITTED_ATTRIBUTE_KEYS:
                continue
            out[key_text] = _json_safe(item, max_list_items=max_list_items)
        return out

    if isinstance(value, (list, tuple)):
        out = []
        for item in list(value)[:max_list_items]:
            out.append(_json_safe(item, max_list_items=max_list_items))
        return out

    return str(value)


def _node_vector(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    safe = _json_safe(value)
    if not isinstance(safe, list):
        return None
    try:
        return [float(v) for v in safe]
    except Exception:
        return None


def _node_to_dict(node: Node) -> Dict[str, Any]:
    attrs = _json_safe(dict(getattr(node, "attributes", {}) or {}))
    if not isinstance(attrs, dict):
        attrs = {}

    pos3d = None
    if getattr(node, "pos3d_np", None) is not None:
        pos3d = _node_vector(node.pos3d_np)
    if pos3d is None:
        pos3d = _node_vector(attrs.get("center_sg"))

    corner_pts = None
    if getattr(node, "corner_pts", None) is not None:
        corner_pts = _json_safe(node.corner_pts)
    else:
        corner_pts = attrs.get("corners_sg")

    bbox2d = None
    if getattr(node, "bbox2d_np", None) is not None:
        bbox2d = _node_vector(node.bbox2d_np)
    else:
        bbox2d = attrs.get("bbox2d")

    return {
        "id": str(node.uid()),
        "class_name": str(getattr(node, "name", "")),
        "states": [str(state) for state in list(getattr(node, "states", []) or [])],
        "global_node": bool(getattr(node, "global_node", False)),
        "pos3d": pos3d,
        "corner_pts": corner_pts,
        "bbox2d": bbox2d,
        "attributes": attrs,
    }


def _relations_to_dicts(graph: SceneGraph) -> List[Dict[str, Any]]:
    get_relations = getattr(graph, "get_relations", None)
    relations = get_relations() if callable(get_relations) else []
    out = []
    for edge in relations or []:
        try:
            out.append(
                {
                    "subject_id": str(edge.start.uid()),
                    "relation": str(edge.edge_type),
                    "object_id": str(edge.end.uid()),
                }
            )
        except Exception:
            continue
    return out


def scene_graph_to_snapshot_dict(
    graph: SceneGraph,
    snapshot: Optional[Dict[str, Any]] = None,
    *,
    runtime_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    snapshot = dict(snapshot or {})
    graph_stamp_sec = _to_sec(snapshot.get("graph_stamp_sec"))
    if graph_stamp_sec is None:
        graph_stamp_sec = _to_sec(snapshot.get("graph_stamp"))

    capture_stamp = _to_sec(snapshot.get("capture_stamp"))
    if capture_stamp is None:
        capture_stamp = _to_sec(snapshot.get("detection_stamp"))
    capture_lower_bound = _to_sec(snapshot.get("capture_lower_bound"))
    capture_upper_bound = _to_sec(snapshot.get("capture_upper_bound"))

    context_fields = {
        "run_id": str(snapshot.get("run_id", "") or "").strip(),
        "attempt_id": snapshot.get("attempt_id"),
        "stage_seq": snapshot.get("stage_seq"),
        "stage_name": str(snapshot.get("stage_name", "") or "").strip(),
        "barrier_id": str(snapshot.get("barrier_id", "") or "").strip(),
    }
    transaction_eligible = bool(snapshot.get("transaction_eligible", False))
    transaction_eligible = bool(
        transaction_eligible
        and snapshot.get("scene_version") is not None
        and snapshot.get("source_frame_seq") is not None
        and capture_stamp is not None
        and capture_lower_bound is not None
        and capture_upper_bound is not None
        and context_fields["run_id"]
        and context_fields["stage_name"]
        and context_fields["barrier_id"]
        and context_fields["attempt_id"] is not None
        and context_fields["stage_seq"] is not None
    )

    return {
        "schema_version": SCENE_GRAPH_SNAPSHOT_SCHEMA_VERSION,
        "source": "scene_graph_builder",
        "scene_version": snapshot.get("scene_version"),
        "graph_stamp": graph_stamp_sec,
        "graph_stamp_sec": graph_stamp_sec,
        "capture_stamp": capture_stamp,
        "capture_lower_bound": capture_lower_bound,
        "capture_upper_bound": capture_upper_bound,
        "capture_uncertainty_sec": _to_sec(snapshot.get("capture_uncertainty_sec")),
        "capture_timestamp_source": str(snapshot.get("capture_timestamp_source", "unknown")),
        "source_timestamp": _to_sec(snapshot.get("source_timestamp")),
        "source_timestamp_domain": str(snapshot.get("source_timestamp_domain", "unknown")),
        "source_frame_seq": snapshot.get("source_frame_seq"),
        "publish_stamp": _to_sec(snapshot.get("publish_stamp")),
        "inference_start_stamp": _to_sec(snapshot.get("inference_start_stamp")),
        "inference_end_stamp": _to_sec(snapshot.get("inference_end_stamp")),
        "detection_stamp": _to_sec(snapshot.get("detection_stamp")) or capture_stamp,
        "detection_frame_index": snapshot.get("detection_frame_index"),
        "detection_age_sec": _to_sec(snapshot.get("detection_age_sec")),
        "detection_receive_age_sec": _to_sec(snapshot.get("detection_receive_age_sec")),
        "detection_receive_time": _to_sec(snapshot.get("detection_receive_time")),
        **context_fields,
        "transaction_eligible": transaction_eligible,
        "gripper_alignment": _json_safe(snapshot.get("gripper_alignment") or {}),
        "nodes": [_node_to_dict(node) for node in list(getattr(graph, "nodes", []) or [])],
        "relations": _relations_to_dicts(graph),
        "runtime_summary": _json_safe(runtime_summary or {}),
    }


def snapshot_dict_to_scene_graph(payload: Dict[str, Any]) -> SceneGraph:
    graph = SceneGraph()
    node_by_id: Dict[str, Node] = {}

    for node_data in list(payload.get("nodes", []) or []):
        if not isinstance(node_data, dict):
            continue
        node_id = str(node_data.get("id", "") or "").strip()
        class_name = str(node_data.get("class_name", "") or "").strip()
        if not node_id or not class_name:
            continue

        attrs = dict(node_data.get("attributes", {}) or {})
        node = Node(
            name=class_name,
            object_id=node_id,
            pos3d=node_data.get("pos3d") or attrs.get("center_sg"),
            corner_pts=node_data.get("corner_pts"),
            bbox2d=node_data.get("bbox2d"),
            pcd=None,
            depth=None,
            global_node=bool(node_data.get("global_node", False)),
            states=[str(state) for state in list(node_data.get("states", []) or [])],
            attributes=attrs,
        )
        graph.add_node_wo_edge(node)
        node_by_id[node_id] = node

    for rel_data in list(payload.get("relations", []) or []):
        if not isinstance(rel_data, dict):
            continue
        subject_id = str(rel_data.get("subject_id", "") or "")
        object_id = str(rel_data.get("object_id", "") or "")
        relation = str(rel_data.get("relation", "") or "")
        if not subject_id or not object_id or not relation:
            continue
        graph.add_edge(node_by_id.get(subject_id), node_by_id.get(object_id), relation)

    return graph


def decode_scene_graph_snapshot(payload: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = dict(payload or {})
    try:
        schema_version = int(snapshot.get("schema_version", 1))
    except (TypeError, ValueError):
        schema_version = 0
    snapshot["schema_version"] = schema_version
    if schema_version != SCENE_GRAPH_SNAPSHOT_SCHEMA_VERSION:
        # Legacy and unknown snapshots remain available for display/replay,
        # but cannot satisfy a transaction barrier.
        snapshot["transaction_eligible"] = False
    snapshot["graph"] = snapshot_dict_to_scene_graph(snapshot)
    return snapshot


def snapshot_to_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def snapshot_from_json(raw: str) -> Dict[str, Any]:
    return decode_scene_graph_snapshot(json.loads(str(raw or "{}")))
