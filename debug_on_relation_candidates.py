#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import Dict, Optional, Tuple

import numpy as np
import rospy
from std_msgs.msg import String

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))



from scene_graph_system.scene_graph.scene_graph import SceneGraphConfig, _horizontal_overlap_ratio_xy
from scene_graph_system.scene_graph.scene_graph_snapshot_codec import snapshot_from_json


DEFAULT_SCENE_GRAPH_TOPIC = "/scene_graph/current"


def _node_id(node) -> str:
    try:
        return str(node.uid())
    except Exception:
        return str(getattr(node, "object_id", "") or getattr(node, "id", ""))


def _node_class(node) -> str:
    return str(getattr(node, "name", "") or getattr(node, "class_name", ""))


def _aabb_gap_distance(aabb_a, aabb_b) -> float:
    a_min, a_max = aabb_a
    b_min, b_max = aabb_b
    gap = np.maximum(np.maximum(a_min - b_max, b_min - a_max), 0.0)
    return float(np.linalg.norm(gap))


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _param_list(name: str, default):
    raw = rospy.get_param(name, default)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [v.strip() for v in text.split(",") if v.strip()]
    if isinstance(raw, (list, tuple, set)):
        return list(raw)
    return [raw]


class OnRelationCandidateDebugger:
    def __init__(self):
        self.scene_graph_topic = str(
            rospy.get_param("~scene_graph_topic", DEFAULT_SCENE_GRAPH_TOPIC)
        ).strip()
        self.only_suspicious = bool(rospy.get_param("~only_suspicious", True))
        self.print_all_pairs = bool(rospy.get_param("~print_all_pairs", False))
        self.min_overlap_to_log = max(
            0.0, float(rospy.get_param("~min_overlap_to_log", 0.05))
        )
        self.max_abs_vertical_gap_to_log = max(
            0.0, float(rospy.get_param("~max_abs_vertical_gap_to_log", 0.08))
        )
        self.max_pairs_per_frame = max(
            1, int(rospy.get_param("~max_pairs_per_frame", 30))
        )
        self.ignore_classes = set(
            str(v).strip()
            for v in _param_list("~ignore_classes", ["gripper"])
            if str(v).strip()
        )
        self.focus_relations = set(
            str(v).strip()
            for v in _param_list(
                "~focus_relations",
                [
                    "adjacent_to",
                    "aligned_with",
                    "on_the_left_of",
                    "on_the_right_of",
                    "above",
                    "below",
                ],
            )
            if str(v).strip()
        )

        self.config = SceneGraphConfig()
        self.last_graph_stamp = None

        rospy.Subscriber(
            self.scene_graph_topic,
            String,
            self._callback,
            queue_size=1,
        )
        rospy.logwarn(
            "[ON_DIAG] started: topic=%s only_suspicious=%s min_overlap=%.3f "
            "max_abs_vertical_gap=%.3f max_pairs=%d ignore_classes=%s focus_relations=%s",
            self.scene_graph_topic,
            str(self.only_suspicious),
            self.min_overlap_to_log,
            self.max_abs_vertical_gap_to_log,
            self.max_pairs_per_frame,
            sorted(self.ignore_classes),
            sorted(self.focus_relations),
        )

    def _relation_map(self, graph) -> Dict[Tuple[str, str], str]:
        out = {}
        get_relations = getattr(graph, "get_relations", None)
        relations = get_relations() if callable(get_relations) else []
        for edge in relations or []:
            try:
                out[(str(edge.start.uid()), str(edge.end.uid()))] = str(edge.edge_type)
            except Exception:
                continue
        return out

    def _diagnose_on(self, upper, lower, relation_map):
        cfg = self.config
        upper_aabb = upper.get_aabb()
        lower_aabb = lower.get_aabb()
        upper_center = upper.get_center()
        lower_center = lower.get_center()

        if upper_aabb is None or lower_aabb is None:
            return None
        if upper_center is None or lower_center is None:
            return None

        upper_id = _node_id(upper)
        lower_id = _node_id(lower)
        upper_cls = _node_class(upper)
        lower_cls = _node_class(lower)

        if upper_cls in self.ignore_classes or lower_cls in self.ignore_classes:
            return None

        u_min, _u_max = upper_aabb
        _l_min, l_max = lower_aabb

        center_dy = float(upper_center[1] - lower_center[1])
        vertical_gap = float(u_min[1] - l_max[1])
        overlap_ratio = float(_horizontal_overlap_ratio_xy(upper_aabb, lower_aabb))
        aabb_gap_dist = _aabb_gap_distance(upper_aabb, lower_aabb)

        relation_ul = relation_map.get((upper_id, lower_id), "")
        relation_lu = relation_map.get((lower_id, upper_id), "")

        if center_dy < cfg.on_top_min_center_dy:
            reason = "center_dy_too_small"
            would_be_on = False
        elif vertical_gap < -(cfg.support_gap_tol * 1.25):
            reason = "vertical_gap_too_negative"
            would_be_on = False
        elif overlap_ratio < cfg.on_top_overlap_ratio:
            reason = "overlap_too_small"
            would_be_on = False
        elif vertical_gap <= cfg.support_gap_tol * 1.2:
            reason = "would_be_on_by_vertical_gap"
            would_be_on = True
        elif aabb_gap_dist <= max(cfg.close_distance * 0.8, cfg.support_gap_tol * 2.0):
            reason = "would_be_on_by_aabb_gap"
            would_be_on = True
        else:
            reason = "distance_too_large"
            would_be_on = False

        return {
            "upper_id": upper_id,
            "upper_class": upper_cls,
            "lower_id": lower_id,
            "lower_class": lower_cls,
            "relation_ul": relation_ul,
            "relation_lu": relation_lu,
            "center_dy": center_dy,
            "vertical_gap": vertical_gap,
            "overlap_ratio": overlap_ratio,
            "aabb_gap_dist": aabb_gap_dist,
            "reason": reason,
            "would_be_on": would_be_on,
        }

    def _should_log(self, diag) -> bool:
        if self.print_all_pairs:
            return True

        relation_ul = str(diag.get("relation_ul", ""))
        relation_lu = str(diag.get("relation_lu", ""))
        overlap_ratio = float(diag.get("overlap_ratio", 0.0))
        vertical_gap = float(diag.get("vertical_gap", float("inf")))
        center_dy = float(diag.get("center_dy", float("-inf")))

        if relation_ul == "on" or relation_lu == "supports":
            return False

        relation_focused = (
            relation_ul in self.focus_relations
            or relation_lu in self.focus_relations
        )
        geometry_near_on = (
            center_dy >= -self.config.support_gap_tol
            and abs(vertical_gap) <= self.max_abs_vertical_gap_to_log
            and overlap_ratio >= self.min_overlap_to_log
        )

        if not self.only_suspicious:
            return relation_focused or geometry_near_on

        return relation_focused and geometry_near_on

    def _callback(self, msg: String):
        raw = str(msg.data or "").strip()
        if not raw:
            return

        try:
            snapshot = snapshot_from_json(raw)
        except Exception as exc:
            rospy.logwarn_throttle(
                2.0,
                "[ON_DIAG] failed to parse scene graph snapshot: %s",
                str(exc),
            )
            return

        graph = snapshot.get("graph")
        if graph is None:
            return

        graph_stamp = snapshot.get("graph_stamp_sec", snapshot.get("graph_stamp"))
        graph_stamp_float = _to_float(graph_stamp)
        if graph_stamp_float is not None and graph_stamp_float == self.last_graph_stamp:
            return
        self.last_graph_stamp = graph_stamp_float

        relation_map = self._relation_map(graph)
        nodes = list(getattr(graph, "nodes", []) or [])
        diagnostics = []
        reason_counts = Counter()

        for upper in nodes:
            for lower in nodes:
                if upper is lower:
                    continue
                diag = self._diagnose_on(upper, lower, relation_map)
                if diag is None:
                    continue
                reason_counts[str(diag["reason"])] += 1
                if self._should_log(diag):
                    diagnostics.append(diag)

        diagnostics.sort(
            key=lambda item: (
                0 if item["relation_ul"] == "adjacent_to" or item["relation_lu"] == "adjacent_to" else 1,
                -float(item["overlap_ratio"]),
                abs(float(item["vertical_gap"])),
            )
        )

        frame_index = snapshot.get("detection_frame_index")
        det_stamp = snapshot.get("detection_stamp")
        rospy.logwarn(
            "[ON_DIAG] frame graph_stamp=%s det_stamp=%s frame_index=%s "
            "nodes=%d candidates=%d reason_counts=%s",
            str(graph_stamp),
            str(det_stamp),
            str(frame_index),
            len(nodes),
            len(diagnostics),
            json.dumps(dict(reason_counts), ensure_ascii=False, sort_keys=True),
        )

        for diag in diagnostics[: self.max_pairs_per_frame]:
            rospy.logwarn(
                "[ON_DIAG] reason=%s would_be_on=%s upper=%s(%s) lower=%s(%s) "
                "rel_ul=%s rel_lu=%s center_dy=%.4f vertical_gap=%.4f "
                "overlap=%.3f aabb_gap=%.4f thresholds={support_gap_tol:%.4f,"
                "on_top_min_center_dy:%.4f,on_top_overlap_ratio:%.3f,"
                "contact_distance:%.4f,close_distance:%.4f}",
                str(diag["reason"]),
                str(diag["would_be_on"]),
                str(diag["upper_id"]),
                str(diag["upper_class"]),
                str(diag["lower_id"]),
                str(diag["lower_class"]),
                str(diag["relation_ul"]),
                str(diag["relation_lu"]),
                float(diag["center_dy"]),
                float(diag["vertical_gap"]),
                float(diag["overlap_ratio"]),
                float(diag["aabb_gap_dist"]),
                self.config.support_gap_tol,
                self.config.on_top_min_center_dy,
                self.config.on_top_overlap_ratio,
                self.config.contact_distance,
                self.config.close_distance,
            )


def main():
    rospy.init_node("debug_on_relation_candidates", anonymous=True)
    OnRelationCandidateDebugger()
    rospy.spin()


if __name__ == "__main__":
    main()
