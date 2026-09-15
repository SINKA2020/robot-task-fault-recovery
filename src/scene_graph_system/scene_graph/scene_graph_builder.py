#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rospy
from std_msgs.msg import String
from scene_graph_system.msg import GripperState

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))



from scene_graph_system.robot.gripper_state_tracker import DEFAULT_GRIPPER_NODE_ID
from scene_graph_system.robot.gripper_state_protocol import GripperSample, GripperSampleHistory
from scene_graph_system.scene_graph.object_id_utils import canonical_object_id, normalize_object_class_name, object_class_from_id
from scene_graph_system.scene_graph.scene_graph import Node, SceneGraph
from scene_graph_system.scene_graph.scene_graph_snapshot_codec import scene_graph_to_snapshot_dict, snapshot_to_json


DEFAULT_SEG_DETECTION_TOPIC = "/global_camera/seg_detections"
DEFAULT_SCENE_GRAPH_TOPIC = "/scene_graph/current"
DEFAULT_SCENE_GRAPH_STATUS_TOPIC = "/scene_graph/status"
DEFAULT_GRIPPER_STATE_TOPIC = "/gripper/state"

class GlobalDetectionTrackManager:
    """
    外部固定相机检测结果的轻量跨帧跟踪器。

    注意：
    - 普通积木目标走这个跟踪器。
    - gripper 不走这个跟踪器，而是在 SceneGraphBuilder 中固定合并为 object_id='gripper'。
    """

    def __init__(self, max_missed_frames: int = 3, dist_gate: float = 0.08):
        self.max_missed_frames = int(max_missed_frames)
        self.dist_gate = float(dist_gate)

        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.next_track_id = 1
        self.next_instance_index_by_class: Dict[str, int] = {}

    def decay_all(self):
        stale_ids = []

        for track_id, track in self.tracks.items():
            track["missed"] = int(track.get("missed", 0)) + 1
            if track["missed"] > self.max_missed_frames:
                stale_ids.append(track_id)

        for track_id in stale_ids:
            self.tracks.pop(track_id, None)

    def assign(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.decay_all()

        assigned_track_ids = set()
        out: List[Dict[str, Any]] = []

        detections_sorted = sorted(
            list(detections or []),
            key=lambda det: (
                str(det.get("class_name", "")),
                float((det.get("center_sg") or [0.0, 0.0, 0.0])[0]),
                float((det.get("center_sg") or [0.0, 0.0, 0.0])[1]),
                float((det.get("center_sg") or [0.0, 0.0, 0.0])[2]),
            ),
        )

        for det in detections_sorted:
            det = dict(det)

            class_name = normalize_object_class_name(
                det.get("class_name", det.get("raw_label", "object"))
            )
            explicit_object_id = det.get("object_id")

            if explicit_object_id:
                object_id = canonical_object_id(str(explicit_object_id))
                det["class_name"] = object_class_from_id(object_id)
                det["object_id"] = object_id
                det["track_id"] = None
                out.append(det)
                continue

            center = self._get_center(det)
            if center is None:
                object_id = self._new_object_id(class_name)
                det["class_name"] = class_name
                det["object_id"] = object_id
                det["track_id"] = None
                out.append(det)
                continue

            best_track_id = None
            best_dist = float("inf")

            for track_id, track in self.tracks.items():
                if track_id in assigned_track_ids:
                    continue

                if track.get("class_name") != class_name:
                    continue

                old_center = np.asarray(track.get("center_sg"), dtype=np.float32).reshape(3)
                dist = float(np.linalg.norm(center - old_center))

                if dist < best_dist:
                    best_dist = dist
                    best_track_id = track_id

            if best_track_id is not None and best_dist <= self.dist_gate:
                track = self.tracks[best_track_id]
                track["center_sg"] = center
                track["missed"] = 0
                assigned_track_ids.add(best_track_id)

                det["class_name"] = class_name
                det["object_id"] = str(track["object_id"])
                det["track_id"] = int(best_track_id)
                out.append(det)
                continue

            object_id = self._new_object_id(class_name)
            track_id = self.next_track_id
            self.next_track_id += 1

            self.tracks[track_id] = {
                "class_name": class_name,
                "object_id": object_id,
                "center_sg": center,
                "missed": 0,
            }
            assigned_track_ids.add(track_id)

            det["class_name"] = class_name
            det["object_id"] = object_id
            det["track_id"] = int(track_id)
            out.append(det)

        return out

    def _get_center(self, det: Dict[str, Any]) -> Optional[np.ndarray]:
        center = det.get("center_sg")
        if center is None:
            return None

        arr = np.asarray(center, dtype=np.float32).reshape(-1)
        if len(arr) != 3:
            return None
        if not np.all(np.isfinite(arr)):
            return None

        return arr.reshape(3)

    def _new_object_id(self, class_name: str) -> str:
        class_name = normalize_object_class_name(class_name)
        next_index = self.next_instance_index_by_class.get(class_name, 1)
        self.next_instance_index_by_class[class_name] = next_index + 1
        return f"{class_name}_{next_index}"


class SceneGraphBuilder:
    """
    外部固定相机版本的 SceneGraphBuilder。

    本版本对 gripper 做了融合：
    - 独立监控话题提供真实夹爪状态：open / closed / holding 等；
    - 外部相机检测 gripper 几何：pos3d / pcd / bbox / corners；
    - 二者合并为同一个 object_id='gripper' 节点；
    - 不再出现 gripper_1 / gripper_42 这种普通检测节点。

    本版本还支持搭建区域 ROI：
    - 从 seg_yolov11.py 的 detection 中读取 in_build_region / outside_build_region；
    - 给 Node.states 加入 in_build_region / outside_build_region；
    - gripper、被抓取物、以及与 gripper intersecting/attached/inside 的节点，
      自动加入 exclude_from_build_validation，避免动作执行过程中误参与搭建结构验证。
    """

    def __init__(
        self,
        init_node=True,
        node_name="scene_graph_builder",
        gripper_state_provider=None,
        gripper_node_id=DEFAULT_GRIPPER_NODE_ID,
        publish_scene_graph: Optional[bool] = None,
    ):
        if init_node and not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True)

        self.detection_topic = str(
            rospy.get_param(
                "~detection_topic",
                rospy.get_param("~seg_detection_topic", DEFAULT_SEG_DETECTION_TOPIC),
            )
        ).strip()

        self.latest_scene_graph = None
        self.latest_scene_graph_stamp = None
        self.latest_detection_payload = None
        self.latest_scene_graph_snapshot = None
        self.latest_dropped_detection_info = None
        self.latest_scene_graph_status = None

        self.last_process_time = 0.0
        # Detection already controls sampling frequency. A second 1 Hz gate
        # can alias with detector jitter and halve the effective graph rate.
        self.output_interval = max(0.0, float(rospy.get_param("~output_interval", 0.0)))
        if publish_scene_graph is None:
            publish_scene_graph = bool(rospy.get_param("~publish_scene_graph", False))
        self.publish_scene_graph = bool(publish_scene_graph)
        self.scene_graph_topic = str(
            rospy.get_param("~scene_graph_topic", DEFAULT_SCENE_GRAPH_TOPIC)
        ).strip()
        self.scene_graph_status_topic = str(
            rospy.get_param("~scene_graph_status_topic", DEFAULT_SCENE_GRAPH_STATUS_TOPIC)
        ).strip()
        self.scene_graph_publish_queue_size = max(
            1,
            int(rospy.get_param("~scene_graph_publish_queue_size", 5)),
        )
        self.scene_graph_status_hz = max(
            0.1,
            float(rospy.get_param("~scene_graph_status_hz", 1.0)),
        )
        self.detection_queue_size = max(1, int(rospy.get_param("~detection_queue_size", 1)))
        self.drop_stale_detections = bool(rospy.get_param("~drop_stale_detections", True))
        self.drop_out_of_order_detections = bool(rospy.get_param("~drop_out_of_order_detections", True))
        self.scene_graph_max_detection_age_sec = max(
            0.0,
            float(rospy.get_param("~scene_graph_max_detection_age_sec", 3.0)),
        )
        self.detection_stamp_tolerance_sec = max(
            0.0,
            float(rospy.get_param("~detection_stamp_tolerance_sec", 0.2)),
        )
        self._last_processed_detection_stamp = None
        self._last_processed_detection_frame_index = None
        self.scene_version = 0

        self.gripper_state_provider = None
        if gripper_state_provider is not None:
            rospy.logwarn(
                "gripper_state_provider injection is deprecated and ignored; "
                "start gripper_state_monitor and use ~gripper_state_source:=topic"
            )
        self.gripper_node_id = gripper_node_id
        self.last_grasped_object_id = None

        self.gripper_state_source = str(
            rospy.get_param("~gripper_state_source", "topic")
        ).strip().lower()
        if self.gripper_state_source not in {"topic", "off"}:
            raise ValueError("unsupported ~gripper_state_source=%s" % self.gripper_state_source)
        self.gripper_state_topic = str(
            rospy.get_param("~gripper_state_topic", DEFAULT_GRIPPER_STATE_TOPIC)
        ).strip()
        self.gripper_history = GripperSampleHistory(
            retention_sec=max(1.0, float(rospy.get_param("~gripper_history_sec", 10.0))),
            max_samples=max(10, int(rospy.get_param("~gripper_history_max_samples", 500))),
        )
        self.gripper_history_lock = threading.Lock()
        self.gripper_camera_max_skew_sec = max(
            0.0, float(rospy.get_param("~gripper_camera_max_skew_sec", 0.20))
        )
        self.gripper_min_stable_count = max(
            1, int(rospy.get_param("~gripper_min_stable_count", 3))
        )
        self.latest_gripper_alignment = {
            "valid": False,
            "reason": "not_evaluated",
            "skew_sec": None,
        }

        # 当前帧按相机采集时间匹配到的 gripper 状态节点。
        # 如果外部相机检测到 gripper，后续会把检测几何与该状态节点合并。
        self._provider_gripper_node = None

        self.frame_index = 0
        self.max_missed_frames = int(rospy.get_param("~max_missed_frames", 3))
        self.dist_gate = float(rospy.get_param("~dist_gate", 0.08))

        self.publish_empty_graph_when_no_detection = bool(
            rospy.get_param("~publish_empty_graph_when_no_detection", True)
        )

        # 默认打印完整场景图，输出格式为：
        # SceneGraph: nodes=..., edges=...
        # [NODES]
        # [EDGES]
        self.print_scene_graph = bool(rospy.get_param("~print_scene_graph", False))

        # 是否额外打印每个 Added node 的调试信息。
        # 默认 False，避免干扰主输出格式。
        self.log_added_nodes = bool(rospy.get_param("~log_added_nodes", False))

        self.track_manager = GlobalDetectionTrackManager(
            max_missed_frames=self.max_missed_frames,
            dist_gate=self.dist_gate,
        )

        self.relation_mode = str(
            rospy.get_param("~relation_mode", "full")
        ).strip().lower()
        if self.relation_mode not in {"full", "gripper_only", "off"}:
            raise ValueError(
                "unsupported ~relation_mode=%s (expected full, gripper_only, or off)"
                % self.relation_mode
            )

        self.mode = "rule_only"

        rospy.loginfo("System mode: %s", self.mode)
        rospy.loginfo("Scene graph relation mode: %s", self.relation_mode)
        rospy.loginfo("SceneGraphBuilder uses external global camera detections.")
        rospy.loginfo("Subscribing detection topic: %s", self.detection_topic)
        rospy.loginfo(
            "Gripper merge enabled: geometry + time-aligned state source=%s -> object_id=%s",
            self.gripper_state_source,
            self.gripper_node_id,
        )
        rospy.loginfo(
            "Region states enabled: build-region states plus in_sort_region_* / outside_sort_regions"
        )

        self.scene_graph_pub = None
        self.scene_graph_status_pub = None
        self.scene_graph_status_timer = None
        if self.publish_scene_graph:
            self.scene_graph_pub = rospy.Publisher(
                self.scene_graph_topic,
                String,
                queue_size=self.scene_graph_publish_queue_size,
                latch=False,
            )
            self.scene_graph_status_pub = rospy.Publisher(
                self.scene_graph_status_topic,
                String,
                queue_size=1,
                latch=False,
            )
            self.scene_graph_status_timer = rospy.Timer(
                rospy.Duration(1.0 / self.scene_graph_status_hz),
                self._scene_graph_status_timer_cb,
            )
            rospy.loginfo(
                "SceneGraph publishing enabled: graph_topic=%s status_topic=%s",
                self.scene_graph_topic,
                self.scene_graph_status_topic,
            )

        rospy.Subscriber(
            self.detection_topic,
            String,
            self.detections_cb,
            queue_size=self.detection_queue_size,
        )
        self.gripper_state_sub = None
        if self.gripper_state_source == "topic":
            self.gripper_state_sub = rospy.Subscriber(
                self.gripper_state_topic,
                GripperState,
                self._gripper_state_callback,
                queue_size=200,
            )

        rospy.loginfo("SceneGraphBuilder initialized. Waiting for seg_yolov11 messages...")

    # ============================================================
    # state helpers
    # ============================================================

    def _dedupe_states(self, states):
        """
        去重并保持顺序。
        """
        out = []
        seen = set()

        for state in states or []:
            if state is None:
                continue

            s = str(state).strip()
            if not s:
                continue

            if s not in seen:
                out.append(s)
                seen.add(s)

        return out

    def _add_state_once(self, node: Node, state: str):
        if node is None:
            return

        if not hasattr(node, "states") or node.states is None:
            node.states = []

        if state not in node.states:
            node.states.append(state)

    def _extract_build_region_states_from_detection(self, det: Dict[str, Any]) -> List[str]:
        """
        从 seg_yolov11.py 的 detection 中提取搭建区域状态。

        支持三种来源：
        1. det["states"]
        2. det["in_build_region"]
        3. det["attributes"]["build_region"]["in_build_region"]

        最终只保留：
          in_build_region
          或
          outside_build_region
        二者不会同时存在。
        """
        states = []
        sorting_states = []

        # 1) det["states"]
        raw_states = det.get("states", []) or []
        if isinstance(raw_states, (list, tuple)):
            for s in raw_states:
                s = str(s).strip()
                if s in ("in_build_region", "outside_build_region"):
                    states.append(s)
                elif s.startswith("in_sort_region_") or s == "outside_sort_regions":
                    sorting_states.append(s)

        # Explicit sorting output wins over any stale state in det["states"].
        explicit_sorting_state = str(
            det.get("sorting_region_state", "") or ""
        ).strip()
        if (
            explicit_sorting_state.startswith("in_sort_region_")
            or explicit_sorting_state == "outside_sort_regions"
        ):
            sorting_states = [explicit_sorting_state]

        # 2) det["in_build_region"]
        if "in_build_region" in det:
            if bool(det.get("in_build_region", False)):
                states.append("in_build_region")
            else:
                states.append("outside_build_region")

        # 3) attributes.build_region.in_build_region
        attrs = dict(det.get("attributes", {}) or {})
        build_region = dict(attrs.get("build_region", {}) or {})

        if "in_build_region" in build_region:
            if bool(build_region.get("in_build_region", False)):
                states.append("in_build_region")
            else:
                states.append("outside_build_region")

        states = self._dedupe_states(states)

        # 冲突处理：如果同时出现，优先 in_build_region
        if "in_build_region" in states:
            states = [s for s in states if s != "outside_build_region"]
        elif "outside_build_region" in states:
            states = [s for s in states if s != "in_build_region"]

        # Sorting areas are mutually exclusive. Keep one inside state, or the
        # outside state when no inside state is present.
        sorting_states = self._dedupe_states(sorting_states)
        inside_sorting_states = [
            state for state in sorting_states
            if state.startswith("in_sort_region_")
        ]
        if inside_sorting_states:
            if len(inside_sorting_states) > 1:
                rospy.logwarn_throttle(
                    2.0,
                    "Detection contains multiple sorting-region states %s; using %s",
                    str(inside_sorting_states),
                    inside_sorting_states[0],
                )
            sorting_states = [inside_sorting_states[0]]
        elif "outside_sort_regions" in sorting_states:
            sorting_states = ["outside_sort_regions"]

        return states + sorting_states

    def _is_node_excluded_from_build_validation(self, node: Node) -> bool:
        if node is None:
            return True

        states = set(getattr(node, "states", []) or [])

        if node.name == "gripper":
            return True

        if "exclude_from_build_validation" in states:
            return True

        if "currently_manipulated" in states:
            return True

        if "grasped" in states:
            return True

        return False

    # ============================================================
    # time-aligned gripper state topic
    # ============================================================

    def _gripper_state_callback(self, msg: GripperState):
        try:
            sample = GripperSample(
                monitor_instance_id=str(msg.monitor_instance_id or ""),
                sample_seq=int(msg.sample_seq),
                sample_start_stamp=float(msg.sample_start_stamp.to_sec()),
                sample_end_stamp=float(msg.sample_end_stamp.to_sec()),
                states=tuple(msg.states or []),
                valid=bool(msg.valid),
                stable_count=int(msg.stable_count),
                holding_latched=bool(msg.holding_latched),
                command_id=str(msg.command_id or ""),
                command=str(msg.command or ""),
            )
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Invalid gripper state sample: %s", str(exc))
            return

        with self.gripper_history_lock:
            accepted = self.gripper_history.add(sample)
        if not accepted:
            rospy.logwarn_throttle(
                2.0,
                "Dropped duplicate/out-of-order gripper sample: instance=%s seq=%s",
                sample.monitor_instance_id,
                str(sample.sample_seq),
            )

    def _time_aligned_gripper_node(self, detection_metadata: Optional[Dict[str, Any]]):
        metadata = dict(detection_metadata or {})
        lower = self._coerce_float(metadata.get("capture_lower_bound"))
        upper = self._coerce_float(metadata.get("capture_upper_bound"))
        capture_stamp = self._coerce_float(metadata.get("capture_stamp"))
        if lower is None or upper is None:
            lower = capture_stamp
            upper = capture_stamp

        if lower is None or upper is None:
            self.latest_gripper_alignment = {
                "valid": False,
                "reason": "missing_capture_interval",
                "skew_sec": None,
            }
            return Node(
                name="gripper",
                object_id=self.gripper_node_id,
                global_node=True,
                states=["unknown"],
                attributes={
                    "gripper_state_source": "topic_aligned",
                    "gripper_alignment_valid": False,
                    "gripper_alignment_reason": "missing_capture_interval",
                },
            )

        with self.gripper_history_lock:
            sample, skew = self.gripper_history.match_capture_interval(
                lower,
                upper,
                max_skew_sec=self.gripper_camera_max_skew_sec,
            )

        if sample is None:
            reason = "no_aligned_gripper_sample"
            self.latest_gripper_alignment = {
                "valid": False,
                "reason": reason,
                "skew_sec": skew,
            }
            return Node(
                name="gripper",
                object_id=self.gripper_node_id,
                global_node=True,
                states=["unknown"],
                attributes={
                    "gripper_state_source": "topic_aligned",
                    "gripper_alignment_valid": False,
                    "gripper_alignment_reason": reason,
                    "gripper_alignment_skew_sec": skew,
                    "gripper_camera_max_skew_sec": self.gripper_camera_max_skew_sec,
                },
            )

        valid = bool(
            sample.valid
            and int(sample.stable_count) >= int(self.gripper_min_stable_count)
        )
        states = list(sample.states if valid else ("unknown",))
        if valid:
            reason = ""
        elif not sample.valid:
            reason = "aligned_sample_invalid"
        else:
            reason = "aligned_sample_unstable"
        attributes = {
            "gripper_state_source": "topic_aligned",
            "gripper_alignment_valid": valid,
            "gripper_alignment_reason": reason,
            "gripper_alignment_skew_sec": float(skew or 0.0),
            "gripper_camera_max_skew_sec": self.gripper_camera_max_skew_sec,
            "gripper_monitor_instance_id": sample.monitor_instance_id,
            "gripper_sample_seq": int(sample.sample_seq),
            "gripper_sample_start_stamp": float(sample.sample_start_stamp),
            "gripper_sample_end_stamp": float(sample.sample_end_stamp),
            "gripper_sample_stable_count": int(sample.stable_count),
            "gripper_min_stable_count": int(self.gripper_min_stable_count),
            "gripper_holding_latched": bool(sample.holding_latched),
            "gripper_command_id": str(sample.command_id or ""),
            "gripper_command": str(sample.command or ""),
        }
        self.latest_gripper_alignment = {
            "valid": valid,
            "reason": reason,
            "skew_sec": float(skew or 0.0),
            "sample_seq": int(sample.sample_seq),
            "sample_start_stamp": float(sample.sample_start_stamp),
            "sample_end_stamp": float(sample.sample_end_stamp),
            "monitor_instance_id": sample.monitor_instance_id,
        }
        return Node(
            name="gripper",
            object_id=self.gripper_node_id,
            global_node=True,
            states=states,
            attributes=attributes,
        )

    def _new_scene_graph(self, detection_metadata: Optional[Dict[str, Any]] = None):
        sg = SceneGraph(event=None, task=None)

        if self.gripper_state_source == "topic":
            self._provider_gripper_node = self._time_aligned_gripper_node(detection_metadata)
        else:
            self._provider_gripper_node = None
        if self._provider_gripper_node is not None:
            # 先加入按采集时间匹配的状态节点。
            # 如果本帧外部相机也检测到 gripper，后续会用同一个 object_id="gripper"
            # 的检测节点替换它，从而同时保留几何信息和真实夹爪状态。
            self._add_node_for_relation_mode(sg, self._provider_gripper_node)

        return sg

    def _add_node_for_relation_mode(self, sg, node: Node) -> None:
        if self.relation_mode == "full":
            sg.add_node(node)
            return

        # add_node_wo_edge has no replacement path. A visual gripper node can
        # replace the state-only gripper node with the same uid.
        existing = sg.get_node(node.uid())
        if existing is not None:
            sg.remove_node(node.uid())
        sg.add_node_wo_edge(node)

    def _finalize_relations_for_mode(self, sg) -> None:
        if self.relation_mode != "gripper_only":
            return

        gripper_node = self._get_gripper_node(sg)
        if gripper_node is None:
            return

        for node in list(getattr(sg, "nodes", []) or []):
            if node.uid() == gripper_node.uid():
                continue
            sg.infer_edge(gripper_node, node)

    def _get_gripper_node(self, sg):
        for node in sg.nodes:
            if node.uid() == self.gripper_node_id or node.name == "gripper":
                return node
        return None

    # ============================================================
    # callback
    # ============================================================

    @staticmethod
    def _now_sec() -> float:
        try:
            return float(rospy.Time.now().to_sec())
        except Exception:
            return float(time.time())

    @staticmethod
    def _coerce_float(value) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_int(value) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _make_detection_metadata(self, payload: Dict[str, Any], receive_time_sec: float) -> Dict[str, Any]:
        capture_stamp = self._coerce_float(payload.get("capture_stamp"))
        if capture_stamp is None:
            capture_stamp = self._coerce_float(payload.get("stamp"))
        source_frame_seq = self._coerce_int(payload.get("source_frame_seq"))
        if source_frame_seq is None:
            source_frame_seq = self._coerce_int(payload.get("frame_index"))
        detection_age_sec = None
        if capture_stamp is not None:
            detection_age_sec = float(receive_time_sec) - float(capture_stamp)

        run_id = str(payload.get("run_id", "") or "").strip()
        stage_name = str(payload.get("stage_name", "") or "").strip()
        barrier_id = str(payload.get("barrier_id", "") or "").strip()
        attempt_id = self._coerce_int(payload.get("attempt_id"))
        stage_seq = self._coerce_int(payload.get("stage_seq"))
        not_before = self._coerce_float(payload.get("not_before"))
        capture_lower_bound = self._coerce_float(payload.get("capture_lower_bound"))
        capture_upper_bound = self._coerce_float(payload.get("capture_upper_bound"))
        context_complete = bool(
            run_id
            and stage_name
            and barrier_id
            and attempt_id is not None
            and attempt_id > 0
            and stage_seq is not None
            and stage_seq > 0
            and not_before is not None
        )
        transaction_eligible = bool(
            context_complete
            and source_frame_seq is not None
            and capture_lower_bound is not None
            and capture_upper_bound is not None
            and capture_lower_bound >= not_before
        )

        return {
            "detection_stamp": capture_stamp,
            "detection_frame_index": source_frame_seq,
            "capture_stamp": capture_stamp,
            "capture_lower_bound": capture_lower_bound,
            "capture_upper_bound": capture_upper_bound,
            "capture_uncertainty_sec": self._coerce_float(payload.get("capture_uncertainty_sec")),
            "capture_timestamp_source": str(payload.get("capture_timestamp_source", "unknown")),
            "source_timestamp": self._coerce_float(payload.get("source_timestamp")),
            "source_timestamp_domain": str(payload.get("source_timestamp_domain", "unknown")),
            "source_frame_seq": source_frame_seq,
            "publish_stamp": self._coerce_float(payload.get("publish_stamp")),
            "inference_start_stamp": self._coerce_float(payload.get("inference_start_stamp")),
            "inference_end_stamp": self._coerce_float(payload.get("inference_end_stamp")),
            "run_id": run_id,
            "attempt_id": attempt_id,
            "stage_seq": stage_seq,
            "stage_name": stage_name,
            "barrier_id": barrier_id,
            "not_before": not_before,
            "transaction_eligible": transaction_eligible,
            "detection_receive_time": float(receive_time_sec),
            "detection_age_sec": detection_age_sec,
        }

    def _should_accept_detection_metadata(self, metadata: Dict[str, Any]) -> Tuple[bool, str]:
        if not self.drop_stale_detections:
            return True, ""

        stamp = metadata.get("detection_stamp")
        frame_index = metadata.get("detection_frame_index")
        age_sec = metadata.get("detection_age_sec")
        tol = float(self.detection_stamp_tolerance_sec)

        if stamp is None or float(stamp) <= 0.0:
            return False, "missing_detection_stamp"

        if age_sec is not None and float(age_sec) > float(self.scene_graph_max_detection_age_sec) + tol:
            return False, "stale_detection_age"

        last_stamp = self._last_processed_detection_stamp
        if self.drop_out_of_order_detections and last_stamp is not None:
            if float(stamp) < float(last_stamp) - tol:
                return False, "out_of_order_detection_stamp"

            last_frame = self._last_processed_detection_frame_index
            if (
                frame_index is not None
                and last_frame is not None
                and int(frame_index) == int(last_frame)
                and abs(float(stamp) - float(last_stamp)) <= 1e-6
            ):
                return False, "duplicate_detection_frame"

        return True, ""

    def _record_dropped_detection(self, metadata: Dict[str, Any], reason: str) -> None:
        self.latest_dropped_detection_info = dict(metadata or {})
        self.latest_dropped_detection_info["reason"] = str(reason)
        rospy.logwarn_throttle(
            2.0,
            "SceneGraphBuilder dropped detection: reason=%s stamp=%s frame_index=%s age=%s max_age=%.3f",
            str(reason),
            str((metadata or {}).get("detection_stamp")),
            str((metadata or {}).get("detection_frame_index")),
            str((metadata or {}).get("detection_age_sec")),
            float(self.scene_graph_max_detection_age_sec),
        )
        self._publish_scene_graph_status()

    def _make_scene_graph_status_dict(self) -> Dict[str, Any]:
        now_sec = self._now_sec()
        snapshot = dict(self.latest_scene_graph_snapshot or {})
        dropped = dict(self.latest_dropped_detection_info or {})

        detection_stamp = snapshot.get("detection_stamp")
        detection_age_sec = None
        if detection_stamp is not None:
            try:
                detection_age_sec = now_sec - float(detection_stamp)
            except Exception:
                detection_age_sec = snapshot.get("detection_age_sec")

        max_age = float(self.scene_graph_max_detection_age_sec)
        tol = float(self.detection_stamp_tolerance_sec)
        fresh = bool(
            self.latest_scene_graph is not None
            and detection_stamp is not None
            and detection_age_sec is not None
            and float(detection_age_sec) <= max_age + tol
        )

        return {
            "source": "scene_graph_builder",
            "has_graph": self.latest_scene_graph is not None,
            "fresh": fresh,
            "now": now_sec,
            "scene_graph_topic": self.scene_graph_topic,
            "detection_topic": self.detection_topic,
            "scene_version": snapshot.get("scene_version"),
            "graph_stamp": snapshot.get("graph_stamp_sec"),
            "capture_stamp": snapshot.get("capture_stamp"),
            "capture_lower_bound": snapshot.get("capture_lower_bound"),
            "capture_upper_bound": snapshot.get("capture_upper_bound"),
            "source_frame_seq": snapshot.get("source_frame_seq"),
            "transaction_eligible": bool(snapshot.get("transaction_eligible", False)),
            "gripper_alignment": dict(snapshot.get("gripper_alignment") or {}),
            "barrier_id": snapshot.get("barrier_id", ""),
            "detection_stamp": detection_stamp,
            "detection_frame_index": snapshot.get("detection_frame_index"),
            "detection_age_sec": detection_age_sec,
            "max_detection_age_sec": max_age,
            "last_drop_reason": dropped.get("reason"),
            "last_dropped_detection_stamp": dropped.get("detection_stamp"),
            "last_dropped_detection_frame_index": dropped.get("detection_frame_index"),
            "last_dropped_detection_age_sec": dropped.get("detection_age_sec"),
        }

    def _publish_scene_graph_status(self) -> None:
        if self.scene_graph_status_pub is None:
            return
        payload = self._make_scene_graph_status_dict()
        self.latest_scene_graph_status = payload
        try:
            self.scene_graph_status_pub.publish(
                String(data=json.dumps(payload, ensure_ascii=False))
            )
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Failed to publish scene graph status: %s", str(exc))

    def _scene_graph_status_timer_cb(self, _event) -> None:
        self._publish_scene_graph_status()

    def detections_cb(self, msg: String):
        raw = str(msg.data).strip()
        if not raw:
            return

        try:
            payload = json.loads(raw)
        except Exception as exc:
            rospy.logwarn("Failed to parse seg detection JSON: %s", str(exc))
            return

        receive_time_sec = self._now_sec()
        detection_metadata = self._make_detection_metadata(payload, receive_time_sec)
        accept, reason = self._should_accept_detection_metadata(detection_metadata)
        if not accept:
            self._record_dropped_detection(detection_metadata, reason)
            return

        detections = payload.get("detections", []) or []
        if not isinstance(detections, list):
            rospy.logwarn("Invalid detections payload: detections is not a list")
            return

        if self.output_interval > 0.0 and receive_time_sec - self.last_process_time < self.output_interval:
            return

        self.last_process_time = receive_time_sec
        self.frame_index += 1
        self.latest_detection_payload = payload
        self._last_processed_detection_stamp = detection_metadata.get("detection_stamp")
        self._last_processed_detection_frame_index = detection_metadata.get("detection_frame_index")

        if len(detections) == 0:
            rospy.logdebug("Skipping: no global segmentation detections")

            if self.publish_empty_graph_when_no_detection:
                self.track_manager.decay_all()
                sg = self._new_scene_graph(detection_metadata)
                self._update_node_states_from_scene_graph(sg)
                self._publish_scene_graph(sg, detection_metadata=detection_metadata)
                self._log_scene_graph_output(sg, nodes=[])

            return

        sg = self._new_scene_graph(detection_metadata)
        nodes: List[Node] = []

        detections = self._assign_detection_ids_with_fixed_gripper(detections)

        for det in detections:
            try:
                node = self._node_from_detection(det)
            except Exception as exc:
                rospy.logwarn(
                    "Failed to convert detection to Node: %s det=%s",
                    str(exc),
                    str(det)[:300],
                )
                continue

            nodes.append(node)
            self._add_node_for_relation_mode(sg, node)

            center = None
            if node.pos3d_np is not None:
                center = node.pos3d_np.astype(float).tolist()

            if self.log_added_nodes:
                rospy.loginfo(
                    "Added node for %s: object_id=%s, track=%s, center=%s, points=%d, states=%s",
                    node.name,
                    node.uid(),
                    str(det.get("track_id")),
                    str(center),
                    0 if node.pcd_np is None else len(node.pcd_np),
                    str(list(getattr(node, "states", []) or [])),
                )
            else:
                rospy.logdebug(
                    "Added node for %s: object_id=%s, track=%s, center=%s, points=%d, states=%s",
                    node.name,
                    node.uid(),
                    str(det.get("track_id")),
                    str(center),
                    0 if node.pcd_np is None else len(node.pcd_np),
                    str(list(getattr(node, "states", []) or [])),
                )

        self._finalize_relations_for_mode(sg)
        self._update_node_states_from_scene_graph(sg)
        self._publish_scene_graph(sg, detection_metadata=detection_metadata)
        self._log_scene_graph_output(sg, nodes)

    # ============================================================
    # detection id assignment
    # ============================================================

    def _assign_detection_ids_with_fixed_gripper(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        普通物体继续使用跨帧追踪器分配 ID；
        gripper 检测不进入普通追踪器，固定使用 object_id='gripper'。

        如果一帧中有多个 gripper 检测，只保留置信度/点数最高的一个，避免重复替换。
        """
        normal_detections = []
        gripper_detections = []

        for raw_det in list(detections or []):
            det = dict(raw_det)
            class_name = normalize_object_class_name(
                det.get("class_name", det.get("raw_label", ""))
            )

            if class_name == "gripper":
                det["class_name"] = "gripper"
                det["object_id"] = self.gripper_node_id
                det["track_id"] = "fixed_gripper"
                gripper_detections.append(det)
            else:
                normal_detections.append(det)

        tracked = self.track_manager.assign(normal_detections)

        best_gripper = self._select_best_gripper_detection(gripper_detections)
        if best_gripper is not None:
            # gripper 放在最后加入图。
            # 因为 _new_scene_graph() 已经先加入了时间对齐的 gripper 状态节点；
            # 这里再加入同 uid 的检测 gripper，会触发 SceneGraph 的替换和关系重建。
            tracked.append(best_gripper)

        return tracked

    def _select_best_gripper_detection(self, gripper_detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        如果一帧中检测出多个 gripper，只保留质量最高的一个。
        优先按 score，其次按 point_count。
        """
        if not gripper_detections:
            return None

        def _score(det: Dict[str, Any]) -> Tuple[float, int]:
            raw_score = det.get("score", 0.0)
            try:
                score = float(raw_score)
            except Exception:
                score = 0.0

            point_count = det.get("point_count", None)
            if point_count is None:
                points = det.get("points_sg", []) or []
                point_count = len(points)

            try:
                point_count = int(point_count)
            except Exception:
                point_count = 0

            return score, point_count

        return max(gripper_detections, key=_score)

    # ============================================================
    # detection -> Node
    # ============================================================

    def _node_from_detection(self, det: Dict[str, Any]) -> Node:
        class_name = normalize_object_class_name(
            det.get("class_name", det.get("raw_label", "object"))
        )

        is_gripper_detection = class_name == "gripper"

        if is_gripper_detection:
            object_id = self.gripper_node_id
            class_name = "gripper"
        else:
            object_id = det.get("object_id")
            if object_id:
                object_id = canonical_object_id(str(object_id))
            else:
                object_id = canonical_object_id(class_name)

        center_sg = np.asarray(det.get("center_sg"), dtype=np.float32).reshape(3)

        points_sg_raw = det.get("points_sg", [])
        points_sg = np.asarray(points_sg_raw, dtype=np.float32)

        if points_sg.size == 0:
            points_sg = np.empty((0, 3), dtype=np.float32)
        else:
            points_sg = points_sg.reshape(-1, 3)

        corners_sg = np.asarray(det.get("corners_sg", []), dtype=np.float32)
        if corners_sg.size == 0:
            corners_sg = None
        else:
            corners_sg = corners_sg.reshape(-1, 3)

        bbox2d = det.get("bbox2d", None)
        if bbox2d is not None:
            bbox2d = np.asarray(bbox2d, dtype=np.float32).reshape(4)

        depth_values = det.get("depth_values", [])
        depth_values = np.asarray(depth_values, dtype=np.float32).reshape(-1)

        # -----------------------------
        # gripper 状态 + 几何融合
        # -----------------------------
        provider_states = []
        provider_attributes = {}

        if is_gripper_detection and self._provider_gripper_node is not None:
            provider_states = list(
                getattr(self._provider_gripper_node, "states", []) or []
            )
            provider_attributes = dict(
                getattr(self._provider_gripper_node, "attributes", {}) or {}
            )

        detection_attributes = dict(det.get("attributes", {}) or {})

        attributes = {}
        attributes.update(detection_attributes)
        if is_gripper_detection:
            # The independent, time-aligned monitor is authoritative for all
            # gripper-state evidence. Vision attributes may add geometry
            # metadata but must never overwrite state timestamps or validity.
            attributes.update(provider_attributes)

        build_region_attrs = dict(detection_attributes.get("build_region", {}) or {})
        sorting_region_value = detection_attributes.get("sorting_region", {}) or {}
        sorting_region_attrs = (
            dict(sorting_region_value)
            if isinstance(sorting_region_value, dict)
            else {}
        )

        attributes.update(
            {
                "source": detection_attributes.get("source", "seg_yolov11"),
                "camera_source": "global",
                "detector_type": detection_attributes.get(
                    "detector_type", "yolo_segmentation"
                ),
                "raw_label": det.get("raw_label"),
                "score": det.get("score"),
                "track_id": det.get("track_id"),
                "center_sg": center_sg.astype(float).tolist(),
                "point_count": int(len(points_sg)),

                # ROI / build region debug fields
                "in_build_region": bool(det.get("in_build_region", build_region_attrs.get("in_build_region", False))),
                "build_region_overlap": det.get("build_region_overlap", build_region_attrs.get("overlap_ratio")),
                "center_in_build_region": det.get("center_in_build_region", build_region_attrs.get("center_in_roi")),
                "region_name": det.get("region_name", build_region_attrs.get("region_name")),
                "build_region_center_px": det.get("build_region_center_px", build_region_attrs.get("mask_center_px")),

                # Sorting-region fields describe location only; they remain
                # independent of the detected class.
                "sorting_region": det.get(
                    "sorting_region", sorting_region_attrs.get("region_name")
                ),
                "sorting_region_state": det.get(
                    "sorting_region_state", sorting_region_attrs.get("state")
                ),
                "sorting_region_overlap": det.get(
                    "sorting_region_overlap", sorting_region_attrs.get("overlap_ratio")
                ),
                "sorting_region_overlaps": det.get(
                    "sorting_region_overlaps", sorting_region_attrs.get("overlaps", {})
                ),
            }
        )

        if is_gripper_detection:
            attributes["merged_gripper_node"] = True
            attributes["gripper_geometry_source"] = "seg_yolov11"
            attributes["gripper_state_source"] = (
                provider_attributes.get("gripper_state_source")
                or (self.gripper_state_source if provider_states else "none")
            )

        # -----------------------------
        # 节点状态：
        # 1. 普通物体：保留外部相机 ROI 判断得到的 in_build_region / outside_build_region
        # 2. gripper：保留时间对齐的真实夹爪状态，同时也保留 ROI 状态
        # 3. gripper 永远不参与搭建结构验证
        # -----------------------------
        region_states = self._extract_build_region_states_from_detection(det)

        if is_gripper_detection:
            # gripper 节点只输出真实夹爪状态，例如 open / closed / holding。
            # 不输出 in_build_region / outside_build_region /
            # exclude_from_build_validation / currently_manipulated。
            states = self._dedupe_states(list(provider_states or []))
        else:
            states = self._dedupe_states(region_states)

        return Node(
            name=class_name,
            object_id=object_id,
            pos3d=center_sg,
            pcd=points_sg,
            corner_pts=corners_sg,
            bbox2d=bbox2d,
            depth=depth_values,
            global_node=bool(is_gripper_detection),
            states=states,
            attributes=attributes,
        )

    # ============================================================
    # node states
    # ============================================================

    def _update_node_states_from_scene_graph(self, sg):
        """
        根据当前帧场景图关系更新动态状态。

        重点：
        - placed / grasped / currently_manipulated / exclude_from_build_validation 每帧重算；
        - gripper 永远不参与搭建结构验证；
        - 被抓取物不参与搭建结构验证；
        - 与 gripper 存在 attached / intersecting / inside 关系的节点也不参与搭建结构验证。
        """
        gripper_node = self._get_gripper_node(sg)
        gripper_states = set(getattr(gripper_node, "states", []) or [])

        # 每帧重新推理这些动态状态，避免上一帧残留。
        # 注意：in_build_region / outside_build_region 不在这里删除，它们来自 detection。
        dynamic_state_names = {
            "placed",
            "grasped",
            "exclude_from_build_validation",
            "currently_manipulated",
        }

        for node in sg.nodes:
            existing_states = list(getattr(node, "states", []) or [])
            node.states = [
                state for state in existing_states
                if state not in dynamic_state_names
            ]

        # gripper 永远不参与搭建结构验证。
        # if gripper_node is not None:
        #     self._add_state_once(gripper_node, "exclude_from_build_validation")
        #     self._add_state_once(gripper_node, "currently_manipulated")

        gripper_evidence_unknown = gripper_node is None or "unknown" in gripper_states

        # 1) placed:
        # 当前节点在某条 "on" 关系中作为上层物体，且夹爪不再 holding。
        for edge in sg.get_relations():
            relation = getattr(edge, "edge_type", None)
            if relation != "on":
                continue

            subject = edge.start
            if subject is None or subject.name == "gripper":
                continue

            if (
                not gripper_evidence_unknown
                and "holding" not in gripper_states
                and "placed" not in subject.states
            ):
                subject.states.append("placed")

        # 2) grasped / excluded by gripper relation:
        # 当 gripper 有几何信息且处于 closed/holding 时，
        # 根据 gripper 与物体的 attached/intersecting/inside 关系推断抓取对象。
        #
        # 同时，为避免夹爪/被抓取物进入搭建区时误参与验证：
        # 凡是与 gripper 有 attached / intersecting / inside 的非 gripper 节点，
        # 都加入 exclude_from_build_validation。
        if gripper_node is None:
            return

        if "unknown" in gripper_states:
            # Missing time-aligned evidence must not erase a previously
            # confirmed grasp or reintroduce the carried object into build
            # validation.  It also must not create a new grasp inference.
            if self.last_grasped_object_id:
                for node in sg.nodes:
                    if node.name != "gripper" and node.uid() == self.last_grasped_object_id:
                        self._add_state_once(node, "grasped")
                        self._add_state_once(node, "exclude_from_build_validation")
                        self._add_state_once(node, "currently_manipulated")
                        node.attributes["grasp_inferred_by"] = "gripper_memory_unknown"
                        break
            return

        # Mechanical closed without holding evidence is an empty grasp.
        if "holding" not in gripper_states:
            self.last_grasped_object_id = None
            return

        candidate_scores = []

        gripper_related_exclusion_relations = {
            "attached",
            "intersecting",
            "inside",
        }

        for edge in sg.get_relations():
            relation = getattr(edge, "edge_type", None)
            start_node = edge.start
            end_node = edge.end

            if start_node is None or end_node is None:
                continue

            other_node = None

            if start_node.uid() == gripper_node.uid() and end_node.name != "gripper":
                other_node = end_node
            elif end_node.uid() == gripper_node.uid() and start_node.name != "gripper":
                other_node = start_node

            if other_node is None:
                continue

            if relation in gripper_related_exclusion_relations:
                self._add_state_once(other_node, "exclude_from_build_validation")
                self._add_state_once(other_node, "currently_manipulated")
                other_node.attributes["excluded_by_gripper_relation"] = relation

            score = 0
            if relation == "attached":
                score = 3
            elif relation == "intersecting":
                score = 2
            elif relation == "inside":
                score = 1

            if score > 0:
                candidate_scores.append((score, other_node))

        if candidate_scores:
            candidate_scores.sort(key=lambda item: item[0], reverse=True)
            _, best_node = candidate_scores[0]

            self._add_state_once(best_node, "grasped")
            self._add_state_once(best_node, "exclude_from_build_validation")
            self._add_state_once(best_node, "currently_manipulated")

            best_node.attributes["grasp_inferred_by"] = "gripper_relation"
            self.last_grasped_object_id = best_node.uid()
            return

        # 如果当前帧没有稳定关系，但上一帧记住了被抓取对象，则继续短暂保持 grasped。
        if self.last_grasped_object_id:
            for node in sg.nodes:
                if node.name == "gripper":
                    continue

                if node.uid() == self.last_grasped_object_id:
                    self._add_state_once(node, "grasped")
                    self._add_state_once(node, "exclude_from_build_validation")
                    self._add_state_once(node, "currently_manipulated")
                    node.attributes["grasp_inferred_by"] = "gripper_memory"
                    return

    # ============================================================
    # publish / output
    # ============================================================

    def _publish_scene_graph(self, sg, detection_metadata: Optional[Dict[str, Any]] = None):
        graph_stamp = rospy.Time.now()
        graph_stamp_sec = float(graph_stamp.to_sec())
        metadata = dict(detection_metadata or {})
        detection_stamp = metadata.get("detection_stamp")
        detection_age_sec = metadata.get("detection_age_sec")
        if detection_stamp is not None:
            try:
                detection_age_sec = graph_stamp_sec - float(detection_stamp)
            except (TypeError, ValueError):
                pass

        self.scene_version += 1
        self.latest_scene_graph = sg
        self.latest_scene_graph_stamp = graph_stamp
        self.latest_scene_graph_snapshot = {
            "graph": sg,
            "scene_version": int(self.scene_version),
            "graph_stamp": graph_stamp,
            "graph_stamp_sec": graph_stamp_sec,
            "detection_stamp": detection_stamp,
            "detection_frame_index": metadata.get("detection_frame_index"),
            "capture_stamp": metadata.get("capture_stamp"),
            "capture_lower_bound": metadata.get("capture_lower_bound"),
            "capture_upper_bound": metadata.get("capture_upper_bound"),
            "capture_uncertainty_sec": metadata.get("capture_uncertainty_sec"),
            "capture_timestamp_source": metadata.get("capture_timestamp_source"),
            "source_timestamp": metadata.get("source_timestamp"),
            "source_timestamp_domain": metadata.get("source_timestamp_domain"),
            "source_frame_seq": metadata.get("source_frame_seq"),
            "publish_stamp": metadata.get("publish_stamp"),
            "inference_start_stamp": metadata.get("inference_start_stamp"),
            "inference_end_stamp": metadata.get("inference_end_stamp"),
            "run_id": metadata.get("run_id", ""),
            "attempt_id": metadata.get("attempt_id"),
            "stage_seq": metadata.get("stage_seq"),
            "stage_name": metadata.get("stage_name", ""),
            "barrier_id": metadata.get("barrier_id", ""),
            "not_before": metadata.get("not_before"),
            "transaction_eligible": bool(metadata.get("transaction_eligible", False)),
            "detection_receive_time": metadata.get("detection_receive_time"),
            "detection_receive_age_sec": metadata.get("detection_age_sec"),
            "detection_age_sec": detection_age_sec,
            "gripper_alignment": dict(self.latest_gripper_alignment or {}),
            "detection_payload": self.latest_detection_payload,
        }
        self._publish_scene_graph_snapshot(sg)
        self._publish_scene_graph_status()

    def _publish_scene_graph_snapshot(self, sg) -> None:
        if self.scene_graph_pub is None:
            return
        try:
            runtime_summary = self.get_runtime_observation_summary(sg)
            payload = scene_graph_to_snapshot_dict(
                sg,
                self.latest_scene_graph_snapshot,
                runtime_summary=runtime_summary,
            )
            self.scene_graph_pub.publish(String(data=snapshot_to_json(payload)))
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Failed to publish scene graph snapshot: %s", str(exc))

    def _log_scene_graph_output(self, sg, nodes: List[Node]):
        """
        输出场景图。

        输出格式示例：

        SceneGraph: nodes=8, edges=8

        [NODES]
          arch_1: arch
          cuboid_2: cuboid, states=['grasped']
          gripper: gripper, states=['closed', 'holding']

        [EDGES]
          cube_1 --on--> cuboid_1
          cuboid_1 --supports--> cube_1
        """
        if not self.print_scene_graph:
            relation_summary = {}
            if hasattr(sg, "relation_summary") and callable(getattr(sg, "relation_summary")):
                try:
                    relation_summary = sg.relation_summary()
                except Exception:
                    relation_summary = {}

            rospy.loginfo(
                "SceneGraph updated: nodes=%d, edges=%d, relations=%s",
                len(getattr(sg, "nodes", []) or []),
                len(getattr(sg, "edges", {}) or {}),
                str(relation_summary),
            )
            return

        text = self.format_scene_graph_plain(sg)
        print(text, flush=True)

    def format_scene_graph_plain(self, graph=None) -> str:
        """
        按固定格式输出场景图。
        不强制排序，直接按照 SceneGraph 内部节点和边的当前顺序输出。
        """
        sg = graph or self.latest_scene_graph

        if sg is None:
            return (
                "SceneGraph: nodes=0, edges=0\n\n"
                "[NODES]\n"
                "  无\n\n"
                "[EDGES]\n"
                "  无"
            )

        nodes = list(getattr(sg, "nodes", []) or [])

        try:
            relations = list(sg.get_relations() or [])
        except Exception:
            relations = []

        lines = []
        lines.append(f"SceneGraph: nodes={len(nodes)}, edges={len(relations)}")
        lines.append("")
        lines.append("[NODES]")

        if not nodes:
            lines.append("  无")
        else:
            for node in nodes:
                node_id = str(node.uid())
                class_name = str(node.name)
                states = list(getattr(node, "states", []) or [])

                if states:
                    lines.append(f"  {node_id}: {class_name}, states={repr(states)}")
                else:
                    lines.append(f"  {node_id}: {class_name}")

        lines.append("")
        lines.append("[EDGES]")

        if not relations:
            lines.append("  无")
        else:
            for edge in relations:
                start_id = str(edge.start.uid())
                end_id = str(edge.end.uid())
                relation = str(edge.edge_type)
                lines.append(f"  {start_id} --{relation}--> {end_id}")

        return "\n".join(lines)

    def format_graph_brief(self, graph=None) -> str:
        """
        兼容旧接口。
        """
        return self.format_scene_graph_plain(graph)

    # ============================================================
    # validation_monitor / run_task_plan_closed_loop 兼容接口
    # ============================================================

    def get_runtime_observation_summary(self, scene_graph=None):
        sg = scene_graph or self.latest_scene_graph
        snapshot = dict(self.latest_scene_graph_snapshot or {})

        if sg is None:
            return {
                "scene_graph_stamp": str(self.latest_scene_graph_stamp),
                "detection_stamp": snapshot.get("detection_stamp"),
                "detection_frame_index": snapshot.get("detection_frame_index"),
                "detection_age_sec": snapshot.get("detection_age_sec"),
                "detection_receive_age_sec": snapshot.get("detection_receive_age_sec"),
                "detection_receive_time": snapshot.get("detection_receive_time"),
                "camera_source": "global",
                "detector_type": "yolo_segmentation",
                "coordinate_frame": "global_table_frame",
                "gripper_states": [],
                "gripper_has_geometry": False,
                "gripper_alignment": dict(snapshot.get("gripper_alignment") or {}),
                "grasped_object_ids": [],
                "placed_object_ids": [],
                "visible_object_ids": [],
                "build_region_object_ids": [],
                "build_validation_object_ids": [],
                "object_geometries": {},
            }

        gripper_states = []
        gripper_has_geometry = False
        gripper_evidence = dict(snapshot.get("gripper_alignment") or {})
        grasped_object_ids = []
        placed_object_ids = []
        visible_object_ids = []
        build_region_object_ids = []
        build_validation_object_ids = []
        object_geometries: Dict[str, Dict[str, Any]] = {}

        for node in sg.nodes:
            if node.uid() == self.gripper_node_id or node.name == "gripper":
                gripper_states = list(getattr(node, "states", []) or [])
                gripper_evidence.update(dict(getattr(node, "attributes", {}) or {}))
                pcd = getattr(node, "pcd_np", None)
                gripper_has_geometry = bool(pcd is not None and len(pcd) > 0)
                continue

            visible_object_ids.append(node.uid())

            node_states = set(getattr(node, "states", []) or [])
            if "grasped" in node_states:
                grasped_object_ids.append(node.uid())
            if "placed" in node_states:
                placed_object_ids.append(node.uid())
            if "in_build_region" in node_states:
                build_region_object_ids.append(node.uid())

            if (
                "in_build_region" in node_states
                and not self._is_node_excluded_from_build_validation(node)
            ):
                build_validation_object_ids.append(node.uid())

            attrs = dict(getattr(node, "attributes", {}) or {})
            object_geometries[node.uid()] = {
                "class_name": node.name,
                "states": list(getattr(node, "states", []) or []),
                "center_sg": attrs.get("center_sg"),
                "center_table": attrs.get("center_table"),
                "table_min": attrs.get("table_min"),
                "table_max": attrs.get("table_max"),
                "height": attrs.get("height"),
                "score": attrs.get("score"),
                "point_count": attrs.get("point_count"),
                "in_build_region": attrs.get("in_build_region"),
                "build_region_overlap": attrs.get("build_region_overlap"),
                "center_in_build_region": attrs.get("center_in_build_region"),
                "region_name": attrs.get("region_name"),
                "build_region_center_px": attrs.get("build_region_center_px"),
                "sorting_region": attrs.get("sorting_region"),
                "sorting_region_state": attrs.get("sorting_region_state"),
                "sorting_region_overlap": attrs.get("sorting_region_overlap"),
                "sorting_region_overlaps": attrs.get("sorting_region_overlaps"),
                "excluded_from_build_validation": self._is_node_excluded_from_build_validation(node),
                "excluded_by_gripper_relation": attrs.get("excluded_by_gripper_relation"),
                "grasp_inferred_by": attrs.get("grasp_inferred_by"),
            }

        return {
            "scene_graph_stamp": str(self.latest_scene_graph_stamp),
            "detection_stamp": snapshot.get("detection_stamp"),
            "detection_frame_index": snapshot.get("detection_frame_index"),
            "detection_age_sec": snapshot.get("detection_age_sec"),
            "detection_receive_age_sec": snapshot.get("detection_receive_age_sec"),
            "detection_receive_time": snapshot.get("detection_receive_time"),
            "camera_source": "global",
            "detector_type": "yolo_segmentation",
            "coordinate_frame": "global_table_frame",
            "gripper_states": gripper_states,
            "gripper_has_geometry": bool(gripper_has_geometry),
            "gripper_alignment": gripper_evidence,
            "grasped_object_ids": grasped_object_ids,
            "placed_object_ids": placed_object_ids,
            "visible_object_ids": visible_object_ids,
            "build_region_object_ids": build_region_object_ids,
            "build_validation_object_ids": build_validation_object_ids,
            "object_geometries": object_geometries,
        }

    def get_latest_scene_graph(self):
        return self.latest_scene_graph

    def get_latest_scene_graph_snapshot(self):
        if self.latest_scene_graph_snapshot is None:
            return None
        return dict(self.latest_scene_graph_snapshot)

    def has_latest_scene_graph(self):
        return self.latest_scene_graph is not None


if __name__ == "__main__":
    if not rospy.core.is_initialized():
        rospy.init_node("scene_graph_builder", anonymous=True)

    builder = SceneGraphBuilder(
        init_node=False,
        gripper_state_provider=None,
        publish_scene_graph=bool(rospy.get_param("~publish_scene_graph", True)),
    )

    rospy.spin()
