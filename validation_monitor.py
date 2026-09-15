#!/usr/bin/env python3
# -*- coding: utf-8 -*-




from scene_graph_system.resources import package_resource_path
import json
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import rospy
from std_msgs.msg import String
from scene_graph_system.msg import GripperState

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


from scene_graph_system.validation.scene_graph_comparator import SceneGraphComparator, SceneGraphComparatorConfig
from scene_graph_system.planning.configurable_expected_state_builder import (
    build_configured_process_supervision_registry,
    compare_configured_bundle_state,
)
from scene_graph_system.planning.generate_expected_state_ai import build_process_supervision_registry_from_execution_plan
from scene_graph_system.scene_graph.scene_graph_builder import SceneGraphBuilder
from scene_graph_system.validation.scene_graph_comparator import SceneGraphComparator
from scene_graph_system.scene_graph.scene_graph_snapshot_codec import snapshot_from_json
from scene_graph_system.planning.task_plan_adapter import build_internal_plan_from_file, load_task_plan
from scene_graph_system.diagnosis.fault_event import (
    TOPIC_RECOVERY_RUNTIME_CONTEXT,
    TOPIC_RECOVERY_RESOLUTION,
    RecoveryRuntimeContext,
    RecoveryResolution,
)
from scene_graph_system.validation.validation_transaction import (
    BarrierRegistry,
    FreshSceneWaitRegistry,
    TOPIC_OBSERVATION_REQUEST,
    TOPIC_VALIDATION_REQUEST,
    TOPIC_VALIDATION_RESULT,
    ValidationRequest,
    ValidationResult,
    ValidationStatus,
    VersionedSnapshotBuffer,
    VisionUnavailableError,
    evaluate_gripper_evidence_for_request,
    snapshot_matches_request,
    validate_snapshot_event_alignment,
)

DEFAULT_PLAN_FILE = package_resource_path('scripts/action.py')
DEFAULT_REALMAN_IP = "192.168.0.18"
DEFAULT_REALMAN_PORT = 8080
DEFAULT_REALMAN_THREAD_MODE = "RM_TRIPLE_MODE_E"
DEFAULT_STAGE_TOPIC = "/task_execution/current_stage"
DEFAULT_STATUS_TOPIC = "/task_execution/status"
DEFAULT_ALERT_TOPIC = "/fault/deviation_alert"
DEFAULT_RESOLUTION_TOPIC = "/recovery/resolution"
DEFAULT_TASK_PROFILE = "block_building"
DEFAULT_SCENE_GRAPH_TOPIC = "/scene_graph/current"
DEFAULT_SCENE_GRAPH_STATUS_TOPIC = "/scene_graph/status"
DEFAULT_GRIPPER_STATE_TOPIC = "/gripper/state"
TOPIC_BREAKPOINT_REQUEST = "/recovery/request_breakpoint_search"
TOPIC_BREAKPOINT_RESULT = "/recovery/breakpoint_result"
_LATEST_STAGE_EVENT_CONTEXT: Dict[str, Any] = {}


def format_expected_state_brief(expected_state):
    node_parts = []
    for node in getattr(expected_state, "expected_nodes", []) or []:
        states = list(getattr(node, "required_states", []) or [])
        states_any = list(getattr(node, "required_states_any", []) or [])
        state_parts = list(states)
        if states_any:
            state_parts.append("any(%s)" % ", ".join(states_any))
        if state_parts:
            node_parts.append(f"{node.node_id}({node.class_name})[{', '.join(state_parts)}]")
        else:
            node_parts.append(f"{node.node_id}({node.class_name})")

    relation_parts = []
    for rel in getattr(expected_state, "expected_relations", []) or []:
        relation_parts.append(f"{rel.subject_id} --{rel.relation}--> {rel.object_id}")

    forbidden_parts = []
    for rel in getattr(expected_state, "forbidden_relations", []) or []:
        forbidden_parts.append(f"{rel.subject_id} --{rel.relation}--> {rel.object_id}")

    lines = []
    lines.append("预期状态:")
    lines.append(f"  节点: {', '.join(node_parts) if node_parts else '无'}")
    lines.append(f"  关系: {', '.join(relation_parts) if relation_parts else '无'}")
    if forbidden_parts:
        lines.append(f"  禁止关系: {', '.join(forbidden_parts)}")
    return "\n".join(lines)


def format_graph_brief(graph):
    node_lines = []
    for node in getattr(graph, "nodes", []) or []:
        states = list(getattr(node, "states", []) or [])
        if states:
            node_lines.append(f"    - {node.uid()}({node.name})[{', '.join(states)}]")
        else:
            node_lines.append(f"    - {node.uid()}({node.name})")

    relation_lines = []
    get_relations = getattr(graph, "get_relations", None)
    if callable(get_relations):
        for edge in get_relations() or []:
            relation_lines.append(f"    - {edge.start.uid()} --{edge.edge_type}--> {edge.end.uid()}")

    lines = []
    lines.append("当前场景图摘要:")
    lines.append("  节点:")
    lines.extend(node_lines if node_lines else ["    - 无"])
    lines.append("  关系:")
    lines.extend(relation_lines if relation_lines else ["    - 无"])
    return "\n".join(lines)


def format_failure_reasons(result):
    if not getattr(result, "issues", None):
        return "故障原因:\n  - 无"

    lines = ["故障原因:"]
    for issue in result.issues:
        lines.append(f"  - {issue.issue_type}: {issue.message}")
    return "\n".join(lines)


def print_compare_output(stage_name, expected_state, graph, result, tag="在线"):
    """
    统一验证端控制台输出格式。

    无论 passed True/False，都输出：
      [tag] 当前动作
      预期状态
      对比结果
      当前场景图摘要

    只有 passed=False 时，才输出：
      故障原因
    """
    print("=" * 60)
    print(f"[{tag}] 当前动作: {stage_name}")
    print(format_expected_state_brief(expected_state))
    print("对比结果:")
    print(f"  passed={result.passed}, severity={result.highest_severity()}")

    if not result.passed:
        print(format_failure_reasons(result))

    print(format_graph_brief(graph))
    print("", flush=True)

def _issue_dict_list(result):
    out = []
    for issue in getattr(result, "issues", []) or []:
        out.append({
            "issue_type": getattr(issue, "issue_type", ""),
            "severity": getattr(issue, "severity", "error"),
            "message": getattr(issue, "message", ""),
            "node_id": getattr(issue, "node_id", None),
            "subject_id": getattr(issue, "subject_id", None),
            "object_id": getattr(issue, "object_id", None),
            "relation": getattr(issue, "relation", None),
            "metadata": dict(getattr(issue, "metadata", {}) or {}),
        })
    return out


def _infer_execution_step_from_stage_name(stage_name: str):
    """
    从当前 catch.py 发布的过程阶段名中提取执行 step。
    支持格式：
      step_1_cube_1__approach_source
      step_1_cube_1__close_gripper
      step_2_triangle_1__leave_target
    """
    try:
        raw = str(stage_name).strip().lower()
        if not raw.startswith("step_"):
            return None

        after_prefix = raw[len("step_"):]
        head = after_prefix.split("__", 1)[0]   # 例如 "1_cube_1"
        first_token = head.split("_", 1)[0]     # 取出 1
        return int(first_token)
    except Exception:
        return None


def _to_sec_value(value) -> Optional[float]:
    if value is None:
        return None
    try:
        if hasattr(value, "to_sec"):
            return float(value.to_sec())
        return float(value)
    except Exception:
        return None


def get_scene_graph_snapshot(builder):
    snapshot = None
    get_snapshot = getattr(builder, "get_latest_scene_graph_snapshot", None)
    if callable(get_snapshot):
        try:
            snapshot = get_snapshot()
        except Exception:
            snapshot = None

    if isinstance(snapshot, dict) and snapshot.get("graph") is not None:
        return snapshot

    graph = None
    get_graph = getattr(builder, "get_latest_scene_graph", None)
    if callable(get_graph):
        try:
            graph = get_graph()
        except Exception:
            graph = None

    if graph is None:
        return None

    graph_stamp = getattr(builder, "latest_scene_graph_stamp", None)
    graph_stamp_sec = _to_sec_value(graph_stamp)
    payload = getattr(builder, "latest_detection_payload", {}) or {}
    if not isinstance(payload, dict):
        payload = {}

    detection_stamp = _to_sec_value(payload.get("stamp"))
    now_sec = float(rospy.Time.now().to_sec())
    detection_age_sec = None
    if detection_stamp is not None:
        detection_age_sec = now_sec - detection_stamp

    return {
        "graph": graph,
        "graph_stamp": graph_stamp,
        "graph_stamp_sec": graph_stamp_sec,
        "detection_stamp": detection_stamp,
        "detection_frame_index": payload.get("frame_index"),
        "detection_age_sec": detection_age_sec,
        "detection_payload": payload,
    }


def snapshot_graph_stamp_key(snapshot, builder=None):
    if isinstance(snapshot, dict):
        graph_stamp_sec = _to_sec_value(snapshot.get("graph_stamp_sec"))
        if graph_stamp_sec is not None:
            return graph_stamp_sec
        graph_stamp = snapshot.get("graph_stamp")
        graph_stamp_sec = _to_sec_value(graph_stamp)
        if graph_stamp_sec is not None:
            return graph_stamp_sec

    if builder is not None:
        return _to_sec_value(getattr(builder, "latest_scene_graph_stamp", None))

    return None


def validate_snapshot_freshness(
    snapshot,
    *,
    now_sec: float,
    require_fresh_detection: bool,
    max_detection_age_sec: float,
    stamp_tolerance_sec: float,
    reference_ts: Optional[float] = None,
):
    if not require_fresh_detection:
        return True, ""

    if not isinstance(snapshot, dict):
        return False, "missing_scene_graph_snapshot"

    detection_stamp = _to_sec_value(snapshot.get("detection_stamp"))
    if detection_stamp is None or detection_stamp <= 0.0:
        return False, "missing_detection_stamp"

    detection_age = float(now_sec) - float(detection_stamp)

    if detection_age > float(max_detection_age_sec) + float(stamp_tolerance_sec):
        return False, "stale_detection_age"

    if reference_ts is not None:
        ref_sec = _to_sec_value(reference_ts)
        if ref_sec is not None and detection_stamp < ref_sec - float(stamp_tolerance_sec):
            return False, "detection_before_reference"

    return True, ""


def _legacy_validate_snapshot_event_alignment(
    snapshot,
    *,
    stamp_tolerance_sec: float,
    reference_ts: Optional[float] = None,
    require_event_alignment: bool = True,
    now_sec: Optional[float] = None,
    max_detection_age_sec: Optional[float] = None,
):
    """
    验证场景图是否适用于当前事件。

    注意：这里不再判断 now - detection_stamp 是否超时。
    检测帧 age 是否过期只由 scene_graph_builder 负责；
    validation_monitor 只判断 detection_stamp 是否早于当前事件。
    """
    if not require_event_alignment:
        return True, ""

    if not isinstance(snapshot, dict):
        return False, "missing_scene_graph_snapshot"

    detection_stamp = _to_sec_value(snapshot.get("detection_stamp"))
    if detection_stamp is None or detection_stamp <= 0.0:
        return False, "missing_detection_stamp"

    if now_sec is not None and max_detection_age_sec is not None:
        detection_age = float(now_sec) - float(detection_stamp)
        tolerance = float(stamp_tolerance_sec)
        if detection_age > float(max_detection_age_sec) + tolerance:
            return False, "stale_detection_age"
        if detection_age < -tolerance:
            return False, "detection_stamp_in_future"

    if reference_ts is not None:
        ref_sec = _to_sec_value(reference_ts)
        if ref_sec is not None and detection_stamp < ref_sec - float(stamp_tolerance_sec):
            return False, "detection_before_reference"

    return True, ""


def log_skipped_event_alignment_validation(kind, stage_name, reason, snapshot, reference_ts=None):
    if not isinstance(snapshot, dict):
        snapshot = {}
    detection_stamp = _to_sec_value(snapshot.get("detection_stamp"))
    current_age = None
    if detection_stamp is not None:
        current_age = float(rospy.Time.now().to_sec()) - detection_stamp

    rospy.logwarn_throttle(
        2.0,
        "[VALIDATION] skip %s compare while waiting event-aligned scene: stage=%s reason=%s "
        "det_stamp=%s det_age=%s snapshot_age=%s frame_index=%s reference_ts=%s",
        str(kind),
        str(stage_name),
        str(reason),
        str(detection_stamp),
        str(current_age),
        str(snapshot.get("detection_age_sec")),
        str(snapshot.get("detection_frame_index")),
        str(reference_ts),
    )


def handle_event_alignment_wait(
    wait_registry: FreshSceneWaitRegistry,
    alert_pub,
    last_alert_signature,
    *,
    kind,
    stage_name,
    reason,
    snapshot,
    now_sec,
    timeout_sec,
    builder,
    source,
    reference_ts=None,
):
    key = (str(kind), str(stage_name))
    elapsed, timed_out = wait_registry.mark_waiting(key, now_sec)
    log_skipped_event_alignment_validation(
        kind,
        stage_name,
        reason,
        snapshot,
        reference_ts=reference_ts,
    )
    rospy.logwarn_throttle(
        2.0,
        "[VALIDATION] waiting event-aligned scene graph: kind=%s stage=%s reason=%s "
        "elapsed=%.1fs timeout=%.1fs",
        str(kind),
        str(stage_name),
        str(reason),
        float(elapsed),
        float(timeout_sec),
    )

    if timed_out:
        last_alert_signature = maybe_publish_vision_unavailable_alert(
            alert_pub,
            last_alert_signature,
            kind=kind,
            stage_name=stage_name,
            reason=reason,
            snapshot=snapshot if isinstance(snapshot, dict) else {},
            elapsed=elapsed,
            timeout_sec=timeout_sec,
            builder=builder,
            source=source,
        )

    return last_alert_signature


def mark_validation_scene_fresh(wait_registry: FreshSceneWaitRegistry, kind, stage_name):
    wait_registry.mark_fresh((str(kind), str(stage_name)))


def build_deviation_alert_dict(stage_name, expected_state, graph, result, builder, source="validation_monitor"):
    runtime_summary = {}
    if builder is not None and hasattr(builder, "get_runtime_observation_summary"):
        try:
            runtime_summary = builder.get_runtime_observation_summary(graph)
        except Exception:
            runtime_summary = {}

    return {
        "source": source,
        "stage_name": str(stage_name),
        "execution_step": _infer_execution_step_from_stage_name(stage_name),
        "severity": result.highest_severity(),
        "comparison_summary": result.summary_dict(),
        "issues": _issue_dict_list(result),
        "expected_state_brief": format_expected_state_brief(expected_state),
        "graph_brief": format_graph_brief(graph),
        "runtime_summary": runtime_summary,
        "message": "; ".join(
            f"{issue.issue_type}: {issue.message}"
            for issue in getattr(result, "issues", []) or []
        ) or "comparison_failed",
        "timestamp": time.time(),
    }


def publish_deviation_alert(alert_pub, payload: dict):
    if alert_pub is None:
        return
    if _LATEST_STAGE_EVENT_CONTEXT.get("run_id"):
        for key in ("run_id", "attempt_id", "stage_seq"):
            payload.setdefault(key, _LATEST_STAGE_EVENT_CONTEXT.get(key))
    try:
        alert_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
    except Exception as exc:
        rospy.logwarn("Failed to publish deviation alert: %s", str(exc))


def maybe_publish_failure_alert(
    alert_pub,
    last_alert_signature,
    *,
    stage_name,
    expected_state,
    graph,
    result,
    builder,
    source,
    always_publish=False,
):
    if result is None or result.passed:
        return last_alert_signature

    alert_signature = (
        str(source),
        str(stage_name),
        result.highest_severity(),
        tuple((issue.issue_type, issue.message) for issue in result.issues),
    )

    if (not always_publish) and alert_signature == last_alert_signature:
        return last_alert_signature

    payload = build_deviation_alert_dict(
        stage_name=stage_name,
        expected_state=expected_state,
        graph=graph,
        result=result,
        builder=builder,
        source=source,
    )
    publish_deviation_alert(alert_pub, payload)
    return alert_signature


def maybe_publish_unknown_stage_alert(
    alert_pub,
    last_alert_signature,
    *,
    stage_name,
    graph,
    builder,
    source,
    always_publish=False,
):
    alert_signature = (
        "unknown_stage",
        str(source),
        str(stage_name),
    )
    if (not always_publish) and alert_signature == last_alert_signature:
        return last_alert_signature

    runtime_summary = {}
    if builder is not None and hasattr(builder, "get_runtime_observation_summary"):
        try:
            runtime_summary = builder.get_runtime_observation_summary(graph)
        except Exception:
            runtime_summary = {}

    payload = {
        "source": source,
        "stage_name": str(stage_name),
        "execution_step": _infer_execution_step_from_stage_name(stage_name),
        "severity": "error",
        "comparison_summary": {
            "stage_name": str(stage_name),
            "passed": False,
            "highest_severity": "error",
            "issue_count": 1,
            "issue_types": {"unknown_stage": 1},
            "issue_severity": {"error": 1},
        },
        "issues": [
            {
                "issue_type": "unknown_stage",
                "severity": "error",
                "message": f"当前阶段 '{stage_name}' 不在 process_registry 中。",
            }
        ],
        "expected_state_brief": "",
        "graph_brief": format_graph_brief(graph),
        "runtime_summary": runtime_summary,
        "message": f"unknown_stage: {stage_name}",
        "timestamp": time.time(),
    }
    publish_deviation_alert(alert_pub, payload)
    return alert_signature


def maybe_publish_vision_unavailable_alert(
    alert_pub,
    last_alert_signature,
    *,
    kind,
    stage_name,
    reason,
    snapshot,
    elapsed,
    timeout_sec,
    builder,
    source,
):
    alert_signature = (
        "vision_unavailable",
        str(source),
        str(kind),
        str(stage_name),
        str(reason),
    )
    if alert_signature == last_alert_signature:
        return last_alert_signature

    graph = snapshot.get("graph") if isinstance(snapshot, dict) else None
    runtime_summary = {}
    if builder is not None and hasattr(builder, "get_runtime_observation_summary"):
        try:
            runtime_summary = builder.get_runtime_observation_summary(graph)
        except Exception:
            runtime_summary = {}

    detection_stamp = _to_sec_value(snapshot.get("detection_stamp")) if isinstance(snapshot, dict) else None
    detection_age = None
    if detection_stamp is not None:
        detection_age = float(rospy.Time.now().to_sec()) - detection_stamp

    payload = {
        "source": source,
        "stage_name": str(stage_name),
        "execution_step": _infer_execution_step_from_stage_name(stage_name),
        "severity": "warning",
        "comparison_summary": {
            "stage_name": str(stage_name),
            "passed": False,
            "highest_severity": "warning",
            "issue_count": 1,
            "issue_types": {"vision_unavailable": 1},
            "issue_severity": {"warning": 1},
            "compare_kind": str(kind),
        },
        "issues": [
            {
                "issue_type": "vision_unavailable",
                "severity": "warning",
                "message": (
                    "No event-aligned scene graph for %s compare within %.1fs "
                    "(reason=%s, elapsed=%.1fs)."
                    % (str(kind), float(timeout_sec), str(reason), float(elapsed))
                ),
                "metadata": {
                    "kind": str(kind),
                    "reason": str(reason),
                    "elapsed": float(elapsed),
                    "timeout_sec": float(timeout_sec),
                    "detection_stamp": detection_stamp,
                    "detection_age_sec": detection_age,
                    "detection_frame_index": (
                        snapshot.get("detection_frame_index")
                        if isinstance(snapshot, dict)
                        else None
                    ),
                },
            }
        ],
        "expected_state_brief": "",
        "graph_brief": format_graph_brief(graph) if graph is not None else "",
        "runtime_summary": runtime_summary,
        "message": "vision_unavailable: %s compare waiting for event-aligned scene graph" % str(kind),
        "timestamp": time.time(),
    }
    publish_deviation_alert(alert_pub, payload)
    return alert_signature


class RecoveryContextTracker:
    def __init__(self):
        self.context: Optional[RecoveryRuntimeContext] = None

    def callback(self, msg: String):
        raw = str(msg.data).strip()
        if not raw:
            return
        try:
            self.context = RecoveryRuntimeContext.from_json(raw)
        except Exception as exc:
            rospy.logwarn("RecoveryContextTracker failed to parse runtime context: %s", str(exc))


def _is_recovery_completion_stage(stage_name: str) -> bool:
    return str(stage_name).strip().endswith("__leave_target")


def publish_recovery_resolution(resolution_pub, resolution: RecoveryResolution):
    if resolution_pub is None:
        return
    try:
        resolution_pub.publish(String(data=resolution.to_json()))
    except Exception as exc:
        rospy.logwarn("Failed to publish recovery resolution: %s", str(exc))


def maybe_publish_recovery_resolution(
    resolution_pub,
    last_resolution_signature,
    *,
    stage_name,
    graph,
    result,
    builder,
    context,
    source,
):
    if result is None or not result.passed:
        return last_resolution_signature

    if context is None or str(getattr(context, "status", "")).strip() != "recovery":
        return last_resolution_signature

    if not _is_recovery_completion_stage(stage_name):
        return last_resolution_signature

    execution_step = _infer_execution_step_from_stage_name(stage_name)
    context_step = getattr(context, "step", None)
    if execution_step is not None and context_step is not None and int(execution_step) != int(context_step):
        return last_resolution_signature

    resolution_signature = (str(source), str(stage_name), execution_step, True)
    if resolution_signature == last_resolution_signature:
        return last_resolution_signature

    runtime_summary = {}
    if builder is not None and hasattr(builder, "get_runtime_observation_summary"):
        try:
            runtime_summary = builder.get_runtime_observation_summary(graph)
        except Exception:
            runtime_summary = {}

    resolution = RecoveryResolution(
        source=source,
        stage_name=str(stage_name),
        execution_step=execution_step,
        passed=True,
        compare_kind="post",
        comparison_summary=result.summary_dict(),
        runtime_summary=runtime_summary,
        graph_brief=format_graph_brief(graph),
        message=f"recovery_resolved: {stage_name}",
    )
    publish_recovery_resolution(resolution_pub, resolution)
    return resolution_signature


class StageTracker:
    def __init__(self):
        self.current_stage_name: Optional[str] = None
        self.last_stage_update_time: Optional[float] = None
        self.run_id: str = ""
        self.attempt_id: Optional[int] = None
        self.stage_seq: Optional[int] = None

        self.pending_started_stage_name: Optional[str] = None
        self.pending_started_since: Optional[float] = None
        self.pending_started_baseline_graph_stamp: Optional[int] = None

        self.pending_finished_stage_name: Optional[str] = None
        self.pending_finished_since: Optional[float] = None
        self.pending_finished_graph_count: int = 0
        self.pending_baseline_graph_stamp: Optional[int] = None
        self.pending_last_counted_graph_stamp: Optional[int] = None

    def update_stage(self, stage_name: Optional[str], event_stamp: Optional[float] = None):
        stage_name = str(stage_name).strip() if stage_name is not None else ""
        if not stage_name:
            return
        self.current_stage_name = stage_name
        self.last_stage_update_time = (
            float(rospy.Time.now().to_sec()) if event_stamp is None else float(event_stamp)
        )

    def stage_callback(self, msg: String):
        self.update_stage(msg.data)

    def status_callback(self, msg: String):
        global _LATEST_STAGE_EVENT_CONTEXT
        raw = str(msg.data).strip()
        if not raw:
            return
        try:
            payload = json.loads(raw)
        except Exception:
            self.update_stage(raw)
            return

        stage_name = payload.get("stage_name")
        event = payload.get("event")
        try:
            event_stamp = float(payload.get("event_stamp"))
        except (TypeError, ValueError):
            event_stamp = float(rospy.Time.now().to_sec())
        self.run_id = str(payload.get("run_id", self.run_id) or self.run_id)
        try:
            self.attempt_id = int(payload.get("attempt_id"))
        except (TypeError, ValueError):
            pass
        try:
            self.stage_seq = int(payload.get("stage_seq"))
        except (TypeError, ValueError):
            pass
        if self.run_id and self.attempt_id is not None:
            _LATEST_STAGE_EVENT_CONTEXT = {
                "run_id": self.run_id,
                "attempt_id": self.attempt_id,
                "stage_seq": self.stage_seq,
            }
        if stage_name:
            self.update_stage(stage_name, event_stamp=event_stamp)

        if stage_name and event == "started":
            self.pending_started_stage_name = str(stage_name).strip()
            self.pending_started_since = event_stamp
            self.pending_started_baseline_graph_stamp = None

        if stage_name and event == "finished":
            self.pending_finished_stage_name = str(stage_name).strip()
            self.pending_finished_since = event_stamp
            self.pending_finished_graph_count = 0
            self.pending_baseline_graph_stamp = None
            self.pending_last_counted_graph_stamp = None

    def clear_pending_started(self):
        self.pending_started_stage_name = None
        self.pending_started_since = None
        self.pending_started_baseline_graph_stamp = None

    def clear_pending_finished(self):
        self.pending_finished_stage_name = None
        self.pending_finished_since = None
        self.pending_finished_graph_count = 0
        self.pending_baseline_graph_stamp = None
        self.pending_last_counted_graph_stamp = None


class ExternalSceneGraphSource:
    """
    Scene graph source backed by /scene_graph/current snapshots.

    It intentionally exposes the same minimal interface as SceneGraphBuilder so
    validation and breakpoint code can use either source without branching.
    """

    def __init__(self, scene_graph_topic: str, status_topic: str, snapshot_queue_size: int = 10):
        self.scene_graph_topic = str(scene_graph_topic or DEFAULT_SCENE_GRAPH_TOPIC).strip()
        self.status_topic = str(status_topic or DEFAULT_SCENE_GRAPH_STATUS_TOPIC).strip()

        self.latest_scene_graph = None
        self.latest_scene_graph_stamp = None
        self.latest_scene_graph_snapshot = None
        self.latest_status = {}
        self.latest_raw_payload = {}
        self._lock = threading.RLock()
        self._transaction_snapshots = VersionedSnapshotBuffer(
            maxlen=max(1, int(snapshot_queue_size))
        )

        rospy.Subscriber(
            self.scene_graph_topic,
            String,
            self._scene_graph_callback,
            queue_size=max(1, int(snapshot_queue_size)),
        )
        rospy.Subscriber(
            self.status_topic,
            String,
            self._status_callback,
            queue_size=1,
        )
        rospy.logwarn(
            "[VALIDATION] external scene graph source enabled: graph_topic=%s status_topic=%s",
            self.scene_graph_topic,
            self.status_topic,
        )

    def _scene_graph_callback(self, msg: String):
        raw = str(msg.data or "").strip()
        if not raw:
            return
        try:
            snapshot = snapshot_from_json(raw)
        except Exception as exc:
            rospy.logwarn_throttle(
                2.0,
                "[VALIDATION] failed to parse external scene graph snapshot: %s",
                str(exc),
            )
            return

        graph = snapshot.get("graph")
        if graph is None:
            rospy.logwarn_throttle(
                2.0,
                "[VALIDATION] external scene graph snapshot has no graph",
            )
            return

        with self._lock:
            self.latest_raw_payload = dict(snapshot or {})
            self.latest_scene_graph = graph
            self.latest_scene_graph_snapshot = snapshot
            self.latest_scene_graph_stamp = (
                snapshot.get("graph_stamp_sec")
                if snapshot.get("graph_stamp_sec") is not None
                else snapshot.get("graph_stamp")
            )
        self._transaction_snapshots.append(snapshot)

    def _status_callback(self, msg: String):
        raw = str(msg.data or "").strip()
        if not raw:
            return
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            self.latest_status = payload

    def has_latest_scene_graph(self):
        with self._lock:
            return self.latest_scene_graph is not None

    def get_latest_scene_graph(self):
        with self._lock:
            return self.latest_scene_graph

    def get_latest_scene_graph_snapshot(self):
        with self._lock:
            if self.latest_scene_graph_snapshot is None:
                return None
            return dict(self.latest_scene_graph_snapshot)

    def drain_transaction_snapshots(self):
        return self._transaction_snapshots.drain()

    def get_runtime_observation_summary(self, scene_graph=None):
        snapshot = dict(self.latest_scene_graph_snapshot or {})
        runtime_summary = snapshot.get("runtime_summary")
        if isinstance(runtime_summary, dict) and runtime_summary:
            return dict(runtime_summary)

        return {
            "scene_graph_stamp": str(self.latest_scene_graph_stamp),
            "detection_stamp": snapshot.get("detection_stamp"),
            "detection_frame_index": snapshot.get("detection_frame_index"),
            "detection_age_sec": snapshot.get("detection_age_sec"),
            "detection_receive_age_sec": snapshot.get("detection_receive_age_sec"),
            "detection_receive_time": snapshot.get("detection_receive_time"),
            "camera_source": "global",
            "detector_type": "yolo_segmentation",
            "external_scene_graph_status": dict(self.latest_status or {}),
        }


def compare_bundle_state(stage_name, kind, bundle, graph, comparator):
    return compare_configured_bundle_state(stage_name, kind, bundle, graph, comparator)


class TransactionValidationCoordinator:
    """ROS adapter around the pure single-request transaction registry."""

    _POLICY_KIND = {
        "initial_baseline": "pre",
        "grasp_commit": "post",
        "preplace_safety": "post",
        "place_commit": "post",
    }

    def __init__(
        self,
        *,
        builder,
        comparator,
        process_registry,
        alert_pub,
        mode: str,
        request_topic: str,
        observation_request_topic: str,
        result_topic: str,
        observation_retry_sec: float,
        terminal_retention_sec: float,
        gripper_state_topic: str,
        gripper_min_stable_count: int,
    ):
        self.builder = builder
        self.comparator = comparator
        self.process_registry = process_registry
        self.alert_pub = alert_pub
        self.mode = str(mode or "off").strip().lower()
        if self.mode not in {"off", "shadow", "enforce"}:
            raise ValueError("unsupported barrier_mode: %s" % self.mode)

        self.registry = BarrierRegistry(terminal_retention_sec=terminal_retention_sec)
        self.lock = threading.RLock()
        self.observation_retry_sec = max(0.1, float(observation_retry_sec))
        self.last_observation_publish_monotonic = 0.0
        self.gripper_min_stable_count = max(1, int(gripper_min_stable_count))
        self.latest_gripper_state = None
        self.gripper_evidence_ready_request_id = ""
        self.latest_gripper_evidence = {}

        self.observation_pub = rospy.Publisher(
            observation_request_topic,
            String,
            queue_size=10,
            latch=False,
        )
        self.result_pub = rospy.Publisher(
            result_topic,
            String,
            queue_size=10,
            latch=False,
        )
        self.request_sub = rospy.Subscriber(
            request_topic,
            String,
            self.request_callback,
            queue_size=10,
        )
        self.gripper_state_sub = rospy.Subscriber(
            gripper_state_topic,
            GripperState,
            self.gripper_state_callback,
            queue_size=100,
        )

    @staticmethod
    def _semantic_signature(result) -> tuple:
        issues = []
        for issue in _issue_dict_list(result):
            issues.append(
                (
                    str(issue.get("issue_type", "")),
                    str(issue.get("severity", "")),
                    str(issue.get("subject_id", "")),
                    str(issue.get("relation", "")),
                    str(issue.get("object_id", "")),
                )
            )
        return bool(getattr(result, "passed", False)), tuple(sorted(issues))

    def _publish_result(self, result: ValidationResult) -> None:
        self.result_pub.publish(String(data=result.to_json()))

    def _publish_observation(self, request: ValidationRequest, event: str = "start") -> None:
        payload = request.to_dict()
        payload["event"] = str(event)
        self.observation_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=False, sort_keys=True))
        )
        self.last_observation_publish_monotonic = time.monotonic()

    def _finish(self, result: ValidationResult) -> None:
        self._publish_result(result)
        if result.terminal:
            self._publish_observation(
                ValidationRequest(
                    request_id=result.request_id,
                    context=result.context,
                    policy="cancel",
                    not_before=0.0,
                    required_distinct_frames=1,
                    timeout_sec=1.0,
                    created_stamp=result.timestamp,
                ),
                event="cancel",
            )

    @staticmethod
    def _gripper_evidence_dict(msg: GripperState) -> Dict[str, Any]:
        return {
            "monitor_instance_id": str(msg.monitor_instance_id or ""),
            "sample_seq": int(msg.sample_seq),
            "sample_start_stamp": float(msg.sample_start_stamp.to_sec()),
            "sample_end_stamp": float(msg.sample_end_stamp.to_sec()),
            "valid": bool(msg.valid),
            "states": list(msg.states or []),
            "stable_count": int(msg.stable_count),
            "holding_latched": bool(msg.holding_latched),
            "command_id": str(msg.command_id or ""),
            "command": str(msg.command or ""),
            "run_id": str(msg.run_id or ""),
            "attempt_id": int(msg.attempt_id),
            "step": int(msg.step),
            "reason": str(msg.reason or ""),
        }

    def _publish_gripper_failure_alert(
        self,
        request: ValidationRequest,
        evidence: Dict[str, Any],
    ) -> None:
        runtime_summary = {}
        try:
            runtime_summary = self.builder.get_runtime_observation_summary()
        except Exception:
            runtime_summary = {}
        runtime_summary = dict(runtime_summary or {})
        runtime_summary["direct_gripper_evidence"] = dict(evidence or {})
        runtime_summary["gripper_states"] = list(evidence.get("states", []) or [])
        actual = list(evidence.get("states", []) or [])
        payload = {
            "source": "validation_monitor/transaction",
            "stage_name": request.context.stage_name,
            "execution_step": _infer_execution_step_from_stage_name(request.context.stage_name),
            "severity": "error",
            "comparison_summary": {
                "stage_name": request.context.stage_name,
                "passed": False,
                "highest_severity": "error",
                "issue_count": 1,
                "issue_types": {"state_mismatch": 1},
                "issue_severity": {"error": 1},
                "compare_kind": "direct_gripper_evidence",
            },
            "issues": [{
                "issue_type": "state_mismatch",
                "severity": "error",
                "node_id": "gripper",
                "expected": ["holding"],
                "actual": actual,
                "message": "gripper holding missing after close command",
                "metadata": {
                    "missing_state": "holding",
                    "gripper_command_id": request.gripper_command_id,
                    "direct_gripper_evidence": dict(evidence or {}),
                },
            }],
            "expected_state_brief": "gripper requires holding",
            "graph_brief": "",
            "runtime_summary": runtime_summary,
            "message": "grasp_commit failed: stable holding evidence missing",
            "timestamp": time.time(),
        }
        payload.update(request.context.to_dict())
        payload["request_id"] = request.request_id
        publish_deviation_alert(self.alert_pub, payload)

    def _evaluate_gripper_state_locked(
        self,
        request: ValidationRequest,
        msg: Optional[GripperState],
    ) -> Optional[ValidationResult]:
        evidence = self._gripper_evidence_dict(msg) if msg is not None else {}
        self.latest_gripper_evidence = evidence
        status, _reason = evaluate_gripper_evidence_for_request(
            request,
            evidence,
            min_stable_count=self.gripper_min_stable_count,
        )
        if status == "passed":
            self.gripper_evidence_ready_request_id = request.request_id
            return None
        if status == "waiting":
            return None

        result = self.registry.complete(
            ValidationResult.for_request(
                request,
                status=ValidationStatus.FAILED,
                message="stable gripper evidence does not contain holding",
            ),
            monotonic_now=time.monotonic(),
        )
        if self.mode == "enforce":
            self._publish_gripper_failure_alert(request, evidence)
        return result

    def gripper_state_callback(self, msg: GripperState) -> None:
        with self.lock:
            self.latest_gripper_state = msg
            request = self.registry.active_request
            if request is None:
                return
            result = self._evaluate_gripper_state_locked(request, msg)
            if result is not None and result.terminal:
                self._finish(result)

    def request_callback(self, msg: String):
        raw = str(msg.data or "").strip()
        try:
            payload = json.loads(raw)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "[VALIDATION][TX] invalid request: %s", str(exc))
            return

        if str(payload.get("event", "start")).strip().lower() == "cancel":
            request_id = str(payload.get("request_id", "") or "").strip()
            with self.lock:
                cancelled = self.registry.cancel(
                    request_id,
                    monotonic_now=time.monotonic(),
                    message="cancelled by execution",
                )
                if cancelled is not None:
                    self._finish(cancelled)
            return

        try:
            request = ValidationRequest.from_dict(payload)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "[VALIDATION][TX] rejected request: %s", str(exc))
            return

        if self.mode == "off":
            self._publish_result(
                ValidationResult.for_request(
                    request,
                    status=ValidationStatus.UNAVAILABLE,
                    message="validation barriers are disabled",
                )
            )
            return

        with self.lock:
            response = self.registry.register(request, monotonic_now=time.monotonic())
            self._publish_result(response)
            if response.terminal:
                return
            self.gripper_evidence_ready_request_id = ""
            self._publish_observation(request, event="start")
            gripper_result = self._evaluate_gripper_state_locked(
                request,
                self.latest_gripper_state,
            )
            if gripper_result is not None and gripper_result.terminal:
                self._finish(gripper_result)

    def process_pending_snapshots(self) -> None:
        drain_fn = getattr(self.builder, "drain_transaction_snapshots", None)
        if not callable(drain_fn):
            return
        snapshots = drain_fn()
        for snapshot in snapshots:
            with self.lock:
                request = self.registry.active_request
                if request is None:
                    return
                if (
                    request.policy == "grasp_commit"
                    and self.gripper_evidence_ready_request_id != request.request_id
                ):
                    continue
                matched, _ = snapshot_matches_request(request, snapshot)
                if not matched:
                    continue
                bundle = self.process_registry.get(request.context.stage_name)
                if bundle is None:
                    result = self.registry.complete(
                        ValidationResult.for_request(
                            request,
                            status=ValidationStatus.INVALID,
                            message="unknown validation stage",
                        ),
                        monotonic_now=time.monotonic(),
                    )
                    self._finish(result)
                    return
                kind = self._POLICY_KIND.get(request.policy, "post")
                try:
                    expected_state, comparison = compare_bundle_state(
                        request.context.stage_name,
                        kind,
                        bundle,
                        snapshot.get("graph"),
                        self.comparator,
                    )
                except Exception as exc:
                    result = self.registry.complete(
                        ValidationResult.for_request(
                            request,
                            status=ValidationStatus.UNAVAILABLE,
                            message="transaction comparison failed: %s" % str(exc),
                        ),
                        monotonic_now=time.monotonic(),
                    )
                    self._finish(result)
                    return

                signature = self._semantic_signature(comparison)
                response = self.registry.observe(
                    snapshot,
                    semantic_signature=signature,
                    comparison_passed=bool(comparison.passed),
                    monotonic_now=time.monotonic(),
                )
                if response is not None and response.terminal:
                    if response.status == ValidationStatus.FAILED and self.mode == "enforce":
                        alert_payload = build_deviation_alert_dict(
                            request.context.stage_name,
                            expected_state,
                            snapshot.get("graph"),
                            comparison,
                            self.builder,
                            source="validation_monitor/transaction",
                        )
                        alert_payload.update(request.context.to_dict())
                        alert_payload["request_id"] = request.request_id
                        publish_deviation_alert(self.alert_pub, alert_payload)
                    self._finish(response)
                    return

    def tick(self) -> None:
        now = time.monotonic()
        with self.lock:
            expired = self.registry.expire(monotonic_now=now)
            if expired is not None:
                self._finish(expired)
                return
            request = self.registry.active_request
            if request is None:
                return
            if now - self.last_observation_publish_monotonic >= self.observation_retry_sec:
                self._publish_observation(request, event="start")


class BreakpointSearcher:
    """
    断点搜索器：接收修复完成后的断点搜索请求，扫描当前场景图，
    找到第一个未完成的步骤作为恢复断点。

    安全要求：
      - 必须使用时间戳晚于请求时间戳的新鲜场景数据；
      - request_id 必须匹配，防止 latch 残留消息误匹配。
    """

    def __init__(self, builder, comparator, process_registry, plan_dict=None):
        self.builder = builder
        self.comparator = comparator
        self.process_registry = process_registry
        self.plan_dict = plan_dict or {}
        self.breakpoint_confirm_count = max(1, int(rospy.get_param("~breakpoint_confirm_count", 1)))
        self.breakpoint_sample_timeout = max(0.1, float(rospy.get_param("~breakpoint_sample_timeout", 12.0)))
        self.breakpoint_wait_fresh_scene_timeout = max(
            self.breakpoint_sample_timeout,
            float(rospy.get_param("~breakpoint_wait_fresh_scene_timeout", 60.0)),
        )
        self.breakpoint_sample_interval = max(0.01, float(rospy.get_param("~breakpoint_sample_interval", 0.05)))
        self.breakpoint_require_event_alignment = bool(
            rospy.get_param("~breakpoint_require_event_alignment", True)
        )
        self.breakpoint_stamp_tolerance_sec = max(
            0.0,
            float(rospy.get_param("~breakpoint_stamp_tolerance_sec", 0.2)),
        )
        self.breakpoint_max_detection_age_sec = max(
            0.0,
            float(rospy.get_param("~breakpoint_max_detection_age_sec", 3.0)),
        )
        # TEMP BP DEBUG: enabled only while diagnosing stale/transition scene graphs.
        self.enable_breakpoint_debug = bool(rospy.get_param("~enable_breakpoint_debug", False))
        self.breakpoint_debug_node_limit = max(1, int(rospy.get_param("~breakpoint_debug_node_limit", 40)))
        self.breakpoint_debug_relation_limit = max(1, int(rospy.get_param("~breakpoint_debug_relation_limit", 80)))

        self._pending_request_id: Optional[str] = None
        self._pending_request_ts: float = 0.0
        self._pending_request_context: Dict[str, Any] = {}

        self.request_sub = rospy.Subscriber(
            TOPIC_BREAKPOINT_REQUEST,
            String,
            self._on_breakpoint_search_request,
            queue_size=5,
        )
        self.result_pub = rospy.Publisher(
            TOPIC_BREAKPOINT_RESULT,
            String,
            queue_size=5,
            latch=True,
        )
        rospy.logwarn("[VALIDATION] BreakpointSearcher started")

    def _on_breakpoint_search_request(self, msg: String):
        request_id = ""
        request_ts = 0.0
        try:
            payload = json.loads(str(msg.data or "{}"))
            request_id = str(payload.get("request_id", ""))
            request_ts = float(payload.get("timestamp", 0.0))
            request_context = {
                "run_id": str(payload.get("run_id", "") or ""),
                "attempt_id": payload.get("attempt_id"),
                "stage_seq": payload.get("stage_seq"),
            }
        except Exception:
            request_context = {}

        rospy.logwarn(
            "[VALIDATION] received breakpoint search request: request_id=%s ts=%.3f",
            request_id, request_ts,
        )

        self._pending_request_id = request_id
        self._pending_request_ts = request_ts
        self._pending_request_context = request_context

        try:
            resume_step = self.search_confirmed_breakpoint(request_ts)
            self._publish_result(resume_step, request_id)
        except Exception as exc:
            rospy.logerr("[VALIDATION] breakpoint search failed: %s", str(exc))
            self._publish_result(
                None,
                request_id,
                error=str(exc),
                error_code=getattr(exc, "error_code", "breakpoint_search_failed"),
            )

    @staticmethod
    def _to_sec(t) -> float:
        if hasattr(t, "to_sec"):
            return t.to_sec()
        return float(t)

    def _wait_for_fresh_scene(self, request_ts: float, timeout: float = 3.0) -> bool:
        """
        等待 builder 产出时间戳晚于 request_ts 的场景图。

        如果请求时场景图已经足够新，立即返回 True。
        否则使用 rospy.Rate(20) 轮询，最多等待 timeout 秒。
        """
        request_sec = self._to_sec(request_ts)
        start_monotonic = time.monotonic()
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            elapsed = time.monotonic() - start_monotonic
            if elapsed > timeout:
                rospy.logerr(
                    "[VALIDATION] timeout waiting for event-aligned scene: "
                    "request_ts=%.3f elapsed=%.1f",
                    request_sec, elapsed,
                )
                return False

            snapshot = get_scene_graph_snapshot(self.builder)
            stamp_sec = snapshot_graph_stamp_key(snapshot, self.builder)
            if isinstance(snapshot, dict) and stamp_sec is not None:
                aligned_ok, _ = validate_snapshot_event_alignment(
                    snapshot,
                    stamp_tolerance_sec=self.breakpoint_stamp_tolerance_sec,
                    reference_ts=request_sec,
                    require_event_alignment=self.breakpoint_require_event_alignment,
                    now_sec=float(rospy.Time.now().to_sec()),
                    max_detection_age_sec=self.breakpoint_max_detection_age_sec,
                )
                if stamp_sec > request_sec and aligned_ok:
                    rospy.logwarn(
                        "[VALIDATION] event-aligned scene: graph_stamp=%.3f request=%.3f det_stamp=%s",
                        stamp_sec,
                        request_sec,
                        str(snapshot.get("detection_stamp")),
                    )
                    return True

            rate.sleep()

        return False

    def _current_class_relation_counts(self, graph):
        """
        将当前 scene graph 转为类别级关系计数：
          ("cube", "on", "arch") -> 1
          ("cuboid", "on", "arch") -> 1

        忽略具体实例 ID。
        """
        id_to_class = {}

        for node in getattr(graph, "nodes", []) or []:
            try:
                node_id = node.uid()
            except Exception:
                node_id = getattr(node, "id", "")

            node_class = getattr(node, "name", None) or getattr(node, "class_name", None)
            if node_id and node_class:
                id_to_class[str(node_id)] = str(node_class)

        counts = {}

        get_relations = getattr(graph, "get_relations", None)
        edges = get_relations() if callable(get_relations) else []

        for edge in edges or []:
            try:
                s_id = edge.start.uid()
                o_id = edge.end.uid()
                rel = edge.edge_type
            except Exception:
                continue

            s_cls = id_to_class.get(str(s_id), "")
            o_cls = id_to_class.get(str(o_id), "")

            if not s_cls or not o_cls:
                continue

            key = (s_cls, str(rel), o_cls)
            counts[key] = counts.get(key, 0) + 1

        return counts

    def _step_required_relation(self, step_dict):
        """
        从 action.py 的 step 中提取该步完成后新增的结构关系。

        例如：
          target_class=cube
          expected_support_object_id=arch_1
        得到：
          ("cube", "on", "arch")
        """
        target_class = str(
            step_dict.get("target_class")
            or step_dict.get("class_name")
            or step_dict.get("object_class")
            or ""
        ).strip()

        support_class = str(step_dict.get("expected_support_class") or "").strip()

        if not support_class:
            sid = str(step_dict.get("expected_support_object_id") or "").strip()
            if sid.startswith("arch"):
                support_class = "arch"
            elif sid.startswith("cuboid"):
                support_class = "cuboid"
            elif sid.startswith("cube"):
                support_class = "cube"
            elif sid.startswith("triangle"):
                support_class = "triangle"

        if not target_class or not support_class:
            return None

        return (target_class, "on", support_class)

    def _iter_plan_steps(self):
        plan = getattr(self, "plan_dict", {}) or {}
        steps = list(plan.get("execution_steps", []) or [])
        steps.sort(key=lambda s: int(s.get("step", 0)))
        return steps

    def _breakpoint_debug_snapshot(self, snapshot, request_sec: float, resume_step) -> None:
        """TEMP BP DEBUG: print original-process graph/detection freshness details."""
        if not self.enable_breakpoint_debug:
            return

        if not isinstance(snapshot, dict):
            snapshot = {}

        graph = snapshot.get("graph")
        graph_stamp_sec = _to_sec_value(snapshot.get("graph_stamp_sec"))
        if graph_stamp_sec is None:
            graph_stamp_sec = _to_sec_value(snapshot.get("graph_stamp"))
        if graph_stamp_sec is None:
            graph_stamp_sec = 0.0

        payload = snapshot.get("detection_payload", {}) or {}
        if not isinstance(payload, dict):
            payload = {}

        det_stamp = _to_sec_value(snapshot.get("detection_stamp"))
        if det_stamp is None:
            det_stamp = _to_sec_value(payload.get("stamp"))
        if det_stamp is None:
            det_stamp = 0.0

        frame_index = snapshot.get("detection_frame_index", payload.get("frame_index", None))
        detections = payload.get("detections", []) or []
        if not isinstance(detections, list):
            detections = []

        in_build_detection_count = 0
        detection_class_counts = {}
        for det in detections:
            if not isinstance(det, dict):
                continue
            attrs = dict(det.get("attributes", {}) or {})
            build_region = dict(attrs.get("build_region", {}) or {})
            in_build = bool(det.get("in_build_region", build_region.get("in_build_region", False)))
            if in_build:
                in_build_detection_count += 1
            label = str(
                det.get("class_name")
                or det.get("label")
                or attrs.get("raw_label")
                or "unknown"
            )
            detection_class_counts[label] = detection_class_counts.get(label, 0) + 1

        node_parts = []
        state_counts = {}
        class_counts = {}
        for node in getattr(graph, "nodes", []) or []:
            try:
                node_id = node.uid()
            except Exception:
                node_id = getattr(node, "id", "")
            node_class = getattr(node, "name", None) or getattr(node, "class_name", None) or ""
            states = list(getattr(node, "states", []) or [])
            class_counts[str(node_class)] = class_counts.get(str(node_class), 0) + 1
            for state in states:
                state_counts[str(state)] = state_counts.get(str(state), 0) + 1
            if len(node_parts) < self.breakpoint_debug_node_limit:
                node_parts.append("%s(%s)%s" % (str(node_id), str(node_class), str(states)))

        relation_parts = []
        get_relations = getattr(graph, "get_relations", None)
        edges = get_relations() if callable(get_relations) else []
        for edge in edges or []:
            if len(relation_parts) >= self.breakpoint_debug_relation_limit:
                break
            try:
                relation_parts.append(
                    "%s--%s-->%s" % (
                        edge.start.uid(),
                        str(edge.edge_type),
                        edge.end.uid(),
                    )
                )
            except Exception:
                continue

        class_relation_counts = {
            str(key): value
            for key, value in self._current_class_relation_counts(graph).items()
        }

        # Use logerr so the temporary diagnostic is visible even when node log_level=ERROR.
        rospy.logerr(
            "[VALIDATION][BP_DIAG][TEMP] request_id=%s request_ts=%.3f "
            "graph_stamp=%.3f det_stamp=%.3f frame_index=%s "
            "graph_after_req=%.3f det_after_req=%.3f resume_step=%s "
            "detections=%d in_build_detections=%d",
            str(self._pending_request_id or ""),
            request_sec,
            graph_stamp_sec,
            det_stamp,
            str(frame_index),
            graph_stamp_sec - request_sec,
            det_stamp - request_sec,
            str(resume_step),
            len(detections),
            in_build_detection_count,
        )
        rospy.logerr(
            "[VALIDATION][BP_DIAG][TEMP] classes detection=%s graph=%s states=%s",
            json.dumps(detection_class_counts, ensure_ascii=False),
            json.dumps(class_counts, ensure_ascii=False),
            json.dumps(state_counts, ensure_ascii=False),
        )
        rospy.logerr(
            "[VALIDATION][BP_DIAG][TEMP] class_relations=%s",
            json.dumps(class_relation_counts, ensure_ascii=False),
        )
        rospy.logerr(
            "[VALIDATION][BP_DIAG][TEMP] nodes=%s",
            str(node_parts),
        )
        rospy.logerr(
            "[VALIDATION][BP_DIAG][TEMP] relations=%s",
            str(relation_parts),
        )

    def search_current_breakpoint(self) -> Optional[int]:
        snapshot = get_scene_graph_snapshot(self.builder)
        graph = snapshot.get("graph") if isinstance(snapshot, dict) else None
        return self.search_breakpoint_for_graph(graph)

    def search_breakpoint_for_graph(self, graph) -> Optional[int]:
        """
        基于当前场景图拓扑关系搜索断点。

        不使用：
          - target_place
          - process_registry leave_target
          - 旧实例 ID
          - gripper 状态

        只使用：
          - 当前 scene graph 中的 class-level relations
          - action.py 中每一步形成的 expected support relation
        """
        if graph is None:
            rospy.logerr("[VALIDATION] no scene graph available for breakpoint search")
            return None

        current_counts = self._current_class_relation_counts(graph)

        rospy.logwarn(
            "[VALIDATION] breakpoint class-level relations: %s",
            json.dumps({str(k): v for k, v in current_counts.items()}, ensure_ascii=False),
        )

        required_counts = {}
        last_step = 0

        for step in self._iter_plan_steps():
            step_num = int(step.get("step", 0))
            last_step = max(last_step, step_num)

            rel = self._step_required_relation(step)

            if rel is not None:
                required_counts[rel] = required_counts.get(rel, 0) + 1

            ok = True
            missing = []

            for key, need_count in required_counts.items():
                have_count = int(current_counts.get(key, 0))
                if have_count < need_count:
                    ok = False
                    missing.append((key, need_count, have_count))

            rospy.logwarn(
                "[VALIDATION] breakpoint topology check step=%d ok=%s missing=%s",
                step_num,
                str(ok),
                str(missing),
            )

            if not ok:
                rospy.logwarn(
                    "[VALIDATION] breakpoint found by topology: resume_step=%d valid_until_step=%d",
                    step_num,
                    step_num - 1,
                )
                return step_num

        rospy.logwarn(
            "[VALIDATION] all topology requirements satisfied; resume_step=%d",
            last_step + 1,
        )
        return last_step + 1

    def search_confirmed_breakpoint(self, request_ts: float) -> Optional[int]:
        """
        使用修复完成后的连续新鲜场景图确认 resume_step。
        只接受时间戳晚于 request_ts 的场景图，且同一帧不重复计数。
        """
        request_sec = self._to_sec(request_ts)
        if request_sec <= 0:
            raise RuntimeError("breakpoint search requires a valid request timestamp")

        wait_started_monotonic = time.monotonic()
        wait_deadline_monotonic = wait_started_monotonic + self.breakpoint_wait_fresh_scene_timeout
        confirm_deadline_monotonic = None
        seen_frame_keys = set()
        last_resume_step: Optional[int] = None
        consecutive_count = 0
        samples = []
        skipped_samples = []
        aligned_scene_seen = False

        while not rospy.is_shutdown():
            now_sec = self._to_sec(rospy.Time.now())
            now_monotonic = time.monotonic()
            active_deadline_monotonic = (
                confirm_deadline_monotonic
                if confirm_deadline_monotonic is not None
                else wait_deadline_monotonic
            )
            if now_monotonic > active_deadline_monotonic:
                break

            snapshot = get_scene_graph_snapshot(self.builder)
            graph = snapshot.get("graph") if isinstance(snapshot, dict) else None
            graph_stamp_sec = snapshot_graph_stamp_key(snapshot, self.builder)

            if graph is not None and graph_stamp_sec is not None:
                aligned_ok, alignment_reason = validate_snapshot_event_alignment(
                    snapshot,
                    stamp_tolerance_sec=self.breakpoint_stamp_tolerance_sec,
                    reference_ts=request_sec,
                    require_event_alignment=self.breakpoint_require_event_alignment,
                    now_sec=float(rospy.Time.now().to_sec()),
                    max_detection_age_sec=self.breakpoint_max_detection_age_sec,
                )
                if not aligned_ok:
                    skipped_samples.append((round(graph_stamp_sec, 3), alignment_reason))
                    det_stamp_for_log = _to_sec_value(snapshot.get("detection_stamp"))
                    det_age_for_log = None
                    if det_stamp_for_log is not None:
                        det_age_for_log = now_sec - det_stamp_for_log
                    rospy.logwarn_throttle(
                        1.0,
                        "[VALIDATION] breakpoint waiting event-aligned scene: reason=%s graph_stamp=%.3f "
                        "det_stamp=%s det_age=%s waited=%.1fs timeout=%.1fs request_id=%s",
                        str(alignment_reason),
                        graph_stamp_sec,
                        str(det_stamp_for_log),
                        str(det_age_for_log),
                        now_monotonic - wait_started_monotonic,
                        self.breakpoint_wait_fresh_scene_timeout,
                        str(self._pending_request_id or ""),
                    )
                    rospy.sleep(self.breakpoint_sample_interval)
                    continue

                if not aligned_scene_seen:
                    aligned_scene_seen = True
                    confirm_deadline_monotonic = now_monotonic + self.breakpoint_sample_timeout
                    rospy.logwarn(
                        "[VALIDATION] breakpoint event-aligned scene restored: request_id=%s "
                        "graph_stamp=%.3f det_stamp=%s confirm_timeout=%.1fs",
                        str(self._pending_request_id or ""),
                        graph_stamp_sec,
                        str(snapshot.get("detection_stamp")),
                        self.breakpoint_sample_timeout,
                    )

                det_stamp = _to_sec_value(snapshot.get("detection_stamp"))
                frame_index = snapshot.get("detection_frame_index")
                if det_stamp is not None:
                    frame_key = ("det", frame_index, round(float(det_stamp), 6))
                else:
                    frame_key = ("graph", round(float(graph_stamp_sec), 6))

                if frame_key in seen_frame_keys:
                    rospy.sleep(self.breakpoint_sample_interval)
                    continue

                seen_frame_keys.add(frame_key)
                resume_step = self.search_breakpoint_for_graph(graph)
                self._breakpoint_debug_snapshot(snapshot, request_sec, resume_step)
                samples.append((round(graph_stamp_sec, 3), snapshot.get("detection_stamp"), resume_step))

                if resume_step == last_resume_step:
                    consecutive_count += 1
                else:
                    last_resume_step = resume_step
                    consecutive_count = 1

                rospy.logwarn(
                    "[VALIDATION] breakpoint sample: graph_stamp=%.3f det_stamp=%s "
                    "resume_step=%s consecutive=%d/%d",
                    graph_stamp_sec,
                    str(snapshot.get("detection_stamp")),
                    str(resume_step),
                    consecutive_count,
                    self.breakpoint_confirm_count,
                )

                if consecutive_count >= self.breakpoint_confirm_count:
                    rospy.logwarn(
                        "[VALIDATION] confirmed breakpoint: resume_step=%s frames=%d request_id=%s",
                        str(resume_step),
                        consecutive_count,
                        str(self._pending_request_id or ""),
                    )
                    return resume_step

            rospy.sleep(self.breakpoint_sample_interval)

        if not aligned_scene_seen:
            raise VisionUnavailableError(
                "vision_unavailable: no event-aligned scene graph after %.1fs; request_id=%s samples=%s skipped=%s"
                % (
                    self.breakpoint_wait_fresh_scene_timeout,
                    str(self._pending_request_id or ""),
                    str(samples),
                    str(skipped_samples[-10:]),
                )
            )

        raise RuntimeError(
            "unable to confirm breakpoint from %d consecutive event-aligned detection frames within %.1fs; "
            "samples=%s skipped=%s"
            % (
                self.breakpoint_confirm_count,
                self.breakpoint_sample_timeout,
                str(samples),
                str(skipped_samples[-10:]),
            )
        )

    def _publish_result(
        self,
        resume_step: Optional[int],
        request_id: str = "",
        error: str = "",
        error_code: str = "",
    ):
        payload = {
            "request_id": request_id,
            "run_id": self._pending_request_context.get("run_id", ""),
            "attempt_id": self._pending_request_context.get("attempt_id"),
            "stage_seq": self._pending_request_context.get("stage_seq"),
            "success": error == "" and resume_step is not None,
            "resume_step": resume_step,
            "error": error,
            "error_code": str(error_code or ""),
            "timestamp": float(rospy.Time.now().to_sec()),
        }
        self.result_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        if error:
            rospy.logerr(
                "[VALIDATION] breakpoint result: error=%s error_code=%s request_id=%s",
                error,
                str(error_code or ""),
                request_id,
            )
        else:
            rospy.logwarn("[VALIDATION] breakpoint result: resume_step=%s request_id=%s", str(resume_step), request_id)


def main():
    rospy.init_node("validation_monitor", anonymous=True, log_level=rospy.ERROR)

    plan_file = str(rospy.get_param("~plan_file", DEFAULT_PLAN_FILE)).strip()
    task_plan_ref = str(rospy.get_param("~task_plan_ref", "")).strip()
    task_plan_root = str(rospy.get_param("~task_plan_root", "")).strip() or None
    requested_stage_name = str(rospy.get_param("~stage_name", "")).strip() or None

    robot_ip = str(rospy.get_param("~realman_ip", DEFAULT_REALMAN_IP)).strip()
    robot_port = int(rospy.get_param("~realman_port", DEFAULT_REALMAN_PORT))
    thread_mode = str(rospy.get_param("~realman_thread_mode", DEFAULT_REALMAN_THREAD_MODE)).strip()

    poll_interval = float(rospy.get_param("~poll_interval", 0.5))
    post_finish_delay = float(rospy.get_param("~post_finish_delay", 0.5))
    stable_graph_count = int(rospy.get_param("~stable_graph_count", 3))
    stream_mode = bool(rospy.get_param("~stream_mode", True))
    stream_interval = float(rospy.get_param("~stream_interval", 0.2))  # 5Hz
    stage_topic = str(rospy.get_param("~stage_topic", DEFAULT_STAGE_TOPIC)).strip()
    status_topic = str(rospy.get_param("~status_topic", DEFAULT_STATUS_TOPIC)).strip()
    use_external_scene_graph = bool(rospy.get_param("~use_external_scene_graph", True))
    scene_graph_topic = str(rospy.get_param("~scene_graph_topic", DEFAULT_SCENE_GRAPH_TOPIC)).strip()
    scene_graph_status_topic = str(
        rospy.get_param("~scene_graph_status_topic", DEFAULT_SCENE_GRAPH_STATUS_TOPIC)
    ).strip()
    alert_topic = str(rospy.get_param("~alert_topic", DEFAULT_ALERT_TOPIC)).strip()
    resolution_topic = str(rospy.get_param("~resolution_topic", DEFAULT_RESOLUTION_TOPIC)).strip()
    task_profile = str(rospy.get_param("~task_profile", DEFAULT_TASK_PROFILE)).strip()
    task_profile_root = str(rospy.get_param("~task_profile_root", "")).strip() or None
    enable_configured_validation = bool(rospy.get_param("~enable_configured_validation", True))
    validation_require_event_alignment = bool(
        rospy.get_param("~validation_require_event_alignment", True)
    )
    validation_stamp_tolerance_sec = max(
        0.0,
        float(rospy.get_param("~validation_stamp_tolerance_sec", 0.2)),
    )
    validation_max_detection_age_sec = max(
        0.0,
        float(rospy.get_param("~validation_max_detection_age_sec", 3.0)),
    )
    validation_wait_fresh_scene_timeout = max(
        0.0,
        float(rospy.get_param("~validation_wait_fresh_scene_timeout", 30.0)),
    )
    barrier_mode = str(rospy.get_param("~barrier_mode", "enforce")).strip().lower()
    validation_request_topic = str(
        rospy.get_param("~validation_request_topic", TOPIC_VALIDATION_REQUEST)
    ).strip()
    observation_request_topic = str(
        rospy.get_param("~observation_request_topic", TOPIC_OBSERVATION_REQUEST)
    ).strip()
    validation_result_topic = str(
        rospy.get_param("~validation_result_topic", TOPIC_VALIDATION_RESULT)
    ).strip()
    transaction_snapshot_queue_size = max(
        3,
        int(rospy.get_param("~transaction_snapshot_queue_size", 10)),
    )
    observation_retry_sec = max(
        0.1,
        float(rospy.get_param("~observation_retry_sec", 0.5)),
    )
    terminal_retention_sec = max(
        1.0,
        float(rospy.get_param("~terminal_retention_sec", 60.0)),
    )
    gripper_state_topic = str(
        rospy.get_param("~gripper_state_topic", DEFAULT_GRIPPER_STATE_TOPIC)
    ).strip()
    gripper_min_stable_count = max(
        1, int(rospy.get_param("~gripper_min_stable_count", 3))
    )

    if task_plan_ref:
        try:
            plan_dict = load_task_plan(task_plan_ref, task_plan_root=task_plan_root)
            rospy.logwarn("[VALIDATION] loaded task_plan_ref=%s", task_plan_ref)
        except Exception as exc:
            rospy.logwarn(
                "[VALIDATION] failed to load task_plan_ref=%s (%s); fallback to plan_file=%s",
                task_plan_ref,
                str(exc),
                plan_file,
            )
            plan_dict = build_internal_plan_from_file(plan_file)
    else:
        plan_dict = build_internal_plan_from_file(plan_file)
    if enable_configured_validation:
        try:
            process_registry = build_configured_process_supervision_registry(
                plan_dict,
                profile_name=task_profile or DEFAULT_TASK_PROFILE,
                profile_root=task_profile_root,
            )
            rospy.logwarn(
                "[VALIDATION] configured validation enabled: profile=%s stages=%d",
                task_profile or DEFAULT_TASK_PROFILE,
                len(process_registry),
            )
        except Exception as exc:
            rospy.logwarn(
                "[VALIDATION] configured validation failed (%s); fallback to legacy registry",
                str(exc),
            )
            process_registry = build_process_supervision_registry_from_execution_plan(plan_dict)
    else:
        rospy.logwarn("[VALIDATION] configured validation disabled; using legacy registry")
        process_registry = build_process_supervision_registry_from_execution_plan(plan_dict)
    process_stage_keys = set(process_registry.keys())
    manual_stage_name = requested_stage_name or None

    stage_tracker = StageTracker()
    recovery_context_tracker = RecoveryContextTracker()
    rospy.Subscriber(stage_topic, String, stage_tracker.stage_callback, queue_size=10)
    rospy.Subscriber(status_topic, String, stage_tracker.status_callback, queue_size=20)
    rospy.Subscriber(TOPIC_RECOVERY_RUNTIME_CONTEXT, String, recovery_context_tracker.callback, queue_size=20)
    alert_pub = rospy.Publisher(alert_topic, String, queue_size=20)
    resolution_pub = rospy.Publisher(resolution_topic, String, queue_size=20)

    if use_external_scene_graph:
        builder = ExternalSceneGraphSource(
            scene_graph_topic=scene_graph_topic,
            status_topic=scene_graph_status_topic,
            snapshot_queue_size=transaction_snapshot_queue_size,
        )
        rospy.logwarn(
            "[VALIDATION] using external scene graph snapshots; internal SceneGraphBuilder disabled"
        )
    else:
        builder = SceneGraphBuilder(init_node=False, gripper_state_provider=None)

    # validation_monitor 内部只使用场景图对象；
    # 不让 SceneGraphBuilder 单独持续打印完整 SceneGraph。
    # 控制台输出统一由 print_compare_output() 负责。
    builder.print_scene_graph = False
    if hasattr(builder, "print_scene_graph"):
        builder.print_scene_graph = False

    comparator = SceneGraphComparator(
        SceneGraphComparatorConfig(
            enable_build_validation_filter=True,
            allow_class_name_fallback=True,
            check_unexpected_nodes=False,
            treat_warning_as_failure=False,
        )
    )

    if barrier_mode != "off" and not use_external_scene_graph:
        rospy.logerr(
            "[VALIDATION][TX] barrier_mode=%s requires external versioned scene graphs; disabling barriers",
            barrier_mode,
        )
        barrier_mode = "off"

    transaction_coordinator = TransactionValidationCoordinator(
        builder=builder,
        comparator=comparator,
        process_registry=process_registry,
        alert_pub=alert_pub,
        mode=barrier_mode,
        request_topic=validation_request_topic,
        observation_request_topic=observation_request_topic,
        result_topic=validation_result_topic,
        observation_retry_sec=observation_retry_sec,
        terminal_retention_sec=terminal_retention_sec,
        gripper_state_topic=gripper_state_topic,
        gripper_min_stable_count=gripper_min_stable_count,
    )

    # 断点搜索器：响应修复后的断点搜索请求
    breakpoint_searcher = BreakpointSearcher(
        builder,
        comparator,
        process_registry,
        plan_dict=plan_dict,
    )

    last_pre_signature = None
    last_runtime_signature = None
    last_post_signature = None
    last_stream_time = 0.0
    last_stream_graph_stamp = None
    last_alert_signature = None
    last_resolution_signature = None
    validation_wait_registry = FreshSceneWaitRegistry(validation_wait_fresh_scene_timeout)

    try:
        while not rospy.is_shutdown():
            transaction_coordinator.process_pending_snapshots()
            transaction_coordinator.tick()

            if not builder.has_latest_scene_graph():
                wait_stage_name = (
                    manual_stage_name
                    or stage_tracker.current_stage_name
                    or stage_tracker.pending_started_stage_name
                    or stage_tracker.pending_finished_stage_name
                )
                if wait_stage_name:
                    last_alert_signature = handle_event_alignment_wait(
                        validation_wait_registry,
                        alert_pub,
                        last_alert_signature,
                        kind="scene_graph",
                        stage_name=wait_stage_name,
                        reason="missing_scene_graph_snapshot",
                        snapshot={},
                        now_sec=time.monotonic(),
                        timeout_sec=validation_wait_fresh_scene_timeout,
                        builder=builder,
                        source="validation_monitor/scene_graph_wait",
                    )
                time.sleep(poll_interval)
                continue

            snapshot = get_scene_graph_snapshot(builder)
            if not isinstance(snapshot, dict) or snapshot.get("graph") is None:
                wait_stage_name = (
                    manual_stage_name
                    or stage_tracker.current_stage_name
                    or stage_tracker.pending_started_stage_name
                    or stage_tracker.pending_finished_stage_name
                )
                if wait_stage_name:
                    last_alert_signature = handle_event_alignment_wait(
                        validation_wait_registry,
                        alert_pub,
                        last_alert_signature,
                        kind="scene_graph",
                        stage_name=wait_stage_name,
                        reason="missing_scene_graph_snapshot",
                        snapshot=snapshot if isinstance(snapshot, dict) else {},
                        now_sec=time.monotonic(),
                        timeout_sec=validation_wait_fresh_scene_timeout,
                        builder=builder,
                        source="validation_monitor/scene_graph_wait",
                    )
                time.sleep(poll_interval)
                continue

            graph = snapshot["graph"]
            graph_stamp = snapshot_graph_stamp_key(snapshot, builder)
            now_ros = float(rospy.Time.now().to_sec())
            now_monotonic = time.monotonic()
            if graph_stamp is None:
                graph_stamp = now_ros
            wait_stage_name = (
                manual_stage_name
                or stage_tracker.current_stage_name
                or stage_tracker.pending_started_stage_name
                or stage_tracker.pending_finished_stage_name
            )
            if wait_stage_name:
                mark_validation_scene_fresh(validation_wait_registry, "scene_graph", wait_stage_name)

            # started -> precondition
            pre_stage_name = None
            if manual_stage_name:
                pre_stage_name = manual_stage_name
                pre_compare_now = False
            else:
                pre_stage_name = stage_tracker.pending_started_stage_name
                pre_compare_now = False
                if pre_stage_name:
                    if stage_tracker.pending_started_baseline_graph_stamp is None:
                        stage_tracker.pending_started_baseline_graph_stamp = graph_stamp
                    elif graph_stamp != stage_tracker.pending_started_baseline_graph_stamp:
                        pre_compare_now = True
                    else:
                        pre_reference_ts = stage_tracker.pending_started_since
                        fresh_ok, stale_reason = validate_snapshot_event_alignment(
                            snapshot,
                            stamp_tolerance_sec=validation_stamp_tolerance_sec,
                            reference_ts=pre_reference_ts,
                            require_event_alignment=validation_require_event_alignment,
                            now_sec=float(rospy.Time.now().to_sec()),
                            max_detection_age_sec=validation_max_detection_age_sec,
                        )
                        if not fresh_ok:
                            last_alert_signature = handle_event_alignment_wait(
                                validation_wait_registry,
                                alert_pub,
                                last_alert_signature,
                                kind="pre",
                                stage_name=pre_stage_name,
                                reason=stale_reason,
                                snapshot=snapshot,
                                now_sec=now_monotonic,
                                timeout_sec=validation_wait_fresh_scene_timeout,
                                builder=builder,
                                source="validation_monitor/pre_wait",
                                reference_ts=pre_reference_ts,
                            )

            if pre_stage_name and pre_compare_now:
                if pre_stage_name not in process_stage_keys:
                    last_alert_signature = maybe_publish_unknown_stage_alert(
                        alert_pub,
                        last_alert_signature,
                        stage_name=pre_stage_name,
                        graph=graph,
                        builder=builder,
                        source="validation_monitor/pre_unknown_stage",
                    )
                    stage_tracker.clear_pending_started()
                else:
                    pre_reference_ts = None if manual_stage_name else stage_tracker.pending_started_since
                    fresh_ok, stale_reason = validate_snapshot_event_alignment(
                        snapshot,
                        stamp_tolerance_sec=validation_stamp_tolerance_sec,
                        reference_ts=pre_reference_ts,
                        require_event_alignment=validation_require_event_alignment,
                        now_sec=float(rospy.Time.now().to_sec()),
                        max_detection_age_sec=validation_max_detection_age_sec,
                    )
                    if not fresh_ok:
                        last_alert_signature = handle_event_alignment_wait(
                            validation_wait_registry,
                            alert_pub,
                            last_alert_signature,
                            kind="pre",
                            stage_name=pre_stage_name,
                            reason=stale_reason,
                            snapshot=snapshot,
                            now_sec=now_monotonic,
                            timeout_sec=validation_wait_fresh_scene_timeout,
                            builder=builder,
                            source="validation_monitor/pre_wait",
                            reference_ts=pre_reference_ts,
                        )
                    else:
                        mark_validation_scene_fresh(validation_wait_registry, "pre", pre_stage_name)
                        bundle = process_registry[pre_stage_name]
                        expected_state, result = compare_bundle_state(pre_stage_name, "pre", bundle, graph, comparator)
                        if expected_state is not None:
                            signature = (
                                pre_stage_name,
                                result.passed,
                                result.highest_severity(),
                                tuple((issue.issue_type, issue.message) for issue in result.issues),
                            )
                            if signature != last_pre_signature:
                                print_compare_output(pre_stage_name, expected_state, graph, result, tag="前提")
                                last_pre_signature = signature

                            last_alert_signature = maybe_publish_failure_alert(
                                alert_pub,
                                last_alert_signature,
                                stage_name=pre_stage_name,
                                expected_state=expected_state,
                                graph=graph,
                                result=result,
                                builder=builder,
                                source="validation_monitor/pre_compare",
                            )

                        stage_tracker.clear_pending_started()

            # runtime -> every stream_interval while stage is active
            if stream_mode:
                runtime_stage_name = manual_stage_name or stage_tracker.current_stage_name
                runtime_fresh_probe = None
                runtime_stale_reason = ""
                runtime_has_stale_scene = False
                if runtime_stage_name and runtime_stage_name in process_stage_keys:
                    runtime_fresh_probe, runtime_stale_reason = validate_snapshot_event_alignment(
                        snapshot,
                        stamp_tolerance_sec=validation_stamp_tolerance_sec,
                        reference_ts=None,
                        require_event_alignment=validation_require_event_alignment,
                        now_sec=float(rospy.Time.now().to_sec()),
                        max_detection_age_sec=validation_max_detection_age_sec,
                    )
                    runtime_has_stale_scene = not runtime_fresh_probe
                if (
                    runtime_stage_name
                    and (now_monotonic - last_stream_time) >= stream_interval
                    and (graph_stamp != last_stream_graph_stamp or runtime_has_stale_scene)
                ):
                    if runtime_stage_name not in process_stage_keys:
                        last_alert_signature = maybe_publish_unknown_stage_alert(
                            alert_pub,
                            last_alert_signature,
                            stage_name=runtime_stage_name,
                            graph=graph,
                            builder=builder,
                            source="validation_monitor/runtime_unknown_stage",
                            always_publish=True,
                        )
                        runtime_graph_consumed = True
                    else:
                        runtime_graph_consumed = False
                        if runtime_fresh_probe is None:
                            fresh_ok, stale_reason = validate_snapshot_event_alignment(
                                snapshot,
                                stamp_tolerance_sec=validation_stamp_tolerance_sec,
                                reference_ts=None,
                                require_event_alignment=validation_require_event_alignment,
                                now_sec=float(rospy.Time.now().to_sec()),
                                max_detection_age_sec=validation_max_detection_age_sec,
                            )
                        else:
                            fresh_ok = runtime_fresh_probe
                            stale_reason = runtime_stale_reason
                        if not fresh_ok:
                            last_alert_signature = handle_event_alignment_wait(
                                validation_wait_registry,
                                alert_pub,
                                last_alert_signature,
                                kind="runtime",
                                stage_name=runtime_stage_name,
                                reason=stale_reason,
                                snapshot=snapshot,
                                now_sec=now_monotonic,
                                timeout_sec=validation_wait_fresh_scene_timeout,
                                builder=builder,
                                source="validation_monitor/runtime_wait",
                            )
                        else:
                            mark_validation_scene_fresh(validation_wait_registry, "runtime", runtime_stage_name)
                            runtime_graph_consumed = True
                            bundle = process_registry[runtime_stage_name]
                            expected_state, result = compare_bundle_state(runtime_stage_name, "run", bundle, graph, comparator)
                            if expected_state is not None:
                                signature = (
                                    runtime_stage_name,
                                    result.passed,
                                    result.highest_severity(),
                                    tuple((issue.issue_type, issue.message) for issue in result.issues),
                                )
                                if signature != last_runtime_signature:
                                    print_compare_output(runtime_stage_name, expected_state, graph, result, tag="在线")
                                    last_runtime_signature = signature

                                last_alert_signature = maybe_publish_failure_alert(
                                    alert_pub,
                                    last_alert_signature,
                                    stage_name=runtime_stage_name,
                                    expected_state=expected_state,
                                    graph=graph,
                                    result=result,
                                    builder=builder,
                                    source="validation_monitor/runtime_compare",
                                    always_publish=True,
                                )

                    last_stream_time = now_monotonic
                    if runtime_graph_consumed:
                        last_stream_graph_stamp = graph_stamp

            # finished -> delayed stable-frame postcondition
            if manual_stage_name:
                post_stage_name = manual_stage_name
                post_compare_now = (graph_stamp != last_stream_graph_stamp)
            else:
                post_stage_name = stage_tracker.pending_finished_stage_name
                post_compare_now = False
                if post_stage_name and stage_tracker.pending_finished_since is not None:
                    if stage_tracker.pending_baseline_graph_stamp is None:
                        stage_tracker.pending_baseline_graph_stamp = graph_stamp
                        stage_tracker.pending_last_counted_graph_stamp = graph_stamp
                    else:
                        if graph_stamp != stage_tracker.pending_last_counted_graph_stamp:
                            stage_tracker.pending_finished_graph_count += 1
                            stage_tracker.pending_last_counted_graph_stamp = graph_stamp
                    elapsed = now_ros - stage_tracker.pending_finished_since
                    if elapsed >= post_finish_delay and stage_tracker.pending_finished_graph_count >= stable_graph_count:
                        post_compare_now = True
                    elif not post_compare_now:
                        post_reference_ts = stage_tracker.pending_finished_since
                        fresh_ok, stale_reason = validate_snapshot_event_alignment(
                            snapshot,
                            stamp_tolerance_sec=validation_stamp_tolerance_sec,
                            reference_ts=post_reference_ts,
                            require_event_alignment=validation_require_event_alignment,
                            now_sec=float(rospy.Time.now().to_sec()),
                            max_detection_age_sec=validation_max_detection_age_sec,
                        )
                        if not fresh_ok:
                            last_alert_signature = handle_event_alignment_wait(
                                validation_wait_registry,
                                alert_pub,
                                last_alert_signature,
                                kind="post",
                                stage_name=post_stage_name,
                                reason=stale_reason,
                                snapshot=snapshot,
                                now_sec=now_monotonic,
                                timeout_sec=validation_wait_fresh_scene_timeout,
                                builder=builder,
                                source="validation_monitor/post_wait",
                                reference_ts=post_reference_ts,
                            )

            if post_stage_name and post_compare_now:
                if post_stage_name not in process_stage_keys:
                    last_alert_signature = maybe_publish_unknown_stage_alert(
                        alert_pub,
                        last_alert_signature,
                        stage_name=post_stage_name,
                        graph=graph,
                        builder=builder,
                        source="validation_monitor/post_unknown_stage",
                    )
                    if not manual_stage_name:
                        stage_tracker.clear_pending_finished()
                else:
                    post_reference_ts = None if manual_stage_name else stage_tracker.pending_finished_since
                    fresh_ok, stale_reason = validate_snapshot_event_alignment(
                        snapshot,
                        stamp_tolerance_sec=validation_stamp_tolerance_sec,
                        reference_ts=post_reference_ts,
                        require_event_alignment=validation_require_event_alignment,
                        now_sec=float(rospy.Time.now().to_sec()),
                        max_detection_age_sec=validation_max_detection_age_sec,
                    )
                    if not fresh_ok:
                        last_alert_signature = handle_event_alignment_wait(
                            validation_wait_registry,
                            alert_pub,
                            last_alert_signature,
                            kind="post",
                            stage_name=post_stage_name,
                            reason=stale_reason,
                            snapshot=snapshot,
                            now_sec=now_monotonic,
                            timeout_sec=validation_wait_fresh_scene_timeout,
                            builder=builder,
                            source="validation_monitor/post_wait",
                            reference_ts=post_reference_ts,
                        )
                    else:
                        mark_validation_scene_fresh(validation_wait_registry, "post", post_stage_name)
                        bundle = process_registry[post_stage_name]
                        expected_state, result = compare_bundle_state(post_stage_name, "post", bundle, graph, comparator)
                        if expected_state is not None:
                            signature = (
                                post_stage_name,
                                result.passed,
                                result.highest_severity(),
                                tuple((issue.issue_type, issue.message) for issue in result.issues),
                            )
                            if signature != last_post_signature:
                                print_compare_output(post_stage_name, expected_state, graph, result, tag="最终")
                                last_post_signature = signature

                            last_alert_signature = maybe_publish_failure_alert(
                                alert_pub,
                                last_alert_signature,
                                stage_name=post_stage_name,
                                expected_state=expected_state,
                                graph=graph,
                                result=result,
                                builder=builder,
                                source="validation_monitor/post_compare",
                            )

                            last_resolution_signature = maybe_publish_recovery_resolution(
                                resolution_pub,
                                last_resolution_signature,
                                stage_name=post_stage_name,
                                graph=graph,
                                result=result,
                                builder=builder,
                                context=recovery_context_tracker.context,
                                source="validation_monitor/recovery_post_compare",
                            )

                        if not manual_stage_name:
                            stage_tracker.clear_pending_finished()

            time.sleep(poll_interval)

    finally:
        # Gripper SDK ownership belongs exclusively to gripper_state_monitor.
        pass


if __name__ == "__main__":
    main()
