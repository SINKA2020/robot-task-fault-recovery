#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import rospy
from std_msgs.msg import String
from typing import Optional

from scene_graph_system.diagnosis.config_driven_diagnoser import ConfigDrivenDiagnoser, load_config_driven_diagnoser
from scene_graph_system.diagnosis.fault_event import (
    TOPIC_DEVIATION_ALERT,
    TOPIC_DIAGNOSIS_REPORT,
    DeviationAlert,
    DiagnosisReport,
    RecoveryRuntimeContext,
)


DEFAULT_TASK_PROFILE = "block_building"


def _stage_action_type(stage_name: str) -> str:
    text = str(stage_name or "")
    if "__" in text:
        return text.split("__", 1)[1].strip()
    return text.strip()


def _issue_types(alert: DeviationAlert):
    return {
        str(item.get("issue_type", "") or "").strip()
        for item in list(alert.issues or [])
        if str(item.get("issue_type", "") or "").strip()
    }


def _is_gripper_issue(issue: dict) -> bool:
    node_id = str(issue.get("node_id", "") or "").strip().lower()
    subject_id = str(issue.get("subject_id", "") or "").strip().lower()
    object_id = str(issue.get("object_id", "") or "").strip().lower()
    msg = str(issue.get("message", "") or "").lower()
    return node_id == "gripper" or subject_id == "gripper" or object_id == "gripper" or "gripper" in msg


def _is_missing_holding_or_closed(issue: dict) -> bool:
    if not _is_gripper_issue(issue):
        return False
    msg = str(issue.get("message", "") or "").lower()
    expected = str(issue.get("expected", "") or "").lower()
    metadata = dict(issue.get("metadata", {}) or {})
    missing_state = str(metadata.get("missing_state", "") or "").lower()
    return (
        "holding" in msg
        or "closed" in msg
        or "holding" in expected
        or "closed" in expected
        or missing_state in {"holding", "closed"}
    )


def _has_gripper_holding_fault(alert: DeviationAlert) -> bool:
    for issue in list(alert.issues or []):
        if str(issue.get("issue_type", "") or "") == "state_mismatch":
            if _is_missing_holding_or_closed(issue):
                return True
    return False


def _gripper_evidence_disposition(runtime: dict) -> str:
    direct = dict((runtime or {}).get("direct_gripper_evidence", {}) or {})
    if direct:
        if not bool(direct.get("valid", False)):
            return "unknown"
        states = set(direct.get("states", []) or [])
        if "unknown" in states:
            return "unknown"
        if "holding" in states or bool(direct.get("holding_latched", False)):
            return "held"
        if states & {"open", "closed"}:
            return "not_held"
        return "unknown"

    states = set((runtime or {}).get("gripper_states", []) or [])
    alignment = dict((runtime or {}).get("gripper_alignment", {}) or {})
    if "holding" in states:
        return "held"
    if bool(alignment.get("valid", False)) and states & {"open", "closed"}:
        return "not_held"
    return "unknown"


def _diagnose_alert_legacy(alert: DeviationAlert) -> DiagnosisReport:
    issue_types = _issue_types(alert)
    runtime = dict(alert.runtime_summary or {})
    gripper_states = set(runtime.get("gripper_states", []) or [])
    grasped_ids = list(runtime.get("grasped_object_ids", []) or [])
    placed_ids = list(runtime.get("placed_object_ids", []) or [])
    gripper_disposition = _gripper_evidence_disposition(runtime)

    stage_name = str(alert.stage_name or "")
    stage_lower = stage_name.lower()
    action_type = _stage_action_type(stage_name)

    fault_type = "unknown_fault"
    root_cause = "未命中规则"
    confidence = 0.45
    retryable = True

    if "unknown_stage" in issue_types:
        fault_type = "stage_sync_error"
        root_cause = "执行阶段名与 ExpectedState registry 不一致"
        confidence = 0.95
        retryable = False

    elif action_type == "close_gripper":
        if _has_gripper_holding_fault(alert) or gripper_disposition == "not_held":
            fault_type = "pick_failed"
            root_cause = "夹爪闭合后未形成 holding / grasped 状态，疑似抓空或抓取失败"
            confidence = 0.90

    elif action_type == "lift_object":
        if _has_gripper_holding_fault(alert) or gripper_disposition == "not_held":
            fault_type = "pick_failed"
            root_cause = "抓取后上抬阶段未保持 holding，疑似未夹住或刚抓起即失败"
            confidence = 0.88

    elif action_type in {"move_to_target_above", "descend_to_place"}:
        if _has_gripper_holding_fault(alert) or gripper_disposition == "not_held":
            fault_type = "object_dropped"
            root_cause = "搬运/放置前阶段夹爪未保持 holding，疑似中途掉落"
            confidence = 0.88

    elif action_type in {"open_gripper", "leave_target"}:
        if "missing_relation" in issue_types:
            fault_type = "support_relation_not_formed"
            root_cause = "释放/离开目标后未形成预期支撑关系"
            confidence = 0.87
        elif "missing_node" in issue_types or len(placed_ids) == 0:
            fault_type = "place_not_completed"
            root_cause = "放置结束后目标结构节点缺失或未观察到 placed 状态"
            confidence = 0.82

    elif "_pick_" in stage_lower:
        if _has_gripper_holding_fault(alert) or gripper_disposition == "not_held":
            fault_type = "pick_failed"
            root_cause = "抓取结束后夹爪未保持 holding，且没有观测到 grasped 目标"
            confidence = 0.90

    elif "_move_" in stage_lower:
        if _has_gripper_holding_fault(alert) or gripper_disposition == "not_held":
            fault_type = "object_dropped"
            root_cause = "搬运阶段夹爪未保持 holding，疑似中途掉落"
            confidence = 0.88

    elif (
        "_place_" in stage_lower
        or stage_lower.endswith("_place")
        or "__descend_to_place" in stage_lower
        or "__open_gripper" in stage_lower
        or "__leave_target" in stage_lower
    ):
        if "missing_relation" in issue_types:
            fault_type = "support_relation_not_formed"
            root_cause = "放置后未形成预期支撑关系"
            confidence = 0.87
        elif "missing_node" in issue_types or len(placed_ids) == 0:
            fault_type = "place_not_completed"
            root_cause = "放置阶段结束后未观察到 placed 状态"
            confidence = 0.80

    if "forbidden_relation" in issue_types:
        fault_type = "structure_collision_or_intersection"
        root_cause = "检测到禁止关系，疑似结构碰撞、穿插或严重接触异常"
        confidence = max(confidence, 0.90)

    report = DiagnosisReport(
        alert_id=alert.alert_id,
        stage_name=alert.stage_name,
        execution_step=alert.execution_step,
        severity=alert.severity,
        fault_type=fault_type,
        root_cause=root_cause,
        confidence=confidence,
        retryable=retryable,
        details={
            "issue_types": sorted(list(issue_types)),
            "runtime_summary": runtime,
            "action_type": action_type,
            "diagnosis_mode": "legacy",
        },
    )
    return report


def diagnose_alert(
    alert: DeviationAlert,
    runtime_context: Optional[RecoveryRuntimeContext] = None,
    *,
    config_diagnoser: Optional[ConfigDrivenDiagnoser] = None,
    profile_name: str = DEFAULT_TASK_PROFILE,
    profile_root: Optional[str] = None,
    enable_configured_diagnosis: bool = True,
) -> DiagnosisReport:
    if enable_configured_diagnosis:
        try:
            diagnoser = config_diagnoser or load_config_driven_diagnoser(
                profile_name=profile_name or DEFAULT_TASK_PROFILE,
                profile_root=profile_root,
            )
            report = diagnoser.try_diagnose(alert, runtime_context)
            if report is not None:
                return report

            legacy = _diagnose_alert_legacy(alert)
            legacy.details["configured_diagnosis_no_match"] = True
            legacy.details["diagnosis_mode"] = "legacy_fallback"
            legacy.details["profile_id"] = getattr(getattr(diagnoser, "profile", None), "profile_id", "")
            return legacy
        except Exception as exc:
            legacy = _diagnose_alert_legacy(alert)
            legacy.details["configured_diagnosis_error"] = str(exc)
            legacy.details["diagnosis_mode"] = "legacy_fallback"
            return legacy

    return _diagnose_alert_legacy(alert)


def _create_config_diagnoser_from_ros_params():
    task_profile = str(rospy.get_param("~task_profile", DEFAULT_TASK_PROFILE)).strip()
    task_profile_root = str(rospy.get_param("~task_profile_root", "")).strip() or None
    enable_configured_diagnosis = bool(rospy.get_param("~enable_configured_diagnosis", True))

    if not enable_configured_diagnosis:
        rospy.logwarn("[DIAGNOSIS] configured diagnosis disabled; using legacy diagnoser")
        return None, task_profile, task_profile_root, False

    try:
        diagnoser = load_config_driven_diagnoser(
            profile_name=task_profile or DEFAULT_TASK_PROFILE,
            profile_root=task_profile_root,
        )
        rospy.logwarn(
            "[DIAGNOSIS] configured diagnosis enabled: profile=%s rules=%d",
            diagnoser.profile.profile_id,
            len(diagnoser.rules),
        )
        return diagnoser, task_profile, task_profile_root, True
    except Exception as exc:
        rospy.logwarn("[DIAGNOSIS] configured diagnosis failed (%s); using legacy diagnoser", str(exc))
        return None, task_profile, task_profile_root, False


class FaultDiagnoserNode:
    def __init__(self):
        (
            self.config_diagnoser,
            self.task_profile,
            self.task_profile_root,
            self.enable_configured_diagnosis,
        ) = _create_config_diagnoser_from_ros_params()
        self.pub = rospy.Publisher(TOPIC_DIAGNOSIS_REPORT, String, queue_size=20)
        rospy.Subscriber(TOPIC_DEVIATION_ALERT, String, self.alert_callback, queue_size=20)

    def alert_callback(self, msg: String):
        raw = str(msg.data).strip()
        if not raw:
            return
        try:
            alert = DeviationAlert.from_json(raw)
            report = diagnose_alert(
                alert,
                config_diagnoser=self.config_diagnoser,
                profile_name=self.task_profile,
                profile_root=self.task_profile_root,
                enable_configured_diagnosis=self.enable_configured_diagnosis,
            )
            self.pub.publish(String(data=report.to_json()))
            rospy.logwarn(
                "Diagnosis published: stage=%s action=%s fault_type=%s confidence=%.2f mode=%s",
                report.stage_name,
                report.details.get("action_type", ""),
                report.fault_type,
                report.confidence,
                report.details.get("diagnosis_mode", ""),
            )
        except Exception as exc:
            rospy.logerr("fault_diagnoser failed: %s", str(exc))


def main():
    rospy.init_node("fault_diagnoser", anonymous=True)
    FaultDiagnoserNode()
    rospy.spin()


if __name__ == "__main__":
    main()
