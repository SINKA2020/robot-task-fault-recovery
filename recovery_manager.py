#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from scene_graph_system.resources import package_resource_path

import os
import re
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


import rospy
from std_msgs.msg import String
from scene_graph_system.msg import GripperState

from scene_graph_system.diagnosis.fault_event import (
    TOPIC_DEVIATION_ALERT,
    TOPIC_DIAGNOSIS_REPORT,
    TOPIC_RECOVERY_PLAN,
    TOPIC_RECOVERY_FEEDBACK,
    TOPIC_RECOVERY_RUNTIME_CONTEXT,
    DeviationAlert,
    RecoveryPlan,
    RecoveryFeedback,
    RecoveryRuntimeContext,
)
from scene_graph_system.diagnosis.config_driven_diagnoser import load_config_driven_diagnoser
from scene_graph_system.recovery.config_driven_recovery_planner import load_config_driven_recovery_planner
from scene_graph_system.diagnosis.fault_diagnoser import diagnose_alert
from scene_graph_system.robot.robot_recovery_adapter import RobotRecoveryAdapter
from scene_graph_system.validation.validation_transaction import event_context_matches


DEFAULT_PLAN_FILE = package_resource_path('scripts/action.py')
DEFAULT_CLEANUP_CONFIG_FILE = package_resource_path('config/cleanup_config.yaml')
DEFAULT_TASK_PROFILE = "block_building"
DEFAULT_GRIPPER_STATE_TOPIC = "/gripper/state"

GRIPPER_HELD = "held"
GRIPPER_NOT_HELD = "not_held"
GRIPPER_UNKNOWN = "unknown"


def _resolve_current_step(context, alert):
    """统一解析当前步骤，context 优先，alert 兜底。"""
    if context is not None and getattr(context, "step", None) is not None:
        return int(context.step)
    return int(getattr(alert, "execution_step", 1) or 1)


_SAFE_RETREAT_ACTION = {
    "action_type": "safe_retreat",
    "params": {
        "lift_z": 0.08,
        "lift_speed": 0.08,
        "move_ready": True,
        "release_object": False,
        "lift_wait": 1.0,
        "wait_after": 0.5,
    },
}


@dataclass
class RecoveryDecision:
    recoverable: bool = False
    reason: str = ""
    kind: str = "ignored"
    resume_step: Optional[int] = None
    failed_steps: List[int] = field(default_factory=list)
    issue_step_debug: List[dict] = field(default_factory=list)
    confirm_count: int = 1
    need_return_held_object: bool = False
    held_object_disposition: str = GRIPPER_UNKNOWN
    safety_block_reason: str = ""
    severity: str = ""


HISTORY_REPAIR_KINDS = {
    "history_structure_failed",
    "history_node_missing",
}


def _int_set(values: Iterable) -> set:
    out = set()
    for value in values or []:
        try:
            out.add(int(value))
        except Exception:
            continue
    return out


def is_no_candidate_scene_verification_eligible(
    repair_result,
    requested_failed_steps,
) -> Tuple[bool, str]:
    """Return whether a failed history repair may use the scene fallback.

    The fallback is deliberately narrow: at least one repair detail must be a
    ``no_candidate`` outcome, and every other detail must be either an already
    completed repair or an already-verified step.  Motion, gripper, execution,
    plan, and generic failures therefore keep their existing fail-closed
    behavior.
    """
    requested_steps = _int_set(requested_failed_steps)
    if not requested_steps:
        return False, "missing_requested_failed_steps"
    if not isinstance(repair_result, dict):
        return False, "missing_repair_result"
    if bool(repair_result.get("success", False)):
        return False, "repair_result_already_successful"

    details = list(repair_result.get("details", []) or [])
    statuses = []
    for detail in details:
        if not isinstance(detail, dict):
            return False, "repair_result_has_unstructured_detail"
        status = str(detail.get("status", "") or "").strip().lower()
        if not status:
            return False, "repair_result_detail_missing_status"
        statuses.append(status)

    if "no_candidate" not in statuses:
        return False, "repair_result_not_no_candidate"

    allowed_statuses = {"repaired", "already_satisfied", "no_candidate"}
    unsafe_statuses = sorted(set(statuses) - allowed_statuses)
    if unsafe_statuses:
        return False, "repair_result_has_non_candidate_failure:%s" % ",".join(unsafe_statuses)

    reported_steps = _int_set(repair_result.get("repaired_steps", []))
    reported_steps.update(_int_set(repair_result.get("failed_steps", [])))
    if reported_steps and not reported_steps.issubset(requested_steps):
        return False, "repair_result_contains_unrequested_steps"

    return True, "no_candidate_requires_scene_verification"


def resolve_no_candidate_scene_verification(
    *,
    requested_failed_steps,
    repair_result,
    breakpoint_result,
) -> Tuple[bool, Optional[int], str, List[int]]:
    """Resolve a no-candidate repair only from a fresh breakpoint result.

    ``validation_monitor`` returns the first incomplete structural step.  A
    breakpoint strictly after every requested failed step therefore proves,
    under the same cumulative topology model used for normal resume, that all
    those historical steps are currently satisfied.
    """
    eligible, reason = is_no_candidate_scene_verification_eligible(
        repair_result,
        requested_failed_steps,
    )
    if not eligible:
        return False, None, reason, []

    requested_steps = _int_set(requested_failed_steps)
    if not isinstance(breakpoint_result, dict) or not bool(
        breakpoint_result.get("success", False)
    ):
        return False, None, "breakpoint_verification_failed", []

    try:
        resume_step = int(breakpoint_result.get("resume_step"))
    except (TypeError, ValueError):
        return False, None, "breakpoint_verification_missing_resume_step", []

    last_requested_step = max(requested_steps)
    if resume_step <= last_requested_step:
        return False, resume_step, "breakpoint_does_not_clear_all_failed_steps", []

    repaired_steps = _int_set(repair_result.get("repaired_steps", []))
    already_satisfied_steps = sorted(requested_steps - repaired_steps)
    return (
        True,
        resume_step,
        "all_failed_steps_satisfied_by_fresh_scene",
        already_satisfied_steps,
    )


def resolve_history_direct_resume_step(
    *,
    decision_kind: str,
    current_step: Optional[int],
    requested_failed_steps,
    repair_result,
    only_adjacent: bool = True,
) -> Tuple[bool, Optional[int], str]:
    """
    Resolve resume_step from confirmed history repair results.

    This intentionally does not inspect the scene graph. It only trusts the
    direct-resume path when the repair executor reports that every requested
    historical failed step was repaired and no failed steps remain.
    """
    if str(decision_kind or "").strip() not in HISTORY_REPAIR_KINDS:
        return False, None, "not_history_repair"

    requested_steps = _int_set(requested_failed_steps)
    if not requested_steps:
        return False, None, "missing_requested_failed_steps"

    if not isinstance(repair_result, dict):
        return False, None, "missing_repair_result"

    remaining_failed_steps = _int_set(repair_result.get("failed_steps", []))
    if remaining_failed_steps:
        return False, None, "repair_result_has_remaining_failed_steps"

    if not bool(repair_result.get("success", False)):
        return False, None, "repair_result_not_successful"

    repaired_steps = _int_set(repair_result.get("repaired_steps", []))
    if not requested_steps.issubset(repaired_steps):
        return False, None, "repair_result_missing_requested_steps"

    resume_step = max(requested_steps) + 1

    if only_adjacent:
        try:
            current_step_int = int(current_step)
        except Exception:
            return False, None, "missing_current_step"
        if resume_step != current_step_int:
            return False, None, "history_boundary_not_adjacent"

    return True, int(resume_step), "history_repair_complete"


def _canonical_id(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    return text


def _safe_int(value, default=None):
    try:
        return int(value)
    except Exception:
        return default


def _parse_relation_from_issue_message(message: str):
    text = str(message or "")
    m = re.search(r"([A-Za-z]+_\d+)\s*--\s*([A-Za-z_]+)\s*-->\s*([A-Za-z]+_\d+)", text)
    if not m:
        return None, None, None
    return _canonical_id(m.group(1)), str(m.group(2)).strip(), _canonical_id(m.group(3))


def _build_step_indices_from_action_plan(plan_file: str):
    relation_to_step: Dict[Tuple[str, str, str], int] = {}
    target_to_step: Dict[str, int] = {}
    step_to_target: Dict[int, dict] = {}
    try:
        from scene_graph_system.planning.task_plan_adapter import build_internal_plan_from_file
        plan = build_internal_plan_from_file(plan_file)
        execution_steps = list(plan.get("execution_steps", []) or [])
    except Exception as exc:
        rospy.logwarn("[RECOVERY] failed to load action plan from %s: %s", plan_file, str(exc))
        return relation_to_step, target_to_step, step_to_target

    for step in execution_steps:
        step_num = _safe_int(step.get("step"))
        if step_num is None:
            continue
        target_id = _canonical_id(step.get("target_object_id"))
        support_id = _canonical_id(
            step.get("expected_support_object_id")
            or step.get("target_support_object_id")
            or step.get("on")
        )
        step_to_target[step_num] = dict(step)
        if target_id:
            target_to_step[target_id] = step_num
        if target_id and support_id:
            relation_to_step[(target_id, "on", support_id)] = step_num
    return relation_to_step, target_to_step, step_to_target


def _issue_to_failed_step(issue: dict, relation_to_step: dict, target_to_step: dict):
    issue_type = str(issue.get("issue_type", "") or "").strip()
    if issue_type in {
        "missing_relation",
        "forbidden_relation",
        "relation_endpoint_missing",
        "relation_endpoint_class_missing",
    }:
        subject_id = _canonical_id(issue.get("subject_id"))
        object_id = _canonical_id(issue.get("object_id"))
        relation = str(issue.get("relation") or "").strip()
        if subject_id and object_id and relation:
            step_num = relation_to_step.get((subject_id, relation, object_id))
            if step_num is not None:
                return step_num
        subject_id, relation, object_id = _parse_relation_from_issue_message(issue.get("message", ""))
        if subject_id and object_id and relation:
            return relation_to_step.get((subject_id, relation, object_id))
        return None

    if issue_type in {"missing_node", "class_mismatch"}:
        node_id = _canonical_id(issue.get("node_id"))
        if node_id:
            return target_to_step.get(node_id)
        metadata = dict(issue.get("metadata", {}) or {})
        for key in ("logical_node_id", "expected_node_id", "target_object_id"):
            node_id = _canonical_id(metadata.get(key))
            if node_id:
                return target_to_step.get(node_id)
        return None
    return None


def infer_resume_step_from_alert(alert: DeviationAlert, context: Optional[RecoveryRuntimeContext], plan_file: str):
    relation_to_step, target_to_step, _ = _build_step_indices_from_action_plan(plan_file)
    failed_steps: List[int] = []
    issue_step_debug = []
    for issue in list(alert.issues or []):
        step_num = _issue_to_failed_step(issue, relation_to_step, target_to_step)
        issue_step_debug.append({
            "issue_type": issue.get("issue_type"),
            "subject_id": issue.get("subject_id"),
            "object_id": issue.get("object_id"),
            "node_id": issue.get("node_id"),
            "relation": issue.get("relation"),
            "message": issue.get("message"),
            "mapped_step": step_num,
        })
        if step_num is not None:
            failed_steps.append(int(step_num))
    failed_steps = sorted(set(failed_steps))
    if failed_steps:
        resume_step = min(failed_steps)
    elif context is not None and context.step is not None:
        resume_step = int(context.step)
    elif alert.execution_step is not None:
        resume_step = int(alert.execution_step)
    else:
        resume_step = 1
    return int(resume_step), failed_steps, issue_step_debug


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
        if str(issue.get("issue_type", "") or "") == "state_mismatch" and _is_missing_holding_or_closed(issue):
            return True
    return False


def _context_source_pose(context: Optional[RecoveryRuntimeContext]):
    if context is None:
        return None
    return getattr(context, "source_pose", None)


def _context_safe_return_poses(context: Optional[RecoveryRuntimeContext]):
    if context is None:
        return None, None, None
    return (
        getattr(context, "source_return_approach_pose", None),
        getattr(context, "source_return_release_pose", None),
        getattr(context, "source_return_lift_pose", None),
    )


def _runtime_holding_disposition(
    alert: DeviationAlert,
    direct_evidence: Optional[dict] = None,
) -> str:
    """Return held/not-held/unknown without converting missing evidence to false."""
    try:
        evidence = dict(direct_evidence or {})
        if evidence:
            if not bool(evidence.get("valid", False)):
                return GRIPPER_UNKNOWN
            states = set(evidence.get("states", []) or [])
            stable_count = _safe_int(evidence.get("stable_count"), 0) or 0
            min_stable_count = _safe_int(evidence.get("min_stable_count"), 1) or 1
            if stable_count < min_stable_count or "unknown" in states:
                return GRIPPER_UNKNOWN
            if "holding" in states or bool(evidence.get("holding_latched", False)):
                return GRIPPER_HELD
            if states & {"open", "closed"}:
                return GRIPPER_NOT_HELD
            return GRIPPER_UNKNOWN

        runtime = dict(getattr(alert, "runtime_summary", {}) or {})
        direct = runtime.get("direct_gripper_evidence")
        if isinstance(direct, dict) and direct:
            return _runtime_holding_disposition(alert, direct)

        gripper_states = set(runtime.get("gripper_states", []) or [])
        grasped_ids = list(runtime.get("grasped_object_ids", []) or [])
        alignment = dict(runtime.get("gripper_alignment", {}) or {})
        if "holding" in gripper_states or len(grasped_ids) > 0:
            return GRIPPER_HELD
        if bool(alignment.get("valid", False)) and gripper_states & {"open", "closed"}:
            return GRIPPER_NOT_HELD
        return GRIPPER_UNKNOWN
    except Exception:
        return GRIPPER_UNKNOWN


def _runtime_holding_evidence(alert: DeviationAlert) -> bool:
    """Compatibility helper: only explicit held evidence returns True."""
    return _runtime_holding_disposition(alert) == GRIPPER_HELD


def _has_safe_return_poses(context: Optional[RecoveryRuntimeContext]) -> bool:
    approach, release, lift = _context_safe_return_poses(context)
    for pose in (approach, release, lift):
        if pose is None:
            return False
        try:
            if len(list(pose)) < 7:
                return False
        except Exception:
            return False
    return True


def _should_return_held_object_before_resume(
    alert: DeviationAlert,
    context: Optional[RecoveryRuntimeContext],
    decision: RecoveryDecision,
    direct_gripper_evidence: Optional[dict] = None,
) -> bool:
    """
    是否允许执行 return_held_object_to_source。

    安全原则：
    1. 不能仅凭 action_type 判断手中有物体；
    2. 只有 runtime_summary 明确显示 gripper holding / grasped_object_ids 非空才允许；
    3. 必须存在 catch.py 在抓取时记录的 7D 安全返回位姿；
    4. pick_failed / object_dropped 不执行放回原位；
    5. 仅对历史故障（failed_step < current_step）执行归还：
       如果是当前步骤故障，手中物体可能就是需要修复的物体，
       归还后再修复会找不到候选。
    """
    if context is None:
        decision.held_object_disposition = GRIPPER_UNKNOWN
        decision.safety_block_reason = "missing_runtime_context_for_held_object_check"
        return False

    decision_kind = str(decision.kind)

    if decision_kind in {"pick_failed", "object_dropped"}:
        return False

    # 仅历史故障（前序步骤）才归还手中物体
    current_step = _safe_int(getattr(context, "step", None), None)
    failed_steps = list(decision.failed_steps or [])
    if current_step is not None and failed_steps:
        min_failed = min(failed_steps)
        if min_failed >= int(current_step):
            rospy.logwarn(
                "[RECOVERY] skip return_held_object: failed_step=%d >= current_step=%d; "
                "held object may be the repair target",
                min_failed, int(current_step),
            )
            return False

    disposition = _runtime_holding_disposition(alert, direct_gripper_evidence)
    decision.held_object_disposition = disposition
    if disposition == GRIPPER_UNKNOWN:
        decision.safety_block_reason = "gripper_holding_evidence_unknown"
        rospy.logerr(
            "[RECOVERY] cannot safely continue recovery: gripper holding evidence is unknown"
        )
        return False

    if disposition == GRIPPER_NOT_HELD:
        return False

    if not _has_safe_return_poses(context):
        decision.safety_block_reason = "held_object_safe_return_poses_missing"
        rospy.logerr(
            "[RECOVERY] cannot safely continue recovery: holding confirmed but safe return poses are missing"
        )
        return False

    return True


def classify_recoverable_alert(
    alert: DeviationAlert,
    context: Optional[RecoveryRuntimeContext],
    plan_file: str,
    *,
    confirm_count_pick_failed: int = 1,
    confirm_count_object_dropped: int = 1,
    confirm_count_current_place_failed: int = 1,
    confirm_count_history_relation: int = 2,
    confirm_count_missing_node: int = 2,
    confirm_count_forbidden_relation: int = 4,
    confirm_count_foreign_object: int = 2,
    direct_gripper_evidence: Optional[dict] = None,
) -> RecoveryDecision:
    if context is None:
        return RecoveryDecision(False, "runtime_context is None")
    severity = str(getattr(alert, "severity", "") or "").strip()
    if severity not in {"error", "critical"}:
        return RecoveryDecision(False, f"ignore severity={severity}", severity=severity)

    action_type = str(getattr(context, "action_type", "") or "").strip()
    current_step = _safe_int(getattr(context, "step", None), None)
    if current_step is None:
        current_step = _safe_int(getattr(alert, "execution_step", None), None)
    if current_step is None:
        return RecoveryDecision(False, "cannot determine current step", severity=severity)

    issue_types = _issue_types(alert)

    if "foreign_object_in_build_region" in issue_types:
        return RecoveryDecision(
            recoverable=True,
            reason="non-cube object detected in sorting region",
            kind="foreign_object_in_build_region",
            resume_step=int(current_step),
            failed_steps=[],
            confirm_count=int(confirm_count_foreign_object),
            severity=severity,
        )

    structural_issue_types = {
        "missing_relation",
        "forbidden_relation",
        "relation_endpoint_missing",
        "relation_endpoint_class_missing",
        "missing_node",
        "class_mismatch",
    }

    # 结构性故障优先于夹爪故障：结构损坏可能连锁导致夹爪状态异常，
    # 若先命中 pick_failed 会误开夹爪丢弃手中积木。
    if issue_types & structural_issue_types:
        resume_step, failed_steps, issue_step_debug = infer_resume_step_from_alert(alert, context, plan_file)
        if not failed_steps:
            return RecoveryDecision(
                False,
                "no issue mapped to action.py structural step",
                resume_step=resume_step,
                issue_step_debug=issue_step_debug,
                severity=severity,
            )

        first_failed_step = min(failed_steps)

        if first_failed_step < int(current_step):
            if "missing_node" in issue_types and not (issue_types & {"missing_relation", "forbidden_relation"}):
                kind = "history_node_missing"
            else:
                kind = "history_structure_failed"

            confirm_count_candidates = []
            if "missing_node" in issue_types:
                confirm_count_candidates.append(int(confirm_count_missing_node))
            if "forbidden_relation" in issue_types:
                confirm_count_candidates.append(int(confirm_count_forbidden_relation))
            if issue_types & {
                "missing_relation",
                "relation_endpoint_missing",
                "relation_endpoint_class_missing",
                "class_mismatch",
            }:
                confirm_count_candidates.append(int(confirm_count_history_relation))

            confirm_count = min(confirm_count_candidates)
            decision = RecoveryDecision(
                recoverable=True,
                reason="previous completed structure failed",
                kind=kind,
                resume_step=int(first_failed_step),
                failed_steps=failed_steps,
                issue_step_debug=issue_step_debug,
                confirm_count=confirm_count,
                severity=severity,
            )
            decision.need_return_held_object = _should_return_held_object_before_resume(
                alert,
                context,
                decision,
                direct_gripper_evidence=direct_gripper_evidence,
            )
            return decision

        if first_failed_step == int(current_step):
            if action_type in {"open_gripper", "leave_target"}:
                decision = RecoveryDecision(
                    recoverable=True,
                    reason="current step placement failed after release/final phase",
                    kind="current_place_failed",
                    resume_step=int(current_step),
                    failed_steps=failed_steps,
                    issue_step_debug=issue_step_debug,
                    confirm_count=int(confirm_count_current_place_failed),
                    severity=severity,
                )
                decision.need_return_held_object = _should_return_held_object_before_resume(
                    alert,
                    context,
                    decision,
                    direct_gripper_evidence=direct_gripper_evidence,
                )
                return decision
            return RecoveryDecision(
                False,
                f"ignore current-step issue before placement completion: action_type={action_type}",
                resume_step=resume_step,
                failed_steps=failed_steps,
                issue_step_debug=issue_step_debug,
                severity=severity,
            )

        return RecoveryDecision(
            False,
            "failed step is after current step",
            resume_step=resume_step,
            failed_steps=failed_steps,
            issue_step_debug=issue_step_debug,
            severity=severity,
        )

    # 纯夹爪故障（无结构问题）：低延迟快速通道。
    if _has_gripper_holding_fault(alert):
        if action_type in {"close_gripper", "lift_object"}:
            return RecoveryDecision(
                recoverable=True,
                reason="gripper holding/closed missing during grasp phase",
                kind="pick_failed",
                resume_step=int(current_step),
                failed_steps=[int(current_step)],
                confirm_count=int(confirm_count_pick_failed),
                severity=severity,
            )
        if action_type in {"move_to_target_above", "descend_to_place"}:
            return RecoveryDecision(
                recoverable=True,
                reason="gripper holding missing during transport/place phase",
                kind="object_dropped",
                resume_step=int(current_step),
                failed_steps=[int(current_step)],
                confirm_count=int(confirm_count_object_dropped),
                severity=severity,
            )
        return RecoveryDecision(False, f"ignore gripper issue at action_type={action_type}", severity=severity)

    return RecoveryDecision(False, f"ignore non-structural issues={sorted(issue_types)}", severity=severity)


def build_recovery_plan_for_alert(report, context: Optional[RecoveryRuntimeContext], alert: DeviationAlert, plan_file: str, decision: Optional[RecoveryDecision] = None) -> RecoveryPlan:
    if report.fault_type == "stage_sync_error":
        return RecoveryPlan(
            diagnosis_id=report.diagnosis_id,
            alert_id=report.alert_id,
            stage_name=report.stage_name,
            execution_step=report.execution_step,
            strategy_name="abort_due_to_stage_mismatch",
            actions=[{"action_type": "abort_execution", "params": {}}],
            max_retry=1,
            abort_if_failed=True,
        )

    if decision is None:
        decision = classify_recoverable_alert(alert, context, plan_file)

    if decision.safety_block_reason:
        return _build_gripper_evidence_block_plan(
            report, alert, context, decision, plan_file
        )

    routing = {
        "pick_failed":             _build_current_action_recovery_plan,
        "object_dropped":          _build_current_action_recovery_plan,
        "current_place_failed":    _build_current_step_repair_plan,
        "current_step_repair":     _build_current_step_repair_plan,
        "history_structure_failed": _build_history_repair_recovery_plan,
        "history_node_missing":    _build_history_repair_recovery_plan,
    }
    builder = routing.get(decision.kind, _build_ignored_plan)
    return builder(report, alert, context, decision, plan_file)


def _build_current_action_recovery_plan(report, alert, context, decision, plan_file=None):
    """抓空/掉落 → 安全撤退 → 张开夹爪 → 等待稳定 → 从当前步骤重试。"""
    current_step = _resolve_current_step(context, alert)
    return RecoveryPlan(
        diagnosis_id=report.diagnosis_id,
        alert_id=report.alert_id,
        stage_name=report.stage_name,
        execution_step=current_step,
        strategy_name="current_action_retry",
        actions=[
            _SAFE_RETREAT_ACTION,
            {"action_type": "open_gripper", "params": {}},
            {"action_type": "wait_stable_scene", "params": {"seconds": 1.0}},
            {"action_type": "resume_from_step", "params": {"resume_step": current_step}},
        ],
        max_retry=1,
    )


def _build_current_step_repair_plan(report, alert, context, decision, plan_file=None):
    """当前步骤修复（current_place_failed 直接进入，current_action 升级后进入）。"""
    current_step = _resolve_current_step(context, alert)
    return RecoveryPlan(
        diagnosis_id=report.diagnosis_id,
        alert_id=report.alert_id,
        stage_name=report.stage_name,
        execution_step=current_step,
        strategy_name="current_step_repair",
        actions=[
            _SAFE_RETREAT_ACTION,
            {"action_type": "open_gripper", "params": {}},
            {"action_type": "wait_stable_scene", "params": {"seconds": 1.0}},
            {"action_type": "repair_build_region", "params": {
                "resume_step": current_step,
                "failed_steps": [current_step],
                "max_repair_count": 1,
            }},
            {"action_type": "request_breakpoint_search", "params": {}},
            {"action_type": "resume_from_dynamic_step", "params": {}},
        ],
        max_retry=1,
        abort_if_failed=True,
    )


def _build_ignored_plan(report, alert, context, decision, plan_file=None):
    current_step = _resolve_current_step(context, alert)
    return RecoveryPlan(
        diagnosis_id=report.diagnosis_id,
        alert_id=report.alert_id,
        stage_name=report.stage_name,
        execution_step=current_step,
        strategy_name="ignored",
        actions=[],
        max_retry=0,
        abort_if_failed=False,
    )


def _build_gripper_evidence_block_plan(report, alert, context, decision, plan_file=None):
    """Retreat without release, then abort instead of repairing with ambiguous load."""
    current_step = _resolve_current_step(context, alert)
    return RecoveryPlan(
        diagnosis_id=report.diagnosis_id,
        alert_id=report.alert_id,
        stage_name=report.stage_name,
        execution_step=current_step,
        strategy_name="safe_block__%s" % decision.safety_block_reason,
        actions=[
            {
                "action_type": "safe_retreat",
                "params": {
                    "lift_z": 0.06,
                    "lift_speed": 0.08,
                    "move_ready": False,
                    "release_object": False,
                    "lift_wait": 1.0,
                    "wait_after": 0.5,
                },
            },
            {"action_type": "abort_execution", "params": {}},
        ],
        max_retry=1,
        abort_if_failed=True,
    )


def _build_history_repair_recovery_plan(report, alert, context, decision, plan_file):
    """历史结构故障修复（原有逻辑提取）。"""

    if not decision.recoverable:
        return _build_ignored_plan(report, alert, context, decision, plan_file)

    if not decision.failed_steps:
        rospy.logwarn("[RECOVERY] repair ignored: failed_steps empty")
        return _build_ignored_plan(report, alert, context, decision, plan_file)

    if context is None or context.step is None:
        rospy.logwarn("[RECOVERY] repair ignored: context.step missing")
        return _build_ignored_plan(report, alert, context, decision, plan_file)

    if min(decision.failed_steps) >= int(context.step):
        rospy.logwarn(
            "[RECOVERY] repair ignored: failed step is not history. "
            "failed_steps=%s current_step=%s",
            str(decision.failed_steps),
            str(context.step),
        )
        return _build_ignored_plan(report, alert, context, decision, plan_file)

    resume_step = int(decision.resume_step or (context.step if context is not None else 1))
    source_pose = _context_source_pose(context)
    actions = []

    # 1. 安全撤离：只做竖直上抬，不在 safe_retreat 里回 ready。
    #    回到任务起始位姿作为恢复流程后面的显式动作执行。
    actions.append({
        "action_type": "safe_retreat",
        "params": {
            "lift_z": 0.06,
            "lift_speed": 0.08,
            "move_ready": False,
            "release_object": False,
            "lift_wait": 1.0,
            "wait_after": 0.5,
        },
    })

    # 2. 如果确认夹爪仍 holding 且有安全返回位姿，则先把手中物体放回原位。
    if decision.need_return_held_object:
        actions.extend([
            {
                "action_type": "return_held_object_to_source",
                "params": {
                    "source_pose": source_pose,
                    "source_return_approach_pose": getattr(context, "source_return_approach_pose", None),
                    "source_return_release_pose": getattr(context, "source_return_release_pose", None),
                    "source_return_lift_pose": getattr(context, "source_return_lift_pose", None),
                    "move_speed": 0.15,
                    "wait_after_release": 1.0,
                },
            },
            {"action_type": "wait_stable_scene", "params": {"seconds": 0.5}},
        ])

    if decision.need_return_held_object:
        strategy_prefix = "return_held_object"
    else:
        strategy_prefix = "safe_retreat"

    strategy_name = f"{strategy_prefix}_repair_then_resume_from_dynamic_step__{decision.kind}"

    actions.extend([
        {
            "action_type": "move_to_cleanup_observe_pose",
            "params": {
                "config_file": DEFAULT_CLEANUP_CONFIG_FILE,
            },
        },
        {
            "action_type": "repair_build_region",
            "params": {
                "plan_file": plan_file,
                "config_file": DEFAULT_CLEANUP_CONFIG_FILE,
                "resume_step": int(resume_step),
                "failed_steps": list(decision.failed_steps or []),
                "max_repair_count": max(1, len(list(decision.failed_steps or []))),
                "comment": "repair: place failed object to target_place",
            },
        },
        {
            "action_type": "wait_stable_scene",
            "params": {"seconds": 1.0},
        },
        {
            "action_type": "resolve_history_resume_step",
            "params": {
                "decision_kind": decision.kind,
                "failed_steps": list(decision.failed_steps or []),
            },
        },
        {
            "action_type": "move_to_task_start_pose",
            "params": {},
        },
    ])

    # 回到起始位姿后、重新执行前，确保夹爪打开。
    if decision.kind in {"pick_failed", "object_dropped"}:
        actions.extend([
            {
                "action_type": "open_gripper",
                "params": {"reason": "ensure_gripper_open_before_retry"},
            },
            {"action_type": "wait_stable_scene", "params": {"seconds": 1.0}},
        ])

    actions.append({
        "action_type": "resume_from_dynamic_step",
        "params": {
            "resume_step": int(resume_step),
            "failed_steps": list(decision.failed_steps or []),
            "decision_kind": decision.kind,
            "decision_reason": decision.reason,
            "returned_held_object_to_source": bool(decision.need_return_held_object),
            "source_pose": source_pose,
            "issue_step_debug": decision.issue_step_debug,
        },
    })

    return RecoveryPlan(
        diagnosis_id=report.diagnosis_id,
        alert_id=report.alert_id,
        stage_name=report.stage_name,
        execution_step=report.execution_step,
        strategy_name=strategy_name,
        actions=actions,
        max_retry=1,
        abort_if_failed=True,
    )


class RecoveryManagerNode:
    def __init__(self):
        self.adapter = RobotRecoveryAdapter()
        self.diag_pub = rospy.Publisher(TOPIC_DIAGNOSIS_REPORT, String, queue_size=20)
        self.plan_pub = rospy.Publisher(TOPIC_RECOVERY_PLAN, String, queue_size=20)
        self.feedback_pub = rospy.Publisher(TOPIC_RECOVERY_FEEDBACK, String, queue_size=20)

        self.plan_file = str(rospy.get_param("~plan_file", DEFAULT_PLAN_FILE)).strip()
        self.task_profile = str(rospy.get_param("~task_profile", DEFAULT_TASK_PROFILE)).strip()
        self.task_profile_root = str(rospy.get_param("~task_profile_root", "")).strip() or None
        self.enable_configured_diagnosis = bool(rospy.get_param("~enable_configured_diagnosis", True))
        self.config_diagnoser = None
        if self.enable_configured_diagnosis:
            try:
                self.config_diagnoser = load_config_driven_diagnoser(
                    profile_name=self.task_profile or DEFAULT_TASK_PROFILE,
                    profile_root=self.task_profile_root,
                )
                rospy.logwarn(
                    "[RECOVERY] configured diagnosis enabled: profile=%s rules=%d",
                    self.config_diagnoser.profile.profile_id,
                    len(self.config_diagnoser.rules),
                )
            except Exception as exc:
                rospy.logwarn("[RECOVERY] configured diagnosis failed (%s); using legacy diagnosis", str(exc))
                self.enable_configured_diagnosis = False

        self.enable_configured_recovery = bool(rospy.get_param("~enable_configured_recovery", True))
        self.config_recovery_planner = None
        if self.enable_configured_recovery:
            try:
                self.config_recovery_planner = load_config_driven_recovery_planner(
                    profile_name=self.task_profile or DEFAULT_TASK_PROFILE,
                    profile_root=self.task_profile_root,
                )
                rospy.logwarn(
                    "[RECOVERY] configured recovery enabled: profile=%s strategies=%d routes=%d",
                    self.config_recovery_planner.profile.profile_id,
                    len(self.config_recovery_planner.strategies),
                    len(self.config_recovery_planner.rules),
                )
            except Exception as exc:
                rospy.logwarn("[RECOVERY] configured recovery failed (%s); using legacy recovery planning", str(exc))
                self.enable_configured_recovery = False

        self.recovery_strategy = str(rospy.get_param("~recovery_strategy", "repair")).strip()
        if self.recovery_strategy != "repair":
            rospy.logwarn(
                "[RECOVERY] unsupported recovery_strategy='%s'; forcing repair",
                self.recovery_strategy,
            )
            self.recovery_strategy = "repair"
        self.fault_confirm_count = int(rospy.get_param("~fault_confirm_count", 2))
        self.fault_sample_timeout = float(rospy.get_param("~fault_sample_timeout", 5.0))
        self.recovery_cooldown_sec = float(rospy.get_param("~recovery_cooldown_sec", 2.0))
        self.history_direct_resume_enabled = bool(
            rospy.get_param("~history_direct_resume_enabled", False)
        )
        self.history_direct_resume_only_adjacent = bool(
            rospy.get_param("~history_direct_resume_only_adjacent", True)
        )
        self.history_direct_resume_fallback_to_breakpoint = bool(
            rospy.get_param("~history_direct_resume_fallback_to_breakpoint", False)
        )
        self.verify_no_candidate_with_scene_graph = bool(
            rospy.get_param("~verify_no_candidate_with_scene_graph", True)
        )
        self.allow_legacy_event_context = bool(
            rospy.get_param("~allow_legacy_event_context", True)
        )
        self.gripper_state_topic = str(
            rospy.get_param("~gripper_state_topic", DEFAULT_GRIPPER_STATE_TOPIC)
        ).strip()
        self.gripper_state_max_age_sec = max(
            0.1, float(rospy.get_param("~gripper_state_max_age_sec", 1.0))
        )
        self.gripper_min_stable_count = max(
            1, int(rospy.get_param("~gripper_min_stable_count", 3))
        )

        self.confirm_count_pick_failed = int(rospy.get_param("~confirm_count_pick_failed", 2))
        self.confirm_count_object_dropped = int(rospy.get_param("~confirm_count_object_dropped", 2))
        self.confirm_count_current_place_failed = int(rospy.get_param("~confirm_count_current_place_failed", 2))
        self.confirm_count_history_relation = int(rospy.get_param("~confirm_count_history_relation", 2))
        self.confirm_count_missing_node = int(rospy.get_param("~confirm_count_missing_node", 4))
        self.confirm_count_forbidden_relation = int(rospy.get_param("~confirm_count_forbidden_relation", 999))
        self.confirm_count_foreign_object = int(rospy.get_param("~confirm_count_foreign_object", 2))

        self._lock = threading.Lock()
        self._recovering = False
        self._runtime_context = None
        self._pending_fault_key = None
        self._pending_fault_count = 0
        self._pending_fault_last_time = 0.0
        self._last_recovery_key = None
        self._last_recovery_time = 0.0
        self._dynamic_resume_step: Optional[int] = None
        self._reuse_no_candidate_breakpoint = False
        self._latest_gripper_evidence: Optional[dict] = None

        # 当前动作故障重试计数与升级
        self._retry_count_by_step: Dict[Tuple[int, str], int] = {}
        self._escalated_keys: set = set()
        self.retry_threshold_pick_failed = int(rospy.get_param("~retry_threshold_pick_failed", 3))
        self.retry_threshold_object_dropped = int(rospy.get_param("~retry_threshold_object_dropped", 2))

        rospy.Subscriber(TOPIC_DEVIATION_ALERT, String, self.alert_callback, queue_size=50)
        rospy.Subscriber(TOPIC_RECOVERY_RUNTIME_CONTEXT, String, self.context_callback, queue_size=20)
        rospy.Subscriber(
            self.gripper_state_topic,
            GripperState,
            self.gripper_state_callback,
            queue_size=100,
        )

        rospy.logwarn("[RECOVERY] RecoveryManager started")
        rospy.logwarn("[RECOVERY] plan_file=%s", self.plan_file)
        rospy.logwarn(
            "[RECOVERY] confirm counts: pick=%d dropped=%d place=%d history_relation=%d missing_node=%d forbidden_relation=%d foreign_object=%d fallback=%d",
            self.confirm_count_pick_failed,
            self.confirm_count_object_dropped,
            self.confirm_count_current_place_failed,
            self.confirm_count_history_relation,
            self.confirm_count_missing_node,
            self.confirm_count_forbidden_relation,
            self.confirm_count_foreign_object,
            self.fault_confirm_count,
        )
        rospy.logwarn(
            "[RECOVERY] retry thresholds: pick_failed=%d object_dropped=%d",
            self.retry_threshold_pick_failed,
            self.retry_threshold_object_dropped,
        )
        rospy.logwarn(
            "[RECOVERY] history direct resume: enabled=%s only_adjacent=%s fallback_to_breakpoint=%s",
            str(self.history_direct_resume_enabled),
            str(self.history_direct_resume_only_adjacent),
            str(self.history_direct_resume_fallback_to_breakpoint),
        )
        rospy.logwarn(
            "[RECOVERY] no-candidate fresh-scene verification: enabled=%s",
            str(self.verify_no_candidate_with_scene_graph),
        )

    def gripper_state_callback(self, msg: GripperState):
        try:
            self._latest_gripper_evidence = {
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
                "min_stable_count": int(self.gripper_min_stable_count),
            }
        except Exception as exc:
            rospy.logwarn_throttle(
                2.0, "[RECOVERY] invalid gripper state sample: %s", str(exc)
            )

    def _current_direct_gripper_evidence(self, context) -> dict:
        evidence = dict(self._latest_gripper_evidence or {})
        if not evidence:
            return {}
        now_sec = float(rospy.Time.now().to_sec())
        age_sec = now_sec - float(evidence.get("sample_end_stamp", 0.0) or 0.0)
        evidence["age_sec"] = age_sec
        if age_sec < -0.2 or age_sec > self.gripper_state_max_age_sec:
            evidence["valid"] = False
            evidence["reason"] = "stale_gripper_state"
            return evidence
        if context is not None:
            context_run = str(getattr(context, "run_id", "") or "")
            context_attempt = _safe_int(getattr(context, "attempt_id", None), None)
            context_command_id = str(
                getattr(context, "gripper_command_id", "") or ""
            )
            if context_run and evidence.get("run_id") != context_run:
                evidence["valid"] = False
                evidence["reason"] = "gripper_run_id_mismatch"
            elif (
                context_attempt is not None
                and _safe_int(evidence.get("attempt_id"), None) != context_attempt
            ):
                evidence["valid"] = False
                evidence["reason"] = "gripper_attempt_id_mismatch"
            elif (
                context_command_id
                and str(evidence.get("command_id", "") or "") != context_command_id
            ):
                evidence["valid"] = False
                evidence["reason"] = "gripper_command_id_mismatch"
        return evidence

    def context_callback(self, msg: String):
        raw = str(msg.data).strip()
        if not raw:
            return
        try:
            new_context = RecoveryRuntimeContext.from_json(raw)
        except Exception as exc:
            rospy.logwarn("RecoveryManager failed to parse runtime context: %s", str(exc))
            return
        old_step = getattr(self._runtime_context, "step", None) if self._runtime_context is not None else None
        old_run_id = getattr(self._runtime_context, "run_id", "") if self._runtime_context is not None else ""
        old_attempt_id = getattr(self._runtime_context, "attempt_id", None) if self._runtime_context is not None else None
        self._runtime_context = new_context
        if (
            old_step != new_context.step
            or old_run_id != new_context.run_id
            or old_attempt_id != new_context.attempt_id
        ):
            self._reset_pending_fault_counter()
            self._on_step_changed(new_context.step)

    def publish_feedback(self, feedback: RecoveryFeedback):
        context = self._runtime_context
        if context is not None and not feedback.run_id:
            feedback.run_id = context.run_id
            feedback.attempt_id = context.attempt_id
            feedback.stage_seq = context.stage_seq
        self.feedback_pub.publish(String(data=feedback.to_json()))

    def _reset_pending_fault_counter(self):
        self._pending_fault_key = None
        self._pending_fault_count = 0
        self._pending_fault_last_time = 0.0

    def _get_threshold(self, fault_kind: str) -> int:
        return {
            "pick_failed": self.retry_threshold_pick_failed,
            "object_dropped": self.retry_threshold_object_dropped,
        }.get(fault_kind, 999)

    def _on_step_changed(self, new_step):
        self._retry_count_by_step.clear()
        self._escalated_keys.clear()

    def _maybe_escalate(self, decision: RecoveryDecision, context, alert) -> RecoveryDecision:
        """pick_failed / object_dropped 重试耗尽后升级为 current_step_repair。"""
        if decision.kind not in {"pick_failed", "object_dropped"}:
            return decision

        current_step = _resolve_current_step(context, alert)
        key = (current_step, decision.kind)

        if key in self._escalated_keys:
            return RecoveryDecision(
                recoverable=True,
                reason=decision.reason,
                kind="current_step_repair",
                resume_step=current_step,
                failed_steps=[current_step],
                confirm_count=decision.confirm_count,
                severity=decision.severity,
            )

        count = self._retry_count_by_step.get(key, 0) + 1
        self._retry_count_by_step[key] = count

        threshold = self._get_threshold(decision.kind)
        if count >= threshold:
            self._escalated_keys.add(key)
            rospy.logwarn(
                "[RECOVERY] escalating (%d, %s) count=%d >= threshold=%d -> current_step_repair",
                current_step, decision.kind, count, threshold,
            )
            return RecoveryDecision(
                recoverable=True,
                reason=decision.reason,
                kind="current_step_repair",
                resume_step=current_step,
                failed_steps=[current_step],
                confirm_count=decision.confirm_count,
                severity=decision.severity,
            )

        return decision

    def _retry_count_for_report(self, report, context, alert) -> int:
        fault_type = str(getattr(report, "fault_type", "") or "")
        if fault_type not in {"pick_failed", "object_dropped"}:
            return 0
        current_step = _resolve_current_step(context, alert)
        return int(self._retry_count_by_step.get((current_step, fault_type), 0))

    def _build_recovery_plan(self, report, context, alert, decision: RecoveryDecision) -> RecoveryPlan:
        if decision.safety_block_reason:
            return _build_gripper_evidence_block_plan(
                report, alert, context, decision, self.plan_file
            )
        if self.enable_configured_recovery and self.config_recovery_planner is not None:
            retry_count = self._retry_count_for_report(report, context, alert)
            try:
                plan = self.config_recovery_planner.build_plan(
                    report=report,
                    alert=alert,
                    context=context,
                    decision=decision,
                    plan_file=self.plan_file,
                    retry_count=retry_count,
                )
                if plan is not None:
                    rospy.logwarn(
                        "[RECOVERY] configured recovery selected: strategy=%s retry_count=%d",
                        plan.strategy_name,
                        retry_count,
                    )
                    return self._apply_history_direct_resume_policy_to_plan(plan, decision, context)
                rospy.logwarn("[RECOVERY] configured recovery had no safe route; fallback to legacy recovery plan")
            except Exception as exc:
                rospy.logwarn("[RECOVERY] configured recovery failed (%s); fallback to legacy recovery plan", str(exc))

        plan = build_recovery_plan_for_alert(
            report=report,
            context=context,
            alert=alert,
            plan_file=self.plan_file,
            decision=decision,
        )
        return self._apply_history_direct_resume_policy_to_plan(plan, decision, context)

    def _apply_history_direct_resume_policy_to_plan(
        self,
        plan: RecoveryPlan,
        decision: RecoveryDecision,
        context: Optional[RecoveryRuntimeContext],
    ) -> RecoveryPlan:
        if str(getattr(decision, "kind", "")) not in {"history_structure_failed", "history_node_missing"}:
            return plan

        requested_failed_steps = [int(v) for v in list(decision.failed_steps or [])]
        if not requested_failed_steps:
            return plan

        desired_max_repair_count = max(1, len(requested_failed_steps))
        actions = []
        inserted_resolver = False
        has_repair_action = False

        for action in list(plan.actions or []):
            action = dict(action or {})
            action_type = str(action.get("action_type", "") or "").strip()
            params = dict(action.get("params", {}) or {})

            # Legacy/fallback history plans may already contain the direct
            # resolver.  When direct resume is disabled, convert that action
            # back to the fresh-scene transaction explicitly.  Merely leaving
            # the resolver in place makes _resolve_history_resume_step()
            # reject the action and abort an otherwise successful repair.
            if (
                not self.history_direct_resume_enabled
                and action_type == "resolve_history_resume_step"
            ):
                action = {
                    "action_type": "request_breakpoint_search",
                    "params": {},
                }
                action_type = "request_breakpoint_search"
                params = {}

            if action_type == "repair_build_region":
                has_repair_action = True
                params["failed_steps"] = list(requested_failed_steps)
                # Marks the narrowly-scoped no-candidate scene verification
                # path.  Current-step repairs must keep their existing
                # fail-closed behavior.
                params["history_repair_kind"] = str(decision.kind)
                try:
                    current_max_repair_count = int(params.get("max_repair_count", 0) or 0)
                except Exception:
                    current_max_repair_count = 0
                params["max_repair_count"] = max(
                    desired_max_repair_count,
                    current_max_repair_count,
                )
                action["params"] = params

            if self.history_direct_resume_enabled and action_type == "request_breakpoint_search":
                action = {
                    "action_type": "resolve_history_resume_step",
                    "params": {
                        "decision_kind": decision.kind,
                        "failed_steps": list(requested_failed_steps),
                        "fallback_to_breakpoint": self.history_direct_resume_fallback_to_breakpoint,
                        "only_adjacent": self.history_direct_resume_only_adjacent,
                    },
                }
                inserted_resolver = True
            elif action_type == "resolve_history_resume_step":
                params.setdefault("decision_kind", decision.kind)
                params.setdefault("failed_steps", list(requested_failed_steps))
                params.setdefault("fallback_to_breakpoint", self.history_direct_resume_fallback_to_breakpoint)
                params.setdefault("only_adjacent", self.history_direct_resume_only_adjacent)
                action["params"] = params
                inserted_resolver = True

            if (
                self.history_direct_resume_enabled
                and action_type == "resume_from_dynamic_step"
                and has_repair_action
                and not inserted_resolver
            ):
                actions.append({
                    "action_type": "resolve_history_resume_step",
                    "params": {
                        "decision_kind": decision.kind,
                        "failed_steps": list(requested_failed_steps),
                        "fallback_to_breakpoint": self.history_direct_resume_fallback_to_breakpoint,
                        "only_adjacent": self.history_direct_resume_only_adjacent,
                    },
                })
                inserted_resolver = True

            actions.append(action)

        plan.actions = actions
        return plan

    def _fault_counter_key(self, alert: DeviationAlert, context: Optional[RecoveryRuntimeContext], decision: RecoveryDecision):
        issue_types = tuple(sorted(_issue_types(alert)))

        # 历史结构故障：只按语义计数，不按当前 action_type/stage 计数。
        # 否则执行阶段变化会导致一直 count=1/2，无法达成 confirm_count。
        if str(decision.kind) in {"history_structure_failed", "history_node_missing"}:
            return (
                "history_fault",
                str(decision.kind),
                tuple(int(v) for v in list(decision.failed_steps or [])),
                issue_types,
            )

        context_step = None
        context_target = ""
        action_type = ""

        if context is not None:
            context_step = context.step
            context_target = str(
                getattr(context, "target_object_id", "")
                or getattr(context, "target_class", "")
                or ""
            ).strip()
            action_type = str(getattr(context, "action_type", "") or "").strip()

        return (
            int(context_step) if context_step is not None else alert.execution_step,
            context_target,
            action_type,
            str(getattr(alert, "stage_name", "")).strip(),
            str(decision.kind),
            tuple(decision.failed_steps or []),
            issue_types,
        )

    def _update_fault_counter(self, alert: DeviationAlert, context: Optional[RecoveryRuntimeContext], decision: RecoveryDecision):
        key = self._fault_counter_key(alert, context, decision)
        now = time.monotonic()
        if key == self._pending_fault_key and (now - self._pending_fault_last_time) <= self.fault_sample_timeout:
            self._pending_fault_count += 1
        else:
            self._pending_fault_key = key
            self._pending_fault_count = 1
        self._pending_fault_last_time = now
        return key, self._pending_fault_count

    def _exec_one_action(self, action: dict, context: Optional[RecoveryRuntimeContext]) -> bool:
        action_type = str(action.get("action_type", "")).strip()
        params = dict(action.get("params", {}) or {})
        if action_type == "pause_execution":
            return self.adapter.pause_execution()
        if action_type == "resume_execution":
            return self.adapter.resume_execution()
        if action_type == "abort_execution":
            return self.adapter.abort_execution()
        if action_type == "move_ready":
            return self.adapter.move_ready()
        if action_type == "open_gripper":
            return self.adapter.open_gripper()
        if action_type == "close_gripper":
            return self.adapter.close_gripper()
        if action_type == "wait_stable_scene":
            return self.adapter.wait_stable_scene(float(params.get("seconds", 1.0)))
        if action_type == "move_to_cleanup_observe_pose":
            return self.adapter.move_to_cleanup_observe_pose(
                config_file=params.get("config_file", DEFAULT_CLEANUP_CONFIG_FILE),
            )
        if action_type == "cleanup_sorting_region":
            return self.adapter.cleanup_sorting_region(
                config_file=params.get("config_file", DEFAULT_CLEANUP_CONFIG_FILE),
                allowed_class=str(params.get("allowed_class", "cube")),
                target_place=params.get("target_place", []),
                max_cleanup_count=int(params.get("max_cleanup_count", 1)),
            )
        if action_type == "cleanup_build_region":
            rospy.logerr(
                "[RECOVERY] cleanup_build_region is disabled. "
                "Current recovery strategy only supports repair_build_region."
            )
            return False
        if action_type == "move_to_task_start_pose":
            return self.adapter.move_to_task_start_pose()
        if action_type == "safe_retreat":
            return self.adapter.safe_retreat(
                lift_z=float(params.get("lift_z", 0.06)),
                lift_speed=float(params.get("lift_speed", 0.08)),
                move_ready=bool(params.get("move_ready", True)),
                release_object=bool(params.get("release_object", False)),
                lift_wait=float(params.get("lift_wait", 1.0)),
                wait_after=float(params.get("wait_after", 1.0)),
            )
        if action_type == "return_held_object_to_source":
            return self.adapter.return_held_object_to_source(
                source_pose=params.get("source_pose"),
                source_return_approach_pose=params.get("source_return_approach_pose"),
                source_return_release_pose=params.get("source_return_release_pose"),
                source_return_lift_pose=params.get("source_return_lift_pose"),
                move_speed=float(params.get("move_speed", 0.15)),
                wait_after_release=float(params.get("wait_after_release", 1.0)),
            )
        if action_type == "repair_build_region":
            max_repair_count = params.get("max_repair_count", None)
            if max_repair_count is not None:
                max_repair_count = int(max_repair_count)
            requested_failed_steps = list(params.get("failed_steps", []) or [])
            repair_ok = self.adapter.repair_build_region(
                plan_file=params.get("plan_file", self.plan_file),
                config_file=params.get("config_file", DEFAULT_CLEANUP_CONFIG_FILE),
                resume_step=int(params.get("resume_step", context.step if context is not None else 1)),
                failed_steps=requested_failed_steps,
                max_repair_count=max_repair_count,
            )
            if repair_ok:
                self._reuse_no_candidate_breakpoint = False
                return True

            history_repair_kind = str(
                params.get("history_repair_kind", "") or ""
            ).strip()
            if (
                not self.verify_no_candidate_with_scene_graph
                or history_repair_kind not in HISTORY_REPAIR_KINDS
            ):
                return False

            return self._verify_no_candidate_repair_with_fresh_scene(
                requested_failed_steps=requested_failed_steps,
                config_file=params.get("config_file", DEFAULT_CLEANUP_CONFIG_FILE),
                context=context,
            )
        if action_type == "resolve_history_resume_step":
            return self._resolve_history_resume_step(params, context)
        if action_type == "request_breakpoint_search":
            return self._request_breakpoint_and_store()
        if action_type == "resume_from_dynamic_step":
            if self._dynamic_resume_step is None:
                rospy.logerr("[RECOVERY] resume_from_dynamic_step: _dynamic_resume_step not set")
                return False
            return self.adapter.resume_from_dynamic_step(self._dynamic_resume_step)
        if action_type == "resume_from_step":
            resume_step = params.get("resume_step", None)
            if resume_step is None:
                rospy.logerr("[RECOVERY] resume_from_step action missing resume_step")
                return False
            return self.adapter.resume_from_step(int(resume_step))
        rospy.logwarn("RecoveryManager unknown action: %s", action_type)
        return False

    def _request_breakpoint_and_store(self) -> bool:
        if self._reuse_no_candidate_breakpoint and self._dynamic_resume_step is not None:
            rospy.logwarn(
                "[RECOVERY] reusing fresh breakpoint from no-candidate verification: "
                "resume_step=%d",
                self._dynamic_resume_step,
            )
            self._reuse_no_candidate_breakpoint = False
            return True

        self._reuse_no_candidate_breakpoint = False
        context_payload = (
            self._runtime_context.to_dict()
            if self._runtime_context is not None
            else {}
        )
        result = self.adapter.request_breakpoint_search(**context_payload)
        if not result.get("success"):
            rospy.logerr("[RECOVERY] breakpoint search failed")
            return False
        self._dynamic_resume_step = int(result["resume_step"])
        rospy.logwarn("[RECOVERY] stored dynamic_resume_step=%d", self._dynamic_resume_step)
        return True

    def _verify_no_candidate_repair_with_fresh_scene(
        self,
        *,
        requested_failed_steps,
        config_file: str,
        context: Optional[RecoveryRuntimeContext],
    ) -> bool:
        repair_result = getattr(self.adapter, "last_repair_result", None)
        eligible, eligibility_reason = is_no_candidate_scene_verification_eligible(
            repair_result,
            requested_failed_steps,
        )
        if not eligible:
            rospy.logerr(
                "[RECOVERY] failed repair is not eligible for no-candidate scene "
                "verification: reason=%s result=%s",
                eligibility_reason,
                str(repair_result),
            )
            return False

        rospy.logwarn(
            "[RECOVERY] no repair candidate for historical steps=%s; "
            "confirming observation pose before fresh-scene verification",
            str(sorted(_int_set(requested_failed_steps))),
        )
        if not self.adapter.move_to_cleanup_observe_pose(config_file=config_file):
            rospy.logerr(
                "[RECOVERY] no-candidate verification blocked: cleanup observe "
                "pose was not confirmed"
            )
            return False

        context_payload = (
            self._runtime_context.to_dict()
            if self._runtime_context is not None
            else (context.to_dict() if context is not None else {})
        )
        breakpoint_result = self.adapter.request_breakpoint_search(**context_payload)
        ok, resume_step, reason, already_satisfied_steps = (
            resolve_no_candidate_scene_verification(
                requested_failed_steps=requested_failed_steps,
                repair_result=repair_result,
                breakpoint_result=breakpoint_result,
            )
        )
        if not ok or resume_step is None:
            rospy.logerr(
                "[RECOVERY] no-candidate fresh-scene verification failed: "
                "reason=%s requested_steps=%s breakpoint_result=%s",
                reason,
                str(sorted(_int_set(requested_failed_steps))),
                str(breakpoint_result),
            )
            return False

        repaired_steps = sorted(_int_set((repair_result or {}).get("repaired_steps", [])))
        resolved_steps = sorted(
            _int_set(repaired_steps) | _int_set(already_satisfied_steps)
        )
        verified_result = dict(repair_result or {})
        verified_result.update({
            "success": True,
            "failed_count": 0,
            "failed_steps": [],
            "partial": False,
            "already_satisfied_steps": list(already_satisfied_steps),
            "resolved_steps": resolved_steps,
            "resolution": "verified_noop_after_no_candidate",
            "verification_resume_step": int(resume_step),
            "verification_reason": str(reason),
        })
        self.adapter.last_repair_result = verified_result
        self._dynamic_resume_step = int(resume_step)
        self._reuse_no_candidate_breakpoint = True
        rospy.logwarn(
            "[RECOVERY] historical no-candidate result verified by fresh scene: "
            "already_satisfied_steps=%s repaired_steps=%s resume_step=%d",
            str(already_satisfied_steps),
            str(repaired_steps),
            self._dynamic_resume_step,
        )
        return True

    def _resolve_history_resume_step(
        self,
        params: dict,
        context: Optional[RecoveryRuntimeContext],
    ) -> bool:
        if self._reuse_no_candidate_breakpoint and self._dynamic_resume_step is not None:
            rospy.logwarn(
                "[RECOVERY] using fresh no-candidate verification instead of "
                "history direct resume: resume_step=%d",
                self._dynamic_resume_step,
            )
            self._reuse_no_candidate_breakpoint = False
            return True

        fallback_to_breakpoint = bool(
            params.get(
                "fallback_to_breakpoint",
                self.history_direct_resume_fallback_to_breakpoint,
            )
        )

        if not self.history_direct_resume_enabled:
            rospy.logwarn(
                "[RECOVERY] history direct resume disabled; fallback_to_breakpoint=%s",
                str(fallback_to_breakpoint),
            )
            return self._request_breakpoint_and_store() if fallback_to_breakpoint else False

        repair_result = getattr(self.adapter, "last_repair_result", None)
        failed_steps = list(params.get("failed_steps", []) or [])
        decision_kind = str(params.get("decision_kind", "") or "")
        only_adjacent = bool(
            params.get("only_adjacent", self.history_direct_resume_only_adjacent)
        )
        current_step = params.get(
            "current_step",
            getattr(context, "step", None) if context is not None else None,
        )

        ok, resume_step, reason = resolve_history_direct_resume_step(
            decision_kind=decision_kind,
            current_step=current_step,
            requested_failed_steps=failed_steps,
            repair_result=repair_result,
            only_adjacent=only_adjacent,
        )

        if ok and resume_step is not None:
            self._dynamic_resume_step = int(resume_step)
            rospy.logwarn(
                "[RECOVERY] history direct resume: failed_steps=%s repaired_steps=%s "
                "resume_step=%d reason=%s",
                str(failed_steps),
                str((repair_result or {}).get("repaired_steps", [])),
                self._dynamic_resume_step,
                str(reason),
            )
            return True

        rospy.logwarn(
            "[RECOVERY] history direct resume unavailable: reason=%s failed_steps=%s "
            "repair_result=%s fallback_to_breakpoint=%s",
            str(reason),
            str(failed_steps),
            str(repair_result),
            str(fallback_to_breakpoint),
        )

        if fallback_to_breakpoint:
            return self._request_breakpoint_and_store()

        return False

    def _check_action_preconditions(self, action: dict, context: Optional[RecoveryRuntimeContext]) -> Tuple[bool, str]:
        action_type = str(action.get("action_type", "") or "").strip()
        preconditions = list(action.get("preconditions", []) or [])
        for precondition in preconditions:
            precondition = str(precondition or "").strip()
            if not precondition:
                continue
            if precondition == "dynamic_resume_step_available":
                if self._dynamic_resume_step is None:
                    return False, "dynamic_resume_step is not available"
                continue
            if precondition == "execution_control_available":
                if self.adapter is None:
                    return False, "execution control adapter is not available"
                continue
            if precondition == "recovery_ready" and action_type == "safe_retreat":
                # safe_retreat establishes the recovery-ready handoff itself.
                continue
            # Hardware and perception preconditions are ultimately verified by
            # the primitive action. Keep them in feedback for traceability.
        return True, ""

    def _execute_plan(self, plan: RecoveryPlan, context: Optional[RecoveryRuntimeContext]):
        self._dynamic_resume_step = None
        self._reuse_no_candidate_breakpoint = False
        if hasattr(self.adapter, "last_repair_result"):
            self.adapter.last_repair_result = None
        self.publish_feedback(RecoveryFeedback(plan_id=plan.plan_id, stage_name=plan.stage_name, execution_step=plan.execution_step, status="started", current_action="", message=f"start strategy={plan.strategy_name}", details={"step": context.step if context is not None else None, "source": "recovery_manager"}))
        for action in list(plan.actions or []):
            action_type = str(action.get("action_type", "")).strip()
            params = dict(action.get("params", {}) or {})
            self.publish_feedback(RecoveryFeedback(plan_id=plan.plan_id, stage_name=plan.stage_name, execution_step=plan.execution_step, status="running", current_action=action_type, message=f"running {action_type}", details={"source": "recovery_manager", "params": params}))
            preconditions_ok, precondition_error = self._check_action_preconditions(action, context)
            if not preconditions_ok:
                rospy.logerr("[RECOVERY] precondition failed for %s: %s", action_type, precondition_error)
                if plan.abort_if_failed:
                    try:
                        self.adapter.abort_execution()
                    except Exception:
                        pass
                self.publish_feedback(RecoveryFeedback(plan_id=plan.plan_id, stage_name=plan.stage_name, execution_step=plan.execution_step, status="failed", current_action=action_type, message=f"precondition failed: {precondition_error}", details={"source": "recovery_manager", "preconditions": list(action.get("preconditions", []) or [])}))
                return False
            ok = self._exec_one_action(action, context)
            if not ok:
                if plan.abort_if_failed:
                    try:
                        self.adapter.abort_execution()
                    except Exception:
                        pass
                self.publish_feedback(RecoveryFeedback(plan_id=plan.plan_id, stage_name=plan.stage_name, execution_step=plan.execution_step, status="failed", current_action=action_type, message=f"action failed: {action_type}", details={"source": "recovery_manager"}))
                return False
        self.publish_feedback(RecoveryFeedback(plan_id=plan.plan_id, stage_name=plan.stage_name, execution_step=plan.execution_step, status="succeeded", current_action="", message=f"strategy succeeded: {plan.strategy_name}", details={"source": "recovery_manager"}))
        return True

    def _make_decision(self, alert: DeviationAlert, context: RecoveryRuntimeContext) -> RecoveryDecision:
        return classify_recoverable_alert(
            alert=alert,
            context=context,
            plan_file=self.plan_file,
            confirm_count_pick_failed=self.confirm_count_pick_failed,
            confirm_count_object_dropped=self.confirm_count_object_dropped,
            confirm_count_current_place_failed=self.confirm_count_current_place_failed,
            confirm_count_history_relation=self.confirm_count_history_relation,
            confirm_count_missing_node=self.confirm_count_missing_node,
            confirm_count_forbidden_relation=self.confirm_count_forbidden_relation,
            confirm_count_foreign_object=self.confirm_count_foreign_object,
            direct_gripper_evidence=self._current_direct_gripper_evidence(context),
        )

    def _handle_confirmed_fault(self, alert: DeviationAlert, context: Optional[RecoveryRuntimeContext], counter_key, decision: RecoveryDecision):
        try:
            report = diagnose_alert(
                alert,
                runtime_context=context,
                config_diagnoser=self.config_diagnoser,
                profile_name=self.task_profile,
                profile_root=self.task_profile_root,
                enable_configured_diagnosis=self.enable_configured_diagnosis,
            )
            report.run_id = alert.run_id
            report.attempt_id = alert.attempt_id
            report.stage_seq = alert.stage_seq
            report.barrier_id = alert.barrier_id
            self.diag_pub.publish(String(data=report.to_json()))
            rospy.logwarn(
                "RecoveryManager diagnosis confirmed: stage=%s fault_type=%s mode=%s decision=%s reason=%s",
                report.stage_name,
                report.fault_type,
                report.details.get("diagnosis_mode", ""),
                decision.kind,
                decision.reason,
            )
            decision = self._maybe_escalate(decision, context, alert)
            plan = self._build_recovery_plan(report, context, alert, decision)
            plan.run_id = alert.run_id
            plan.attempt_id = alert.attempt_id
            plan.stage_seq = alert.stage_seq
            plan.barrier_id = alert.barrier_id
            self.plan_pub.publish(String(data=plan.to_json()))
            resume_step = None
            need_return = False
            for action in list(plan.actions or []):
                if str(action.get("action_type")) == "resume_from_step":
                    params = dict(action.get("params", {}) or {})
                    resume_step = params.get("resume_step")
                    need_return = bool(params.get("returned_held_object_to_source", False))
            rospy.logwarn("RecoveryManager plan: stage=%s strategy=%s resume_step=%s return_held=%s", plan.stage_name, plan.strategy_name, str(resume_step), str(need_return))
            ok = self._execute_plan(plan, context)
            if ok:
                self._last_recovery_key = counter_key
                self._last_recovery_time = time.monotonic()
                rospy.logwarn("RecoveryManager recovery succeeded: stage=%s resume_step=%s", plan.stage_name, str(resume_step))
            else:
                rospy.logerr("RecoveryManager recovery failed: stage=%s strategy=%s", plan.stage_name, plan.strategy_name)
        except Exception as exc:
            traceback.print_exc()
            self.publish_feedback(RecoveryFeedback(stage_name=getattr(alert, "stage_name", ""), execution_step=getattr(alert, "execution_step", None), status="failed", current_action="", message=f"recovery_manager exception: {exc}", details={"source": "recovery_manager"}))
        finally:
            with self._lock:
                self._recovering = False

    def alert_callback(self, msg: String):
        raw = str(msg.data).strip()
        if not raw:
            return
        try:
            alert = DeviationAlert.from_json(raw)
        except Exception as exc:
            rospy.logerr("RecoveryManager failed to parse alert: %s", str(exc))
            return

        context = self._runtime_context
        if context is None:
            rospy.logwarn("[RECOVERY DEBUG] ignore alert: runtime_context is None, stage=%s step=%s", str(getattr(alert, "stage_name", "")), str(getattr(alert, "execution_step", None)))
            return

        context_matches, context_reason = event_context_matches(
            context.to_dict(),
            alert.to_dict(),
            allow_legacy=self.allow_legacy_event_context,
        )
        if not context_matches:
            rospy.logwarn(
                "[RECOVERY] ignore alert with stale execution context: reason=%s alert_run=%s alert_attempt=%s current_run=%s current_attempt=%s",
                context_reason,
                str(alert.run_id),
                str(alert.attempt_id),
                str(context.run_id),
                str(context.attempt_id),
            )
            return

        context_status = str(getattr(context, "status", "")).strip()
        if context_status not in {"active", "recovery", "waiting_recovery"}:
            rospy.logwarn("[RECOVERY DEBUG] ignore alert: invalid context.status=%s", context_status)
            return

        if alert.execution_step is not None and context.step is not None:
            if int(alert.execution_step) != int(context.step):
                rospy.logwarn("[RECOVERY DEBUG] ignore alert: step mismatch alert.step=%s context.step=%s", str(alert.execution_step), str(context.step))
                return

        decision = self._make_decision(alert, context)
        if not decision.recoverable:
            rospy.logwarn(
                "[RECOVERY] ignore alert: reason=%s stage=%s context_step=%s action_type=%s severity=%s resume_step=%s failed_steps=%s issues=%s",
                decision.reason,
                str(getattr(alert, "stage_name", "")),
                str(getattr(context, "step", None)),
                str(getattr(context, "action_type", "")),
                str(getattr(alert, "severity", "")),
                str(decision.resume_step),
                str(decision.failed_steps),
                str(sorted(_issue_types(alert))),
            )
            return

        with self._lock:
            if self._recovering:
                rospy.logwarn("RecoveryManager got new fault during recovery, ignored: step=%s stage=%s", str(context.step), str(alert.stage_name))
                return

        counter_key, counter_value = self._update_fault_counter(alert, context, decision)
        if str(alert.source) == "validation_monitor/transaction":
            # The transaction validator has already required consecutive,
            # semantically stable scene versions before emitting this alert.
            required_count = 1
        else:
            required_count = max(1, int(decision.confirm_count or self.fault_confirm_count or 1))
        rospy.logwarn("RecoveryManager fault confirmation progress: step=%s stage=%s kind=%s count=%d/%d reason=%s", str(context.step), str(alert.stage_name), str(decision.kind), counter_value, required_count, str(decision.reason))

        if counter_value < required_count:
            self.publish_feedback(RecoveryFeedback(stage_name=alert.stage_name, execution_step=alert.execution_step, status="running", current_action="confirm_fault", message=f"fault sample {counter_value}/{required_count}, not confirmed yet", details={"counter_key": str(counter_key), "counter_value": counter_value, "threshold": required_count, "decision_kind": decision.kind, "decision_reason": decision.reason, "source": "recovery_manager"}))
            return

        self._reset_pending_fault_counter()
        now = time.monotonic()
        with self._lock:
            if counter_key == self._last_recovery_key and (now - self._last_recovery_time) < self.recovery_cooldown_sec:
                rospy.logwarn("RecoveryManager skip confirmed fault: in post-recovery cooldown")
                return
            self._recovering = True

        worker = threading.Thread(target=self._handle_confirmed_fault, args=(alert, context, counter_key, decision), daemon=True)
        worker.start()


def main():
    rospy.init_node("recovery_manager", anonymous=True)
    node = RecoveryManagerNode()
    rospy.logwarn("RecoveryManager running: fallback_confirm_count=%d fault_sample_timeout=%.2f cooldown=%.2f", node.fault_confirm_count, node.fault_sample_timeout, node.recovery_cooldown_sec)
    rospy.spin()


if __name__ == "__main__":
    main()
