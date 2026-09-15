#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import fnmatch
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

from scene_graph_system.diagnosis.fault_event import DeviationAlert, DiagnosisReport, RecoveryRuntimeContext
from scene_graph_system.configuration.profile_validator import validate_profile
from scene_graph_system.configuration.rule_config_loader import ProfileConfig, load_profile


def stage_action_type(stage_name: str) -> str:
    text = str(stage_name or "")
    if "__" in text:
        return text.rsplit("__", 1)[1].strip()
    return text.strip()


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _string_list(value: Any) -> List[str]:
    return [str(item) for item in _as_list(value) if str(item).strip()]


def _normalized_set(value: Any) -> set:
    return {str(item).strip().lower() for item in _as_list(value) if str(item).strip()}


def _issue_types(alert: DeviationAlert) -> List[str]:
    return sorted({
        str(item.get("issue_type", "") or "").strip()
        for item in list(alert.issues or [])
        if str(item.get("issue_type", "") or "").strip()
    })


def _issue_involves_object(issue: Dict[str, Any], object_id: str) -> bool:
    target = str(object_id or "").strip().lower()
    if not target:
        return False
    for key in ("node_id", "subject_id", "object_id"):
        if str(issue.get(key, "") or "").strip().lower() == target:
            return True
    return target in str(issue.get("message", "") or "").lower()


def _issue_missing_any_state(issue: Dict[str, Any], states: Iterable[str]) -> bool:
    wanted = {str(item).strip().lower() for item in states if str(item).strip()}
    if not wanted:
        return False

    metadata = dict(issue.get("metadata", {}) or {})
    candidates = [
        str(metadata.get("missing_state", "") or ""),
        str(issue.get("expected", "") or ""),
        str(issue.get("message", "") or ""),
    ]
    text = " ".join(candidates).lower()
    return any(state in text for state in wanted)


def _relation_from_issue(issue: Dict[str, Any]) -> str:
    relation = str(issue.get("relation", "") or "").strip()
    if relation:
        return relation
    metadata = dict(issue.get("metadata", {}) or {})
    return str(metadata.get("relation", "") or "").strip()


def _runtime_summary(alert: DeviationAlert) -> Dict[str, Any]:
    return dict(getattr(alert, "runtime_summary", {}) or {})


def _context_action(alert: DeviationAlert, runtime_ctx: Optional[RecoveryRuntimeContext]) -> str:
    actions = _context_actions(alert, runtime_ctx)
    return actions[0] if actions else ""


def _context_actions(alert: DeviationAlert, runtime_ctx: Optional[RecoveryRuntimeContext]) -> List[str]:
    actions: List[str] = []
    stage_action = stage_action_type(getattr(alert, "stage_name", ""))
    if stage_action:
        actions.append(stage_action)
    if runtime_ctx is not None:
        action = str(getattr(runtime_ctx, "action_type", "") or "").strip()
        if action and action not in actions:
            actions.append(action)
    return actions


def _runtime_count(alert: DeviationAlert, field: str) -> int:
    value = _runtime_summary(alert).get(field, []) or []
    if isinstance(value, (list, tuple, set)):
        return len(value)
    try:
        return int(value)
    except Exception:
        return 0


class ConfigDrivenDiagnoser:
    """
    Diagnosis-layer rule engine.

    It consumes only DeviationAlert plus optional runtime context. It does not
    access SceneGraph or ExpectedState, preserving the validation/diagnosis
    boundary introduced by the Profile schema.
    """

    def __init__(self, profile: ProfileConfig):
        self.profile = profile
        self.profile_data = profile.to_dict()
        self.rules = self._sorted_rules()
        self.fault_types = self._fault_type_map()

    def _sorted_rules(self) -> List[Dict[str, Any]]:
        rules = list(self.profile.diagnosis.get("rules", []) or [])
        return sorted(
            [rule for rule in rules if isinstance(rule, dict) and bool(rule.get("enabled", True))],
            key=lambda rule: int(rule.get("priority", 0) or 0),
            reverse=True,
        )

    def _fault_type_map(self) -> Dict[str, Dict[str, Any]]:
        out = {}
        for item in list(self.profile.fault_types or []):
            if isinstance(item, dict) and item.get("id") is not None:
                out[str(item.get("id"))] = dict(item)
        return out

    def try_diagnose(
        self,
        alert: DeviationAlert,
        runtime_ctx: Optional[RecoveryRuntimeContext] = None,
    ) -> Optional[DiagnosisReport]:
        for rule in self.rules:
            if self._condition_passes(rule.get("when"), alert, runtime_ctx):
                return self._make_report(alert, runtime_ctx, rule)
        return None

    def _make_report(
        self,
        alert: DeviationAlert,
        runtime_ctx: Optional[RecoveryRuntimeContext],
        rule: Dict[str, Any],
    ) -> DiagnosisReport:
        fault_type = str(rule.get("fault_type", "") or "unknown_fault")
        fault_def = self.fault_types.get(fault_type, {})
        confidence = float(rule.get("confidence", 0.75) or 0.75)
        retryable = bool(fault_def.get("retryable", True))
        root_cause = str(
            rule.get("root_cause")
            or rule.get("description")
            or fault_def.get("description")
            or "matched configured diagnosis rule"
        )
        action_type = _context_action(alert, runtime_ctx)

        return DiagnosisReport(
            alert_id=alert.alert_id,
            stage_name=alert.stage_name,
            execution_step=alert.execution_step,
            severity=alert.severity,
            fault_type=fault_type,
            root_cause=root_cause,
            confidence=confidence,
            retryable=retryable,
            details={
                "diagnosis_mode": "configured",
                "profile_id": self.profile.profile_id,
                "matched_rule": str(rule.get("id", "") or ""),
                "rule_priority": int(rule.get("priority", 0) or 0),
                "issue_types": _issue_types(alert),
                "runtime_summary": _runtime_summary(alert),
                "action_type": action_type,
                "context_step": getattr(runtime_ctx, "step", None) if runtime_ctx is not None else None,
            },
        )

    def _condition_passes(
        self,
        condition: Any,
        alert: DeviationAlert,
        runtime_ctx: Optional[RecoveryRuntimeContext],
    ) -> bool:
        if condition in (None, {}, []):
            return True
        if isinstance(condition, list):
            return all(self._condition_passes(item, alert, runtime_ctx) for item in condition)
        if not isinstance(condition, dict):
            return False

        if "all" in condition:
            return all(self._condition_passes(item, alert, runtime_ctx) for item in _as_list(condition.get("all")))
        if "any" in condition:
            return any(self._condition_passes(item, alert, runtime_ctx) for item in _as_list(condition.get("any")))
        if "not" in condition:
            return not self._condition_passes(condition.get("not"), alert, runtime_ctx)

        checks = []
        for key, value in condition.items():
            checks.append(self._leaf_condition_passes(str(key), value, alert, runtime_ctx))
        return all(checks) if checks else True

    def _leaf_condition_passes(
        self,
        key: str,
        value: Any,
        alert: DeviationAlert,
        runtime_ctx: Optional[RecoveryRuntimeContext],
    ) -> bool:
        issues = list(getattr(alert, "issues", []) or [])
        key = str(key or "").strip()

        if key == "alert_issue_type":
            expected = str(value or "").strip()
            return any(str(issue.get("issue_type", "") or "").strip() == expected for issue in issues)

        if key == "alert_issue_type_in":
            expected = set(_string_list(value))
            return any(str(issue.get("issue_type", "") or "").strip() in expected for issue in issues)

        if key == "alert_object":
            return any(_issue_involves_object(issue, str(value)) for issue in issues)

        if key == "alert_object_in":
            objects = _string_list(value)
            return any(_issue_involves_object(issue, obj) for issue in issues for obj in objects)

        if key == "alert_relation_type":
            expected = str(value or "").strip()
            return any(_relation_from_issue(issue) == expected for issue in issues)

        if key == "alert_relation_type_in":
            expected = set(_string_list(value))
            return any(_relation_from_issue(issue) in expected for issue in issues)

        if key == "missing_state_any":
            states = _string_list(value)
            return any(_issue_missing_any_state(issue, states) for issue in issues)

        if key == "context_action":
            return str(value or "").strip() in _context_actions(alert, runtime_ctx)

        if key == "context_action_in":
            expected = set(_string_list(value))
            return bool(expected & set(_context_actions(alert, runtime_ctx)))

        if key == "stage_name_pattern":
            return fnmatch.fnmatchcase(str(getattr(alert, "stage_name", "") or ""), str(value or "*"))

        if key == "alert_severity":
            return str(getattr(alert, "severity", "") or "") == str(value or "")

        if key == "alert_source":
            return str(getattr(alert, "source", "") or "") == str(value or "")

        if key == "runtime_gripper_state_any":
            observed = _normalized_set(_runtime_summary(alert).get("gripper_states", []))
            return bool(observed & _normalized_set(value))

        if key == "runtime_grasped_count_gte":
            return _runtime_count(alert, "grasped_object_ids") >= int(value)

        if key == "runtime_placed_count_gte":
            return _runtime_count(alert, "placed_object_ids") >= int(value)

        return False


def load_config_driven_diagnoser(
    profile_name: str = "block_building",
    profile_root: Optional[str] = None,
) -> ConfigDrivenDiagnoser:
    profile = load_profile(profile_name, profile_root=profile_root)
    errors = validate_profile(profile)
    if errors:
        raise ValueError("invalid profile %s: %s" % (profile.profile_id, "; ".join(str(e) for e in errors)))
    return ConfigDrivenDiagnoser(profile)


def diagnose_with_config(
    alert: DeviationAlert,
    runtime_ctx: Optional[RecoveryRuntimeContext] = None,
    profile_name: str = "block_building",
    profile_root: Optional[str] = None,
) -> Optional[DiagnosisReport]:
    return load_config_driven_diagnoser(profile_name, profile_root).try_diagnose(alert, runtime_ctx)
