#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
import os
import sys
from typing import Any, Dict, Iterable, List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

from scene_graph_system.diagnosis.fault_event import DeviationAlert, DiagnosisReport, RecoveryPlan, RecoveryRuntimeContext
from scene_graph_system.configuration.profile_validator import validate_profile
from scene_graph_system.configuration.rule_config_loader import ProfileConfig, load_profile, merge_profile_dicts


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _string_list(value: Any) -> List[str]:
    return [str(item) for item in _as_list(value) if str(item).strip()]


def _is_expression(value: Any) -> bool:
    return isinstance(value, str) and value.strip().startswith("$")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _type_ok(value: Any, expected_type: str) -> bool:
    expected_type = str(expected_type or "").strip()
    if expected_type == "bool":
        return isinstance(value, bool)
    if expected_type == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "float":
        return _is_number(value)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "list":
        return isinstance(value, list)
    if expected_type == "dict":
        return isinstance(value, dict)
    return True


def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _resolve_path(root: Any, path: str) -> Any:
    current = root
    for part in str(path or "").split("."):
        if not part:
            continue
        current = _get_attr_or_key(current, part, None)
        if current is None:
            return None
    return current


def _context_step_value(context: Optional[RecoveryRuntimeContext], alert: DeviationAlert) -> Optional[int]:
    value = _get_attr_or_key(context, "step", None)
    if value is None:
        value = _get_attr_or_key(alert, "execution_step", None)
    try:
        return int(value)
    except Exception:
        return None


class ConfigDrivenRecoveryPlanner:
    """
    Recovery-layer route and strategy builder.

    It only creates RecoveryPlan objects from profile.routing and
    profile.recovery. Execution remains in recovery_manager.py, so this layer
    cannot call arbitrary robot functions.
    """

    def __init__(self, profile: ProfileConfig):
        self.profile = profile
        self.recovery = profile.recovery
        self.routing = self._build_effective_routing(profile)
        self.action_schemas = dict(self.recovery.get("action_schemas", {}) or {})
        self.strategies = {
            str(item.get("id")): dict(item)
            for item in list(self.recovery.get("strategies", []) or [])
            if isinstance(item, dict) and item.get("id") is not None and bool(item.get("enabled", True))
        }
        self.rules = sorted(
            [
                dict(item)
                for item in list(self.routing.get("rules", []) or [])
                if isinstance(item, dict) and bool(item.get("enabled", True))
            ],
            key=lambda item: int(item.get("priority", 0) or 0),
            reverse=True,
        )

    def _build_effective_routing(self, profile: ProfileConfig) -> Dict[str, Any]:
        template_routing = self._collect_action_template_routing(profile)
        return merge_profile_dicts(template_routing, profile.routing)

    def _collect_action_template_routing(self, profile: ProfileConfig) -> Dict[str, Any]:
        planning = profile.planning
        bindings = dict(planning.get("action_template_bindings", {}) or {})
        if not bindings:
            return {}

        try:
            from scene_graph_system.configuration.action_template_loader import build_action_template_alias_map, load_all_action_templates

            templates = load_all_action_templates()
            alias_map = build_action_template_alias_map(templates)
        except Exception:
            return {}

        used_template_ids = set()
        for template_ref in bindings.values():
            template = alias_map.get(str(template_ref or "").strip())
            if template is not None:
                used_template_ids.add(str(template.template_id))

        rules = []
        for template in templates:
            if str(template.template_id) not in used_template_ids:
                continue
            data = template.to_dict()
            for route in list((data.get("recovery_defaults", {}) or {}).get("routes", []) or []):
                if not isinstance(route, dict):
                    continue
                fault_type = str(route.get("fault_type", "") or "").strip()
                strategy = str(route.get("strategy", "") or "").strip()
                if not fault_type or not strategy:
                    continue
                rules.append({
                    "id": str(route.get("id") or ("%s_%s_default_route" % (template.template_id, fault_type))),
                    "priority": int(route.get("priority", 50) or 50),
                    "when": {"fault_type": fault_type},
                    "strategy": strategy,
                    "source": "action_template:%s" % template.template_id,
                })
        return {"rules": rules} if rules else {}

    def build_plan(
        self,
        report: DiagnosisReport,
        alert: DeviationAlert,
        context: Optional[RecoveryRuntimeContext],
        decision: Any,
        *,
        plan_file: str = "",
        retry_count: int = 0,
    ) -> Optional[RecoveryPlan]:
        route = self._select_route(report, alert, context, decision, retry_count)
        if route is None:
            return None
        strategy_id = str(route.get("strategy", "") or "")
        strategy = self.strategies.get(strategy_id)
        if strategy is None:
            return None
        if not self._strategy_can_preserve_safety(strategy, decision):
            return None
        return self._build_plan_from_strategy(
            strategy=strategy,
            route=route,
            report=report,
            alert=alert,
            context=context,
            decision=decision,
            plan_file=plan_file,
            retry_count=retry_count,
        )

    def _select_route(
        self,
        report: DiagnosisReport,
        alert: DeviationAlert,
        context: Optional[RecoveryRuntimeContext],
        decision: Any,
        retry_count: int,
    ) -> Optional[Dict[str, Any]]:
        for rule in self.rules:
            if self._condition_passes(rule.get("when"), report, alert, context, decision, retry_count):
                return rule
        return None

    def _condition_passes(
        self,
        condition: Any,
        report: DiagnosisReport,
        alert: DeviationAlert,
        context: Optional[RecoveryRuntimeContext],
        decision: Any,
        retry_count: int,
    ) -> bool:
        if condition in (None, {}, []):
            return True
        if isinstance(condition, list):
            return all(self._condition_passes(item, report, alert, context, decision, retry_count) for item in condition)
        if not isinstance(condition, dict):
            return False

        if "all" in condition:
            return all(
                self._condition_passes(item, report, alert, context, decision, retry_count)
                for item in _as_list(condition.get("all"))
            )
        if "any" in condition:
            return any(
                self._condition_passes(item, report, alert, context, decision, retry_count)
                for item in _as_list(condition.get("any"))
            )
        if "not" in condition:
            return not self._condition_passes(condition.get("not"), report, alert, context, decision, retry_count)

        checks = []
        for key, value in condition.items():
            checks.append(self._leaf_passes(str(key), value, report, alert, context, decision, retry_count))
        return all(checks) if checks else True

    def _leaf_passes(
        self,
        key: str,
        value: Any,
        report: DiagnosisReport,
        alert: DeviationAlert,
        context: Optional[RecoveryRuntimeContext],
        decision: Any,
        retry_count: int,
    ) -> bool:
        if key == "fault_type":
            return str(report.fault_type) == str(value)
        if key == "fault_type_in":
            return str(report.fault_type) in set(_string_list(value))
        if key == "decision_kind":
            return str(_get_attr_or_key(decision, "kind", "")) == str(value)
        if key == "decision_kind_in":
            return str(_get_attr_or_key(decision, "kind", "")) in set(_string_list(value))
        if key == "recoverable":
            return bool(_get_attr_or_key(decision, "recoverable", False)) == bool(value)
        if key == "retry_count_gte":
            return int(retry_count or 0) >= int(value)
        if key == "retry_count_lte":
            return int(retry_count or 0) <= int(value)
        if key == "retry_count_eq":
            return int(retry_count or 0) == int(value)
        if key in {"severity", "alert_severity"}:
            return str(alert.severity) == str(value)
        if key == "context_status":
            return str(_get_attr_or_key(context, "status", "")) == str(value)
        if key == "context_action":
            return str(_get_attr_or_key(context, "action_type", "")) == str(value)
        if key == "context_action_in":
            return str(_get_attr_or_key(context, "action_type", "")) in set(_string_list(value))
        if key == "context_step":
            context_step = _context_step_value(context, alert)
            return context_step is not None and context_step == int(value)
        if key == "context_step_in":
            context_step = _context_step_value(context, alert)
            if context_step is None:
                return False
            expected = set()
            for item in _as_list(value):
                try:
                    expected.add(int(item))
                except Exception:
                    continue
            return context_step in expected
        return False

    def _strategy_can_preserve_safety(self, strategy: Dict[str, Any], decision: Any) -> bool:
        if not bool(_get_attr_or_key(decision, "need_return_held_object", False)):
            return True
        actions = list(strategy.get("actions", []) or [])
        action_names = {
            str(action.get("action", "") or "")
            for action in actions
            if isinstance(action, dict)
        }
        return "return_held_object_to_source" in action_names

    def _build_plan_from_strategy(
        self,
        strategy: Dict[str, Any],
        route: Dict[str, Any],
        report: DiagnosisReport,
        alert: DeviationAlert,
        context: Optional[RecoveryRuntimeContext],
        decision: Any,
        *,
        plan_file: str,
        retry_count: int,
    ) -> RecoveryPlan:
        actions = []
        for action_cfg in list(strategy.get("actions", []) or []):
            if not isinstance(action_cfg, dict):
                continue
            action_name = str(action_cfg.get("action", "") or "").strip()
            if action_name not in self.action_schemas:
                raise ValueError("action '%s' is not in recovery.action_schemas" % action_name)
            params = self._resolve_params(
                dict(action_cfg.get("params", {}) or {}),
                self.action_schemas.get(action_name, {}) or {},
                report=report,
                alert=alert,
                context=context,
                decision=decision,
                plan_file=plan_file,
                retry_count=retry_count,
            )
            action_out = {"action_type": action_name, "params": params}
            preconditions = list((self.action_schemas.get(action_name, {}) or {}).get("preconditions", []) or [])
            if preconditions:
                action_out["preconditions"] = preconditions
            actions.append(action_out)

        execution_step = _get_attr_or_key(context, "step", None)
        if execution_step is None:
            execution_step = alert.execution_step

        return RecoveryPlan(
            diagnosis_id=report.diagnosis_id,
            alert_id=report.alert_id,
            stage_name=report.stage_name,
            execution_step=execution_step,
            strategy_name=str(strategy.get("id", "") or ""),
            actions=actions,
            max_retry=int(strategy.get("max_retry", 1) or 0),
            abort_if_failed=bool(strategy.get("abort_if_failed", True)),
        )

    def _resolve_params(
        self,
        params: Dict[str, Any],
        schema: Dict[str, Any],
        *,
        report: DiagnosisReport,
        alert: DeviationAlert,
        context: Optional[RecoveryRuntimeContext],
        decision: Any,
        plan_file: str,
        retry_count: int,
    ) -> Dict[str, Any]:
        params_schema = dict(schema.get("params", {}) or {})
        resolved = {}
        for key, value in params.items():
            if key not in params_schema:
                raise ValueError("parameter '%s' is not defined by action schema" % key)
            param_schema = dict(params_schema.get(key, {}) or {})
            resolved_value = self._resolve_value(
                value,
                report=report,
                alert=alert,
                context=context,
                decision=decision,
                plan_file=plan_file,
                retry_count=retry_count,
            )
            self._validate_param_value(key, resolved_value, param_schema)
            resolved[key] = resolved_value
        for key, spec in params_schema.items():
            if bool((spec or {}).get("required", False)) and key not in resolved:
                if "default" in (spec or {}):
                    resolved[key] = copy.deepcopy((spec or {}).get("default"))
                else:
                    raise ValueError("required parameter '%s' is missing" % key)
        return resolved

    def _resolve_value(
        self,
        value: Any,
        *,
        report: DiagnosisReport,
        alert: DeviationAlert,
        context: Optional[RecoveryRuntimeContext],
        decision: Any,
        plan_file: str,
        retry_count: int,
    ) -> Any:
        if isinstance(value, list):
            return [
                self._resolve_value(
                    item,
                    report=report,
                    alert=alert,
                    context=context,
                    decision=decision,
                    plan_file=plan_file,
                    retry_count=retry_count,
                )
                for item in value
            ]
        if isinstance(value, dict):
            return {
                key: self._resolve_value(
                    item,
                    report=report,
                    alert=alert,
                    context=context,
                    decision=decision,
                    plan_file=plan_file,
                    retry_count=retry_count,
                )
                for key, item in value.items()
            }
        if not _is_expression(value):
            return copy.deepcopy(value)

        expr = str(value).strip()[1:]
        roots = {
            "context": context,
            "decision": decision,
            "alert": alert,
            "report": report,
            "plan_file": plan_file,
            "retry_count": retry_count,
        }
        if expr in roots:
            return roots[expr]
        root_name, _, rest = expr.partition(".")
        if root_name not in roots:
            raise ValueError("unsupported expression '%s'" % value)
        return _resolve_path(roots[root_name], rest)

    def _validate_param_value(self, key: str, value: Any, schema: Dict[str, Any]) -> None:
        expected_type = str(schema.get("type", "") or "")
        if expected_type and not _type_ok(value, expected_type):
            raise ValueError("parameter '%s' expected type '%s', got '%s'" % (key, expected_type, type(value).__name__))
        if _is_number(value):
            if schema.get("min") is not None and float(value) < float(schema.get("min")):
                raise ValueError("parameter '%s' value %.6g is below minimum %.6g" % (key, float(value), float(schema.get("min"))))
            if schema.get("max") is not None and float(value) > float(schema.get("max")):
                raise ValueError("parameter '%s' value %.6g is above maximum %.6g" % (key, float(value), float(schema.get("max"))))


def load_config_driven_recovery_planner(
    profile_name: str = "block_building",
    profile_root: Optional[str] = None,
) -> ConfigDrivenRecoveryPlanner:
    profile = load_profile(profile_name, profile_root=profile_root)
    errors = validate_profile(profile)
    if errors:
        raise ValueError("invalid profile %s: %s" % (profile.profile_id, "; ".join(str(e) for e in errors)))
    return ConfigDrivenRecoveryPlanner(profile)
