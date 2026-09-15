#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Set

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

from scene_graph_system.configuration.rule_config_loader import ProfileConfig, load_profile


VALIDATION_ONLY_FIELDS = {
    "object_coordinate",
    "within_region",
    "not_in_region",
    "node_has_state",
    "relation_exists",
    "count_in_region",
    "scene_graph",
    "expected",
}

DIAGNOSIS_ONLY_FIELDS = {
    "alert_issue_type",
    "alert_issue_type_in",
    "alert_object",
    "alert_object_in",
    "alert_relation_type",
    "alert_relation_type_in",
    "missing_state_any",
    "context_action",
    "context_action_in",
    "stage_name_pattern",
    "alert_severity",
    "alert_source",
    "runtime_gripper_state_any",
    "runtime_grasped_count_gte",
    "runtime_placed_count_gte",
    "fault_type_history_has",
}

DIAGNOSIS_CONDITION_KEYS = DIAGNOSIS_ONLY_FIELDS | {"all", "any", "not"}

VALID_VALIDATION_OPERATIONS = {
    "append_expected",
    "append_custom_check",
    "disable_check",
    "relax",
    "replace_phase",
}

VALID_OBJECT_MATCH_MODES = {"instance", "class", "count", "region"}
VALID_RELATION_MATCH_MODES = {"instance", "class"}

KNOWN_PRECONDITIONS = {
    "arm_not_in_error",
    "cleanup_observe_pose_reached",
    "dynamic_resume_step_available",
    "execution_control_available",
    "fresh_scene_available",
    "recovery_ready",
    "tcp_pose_available",
}

ROUTING_CONDITION_KEYS = {
    "all",
    "any",
    "not",
    "fault_type",
    "fault_type_in",
    "decision_kind",
    "decision_kind_in",
    "recoverable",
    "retry_count_gte",
    "retry_count_lte",
    "retry_count_eq",
    "severity",
    "alert_severity",
    "context_status",
    "context_action",
    "context_action_in",
    "context_step",
    "context_step_in",
}


@dataclass
class ProfileValidationError:
    path: str
    message: str

    def __str__(self) -> str:
        return "%s: %s" % (self.path, self.message)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _iter_condition_keys(node: Any) -> Iterable[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield str(key)
            yield from _iter_condition_keys(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_condition_keys(item)


def _list_ids(items: Iterable[Any]) -> List[str]:
    ids = []
    for item in list(items or []):
        if isinstance(item, dict) and item.get("id") is not None:
            ids.append(str(item.get("id")))
    return ids


def _duplicates(items: Iterable[str]) -> List[str]:
    seen = set()
    dup = []
    for item in items:
        if item in seen and item not in dup:
            dup.append(item)
        seen.add(item)
    return dup


def _section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = data.get(name, {}) or {}
    return value if isinstance(value, dict) else {}


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


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


class ProfileValidator:
    def validate(self, profile: ProfileConfig) -> List[ProfileValidationError]:
        errors: List[ProfileValidationError] = []
        data = profile.to_dict()

        self._validate_top_level(data, errors)
        self._validate_duplicate_ids(data, errors)
        self._validate_enabled_flags(data, errors)
        self._validate_validation_section(data, errors)
        self._validate_planning_section(data, errors)
        self._validate_diagnosis_section(data, errors)
        self._validate_recovery_section(data, errors)
        self._validate_routing_section(data, errors)

        return errors

    def _error(self, errors: List[ProfileValidationError], path: str, message: str) -> None:
        errors.append(ProfileValidationError(path=path, message=message))

    def _validate_top_level(self, data: Dict[str, Any], errors: List[ProfileValidationError]) -> None:
        if not str(data.get("profile_id", "") or "").strip():
            self._error(errors, "profile_id", "profile_id is required")

        for group in ("planning", "validation", "diagnosis", "recovery", "routing"):
            value = data.get(group, {})
            if value is not None and not isinstance(value, dict):
                self._error(errors, group, "section must be a mapping")

        if data.get("fault_types") is not None and not isinstance(data.get("fault_types"), list):
            self._error(errors, "fault_types", "fault_types must be a list")

    def _validate_duplicate_ids(self, data: Dict[str, Any], errors: List[ProfileValidationError]) -> None:
        validation = _section(data, "validation")
        diagnosis = _section(data, "diagnosis")
        recovery = _section(data, "recovery")
        routing = _section(data, "routing")
        lists = {
            "fault_types": data.get("fault_types", []),
            "validation.overrides": validation.get("overrides", []),
            "validation.custom_checks": validation.get("custom_checks", []),
            "diagnosis.rules": diagnosis.get("rules", []),
            "recovery.strategies": recovery.get("strategies", []),
            "routing.rules": routing.get("rules", []),
        }
        for path, items in lists.items():
            for duplicate in _duplicates(_list_ids(items)):
                self._error(errors, path, "duplicate id '%s'" % duplicate)

    def _validate_enabled_flags(self, data: Dict[str, Any], errors: List[ProfileValidationError]) -> None:
        validation = _section(data, "validation")
        diagnosis = _section(data, "diagnosis")
        recovery = _section(data, "recovery")
        routing = _section(data, "routing")
        lists = {
            "fault_types": data.get("fault_types", []),
            "validation.overrides": validation.get("overrides", []),
            "validation.custom_checks": validation.get("custom_checks", []),
            "diagnosis.rules": diagnosis.get("rules", []),
            "recovery.strategies": recovery.get("strategies", []),
            "routing.rules": routing.get("rules", []),
        }
        for path, items in lists.items():
            for idx, item in enumerate(list(items or [])):
                if isinstance(item, dict) and item.get("enabled") is not None and not isinstance(item.get("enabled"), bool):
                    self._error(errors, "%s[%d].enabled" % (path, idx), "enabled must be a bool")

    def _defined_fault_types(self, data: Dict[str, Any]) -> Set[str]:
        return {
            str(item.get("id"))
            for item in list(data.get("fault_types", []) or [])
            if isinstance(item, dict) and item.get("id") is not None
        }

    def _defined_strategies(self, data: Dict[str, Any]) -> Set[str]:
        return {
            str(item.get("id"))
            for item in list(_section(data, "recovery").get("strategies", []) or [])
            if isinstance(item, dict) and item.get("id") is not None
        }

    def _validate_validation_section(self, data: Dict[str, Any], errors: List[ProfileValidationError]) -> None:
        validation = _section(data, "validation")
        comparison = validation.get("comparison", {}) or {}
        if comparison and not isinstance(comparison, dict):
            self._error(errors, "validation.comparison", "comparison must be a mapping")
        elif isinstance(comparison, dict):
            object_mode = comparison.get("object_match_mode")
            if object_mode is not None and str(object_mode) not in VALID_OBJECT_MATCH_MODES:
                self._error(
                    errors,
                    "validation.comparison.object_match_mode",
                    "must be one of %s" % sorted(VALID_OBJECT_MATCH_MODES),
                )
            relation_mode = comparison.get("relation_match_mode")
            if relation_mode is not None and str(relation_mode) not in VALID_RELATION_MATCH_MODES:
                self._error(
                    errors,
                    "validation.comparison.relation_match_mode",
                    "must be one of %s" % sorted(VALID_RELATION_MATCH_MODES),
                )

        for path, rules in (
            ("validation.overrides", validation.get("overrides", [])),
            ("validation.custom_checks", validation.get("custom_checks", [])),
        ):
            if rules is None:
                continue
            if not isinstance(rules, list):
                self._error(errors, path, "must be a list")
                continue
            for idx, rule in enumerate(rules):
                rule_path = "%s[%d]" % (path, idx)
                if not isinstance(rule, dict):
                    self._error(errors, rule_path, "rule must be a mapping")
                    continue
                if not str(rule.get("id", "") or "").strip():
                    self._error(errors, rule_path + ".id", "id is required")
                self._validate_validation_scope(rule.get("scope"), rule_path + ".scope", errors)
                operation = rule.get("operation")
                if path == "validation.overrides" and operation is not None:
                    if str(operation) not in VALID_VALIDATION_OPERATIONS:
                        self._error(
                            errors,
                            rule_path + ".operation",
                            "must be one of %s" % sorted(VALID_VALIDATION_OPERATIONS),
                        )
                when = rule.get("when")
                for key in _iter_condition_keys(when):
                    if key.startswith("alert_") or key in DIAGNOSIS_ONLY_FIELDS:
                        self._error(
                            errors,
                            rule_path + ".when",
                            "validation rules cannot reference diagnosis field '%s'" % key,
                        )

    def _validate_validation_scope(self, scope: Any, path: str, errors: List[ProfileValidationError]) -> None:
        if scope is None:
            return
        if not isinstance(scope, dict):
            self._error(errors, path, "scope must be a mapping")
            return
        level = str(scope.get("level", "") or "").strip()
        if level not in {"action", "step", "pattern"}:
            self._error(errors, path + ".level", "must be one of ['action', 'pattern', 'step']")
        for key in ("action_type", "template_id", "stage_id"):
            if scope.get(key) is not None and not isinstance(scope.get(key), str):
                self._error(errors, path + "." + key, "must be a string")
        if level == "action" and not str(scope.get("stage_id", "") or "").strip():
            self._error(errors, path + ".stage_id", "stage_id is required for action scope")
        if level == "step":
            if not isinstance(scope.get("step"), int) or isinstance(scope.get("step"), bool) or int(scope.get("step")) < 1:
                self._error(errors, path + ".step", "step scope requires a positive integer step")
            if not str(scope.get("stage_id", "") or "").strip():
                self._error(errors, path + ".stage_id", "stage_id is required for step scope")

    def _validate_planning_section(self, data: Dict[str, Any], errors: List[ProfileValidationError]) -> None:
        planning = _section(data, "planning")
        if not planning:
            return

        for key in ("task_plan_ref", "capability_ref", "validation_mode"):
            if planning.get(key) is not None and not isinstance(planning.get(key), str):
                self._error(errors, "planning.%s" % key, "must be a string")

        validation_mode = planning.get("validation_mode")
        if validation_mode is not None and str(validation_mode) not in {"constrained", "advanced"}:
            self._error(errors, "planning.validation_mode", "must be one of ['advanced', 'constrained']")

        bindings = planning.get("action_template_bindings", {}) or {}
        if bindings is not None and not isinstance(bindings, dict):
            self._error(errors, "planning.action_template_bindings", "must be a mapping")
        elif isinstance(bindings, dict):
            for action_type, template_ref in bindings.items():
                if not str(action_type or "").strip():
                    self._error(errors, "planning.action_template_bindings", "action_type key must be non-empty")
                if not isinstance(template_ref, str) or not template_ref.strip():
                    self._error(
                        errors,
                        "planning.action_template_bindings.%s" % str(action_type),
                        "template ref must be a non-empty string",
                    )

        if planning.get("strict_action_templates") is not None and not isinstance(planning.get("strict_action_templates"), bool):
            self._error(errors, "planning.strict_action_templates", "must be a bool")

    def _validate_diagnosis_section(self, data: Dict[str, Any], errors: List[ProfileValidationError]) -> None:
        fault_types = self._defined_fault_types(data)
        diagnosis = _section(data, "diagnosis")
        rules = diagnosis.get("rules", []) or []
        if not isinstance(rules, list):
            self._error(errors, "diagnosis.rules", "must be a list")
            return

        for idx, rule in enumerate(rules):
            path = "diagnosis.rules[%d]" % idx
            if not isinstance(rule, dict):
                self._error(errors, path, "rule must be a mapping")
                continue

            rule_id = str(rule.get("id", "") or "").strip()
            if not rule_id:
                self._error(errors, path + ".id", "id is required")

            fault_type = str(rule.get("fault_type", "") or "").strip()
            if not fault_type:
                self._error(errors, path + ".fault_type", "fault_type is required")
            elif fault_type not in fault_types:
                self._error(errors, path + ".fault_type", "undefined fault_type '%s'" % fault_type)

            confidence = rule.get("confidence")
            if confidence is not None:
                if not _is_number(confidence) or float(confidence) < 0.0 or float(confidence) > 1.0:
                    self._error(errors, path + ".confidence", "confidence must be within [0.0, 1.0]")

            for key in _iter_condition_keys(rule.get("when")):
                if key.startswith("scene_graph") or key.startswith("expected") or key in VALIDATION_ONLY_FIELDS:
                    self._error(
                        errors,
                        path + ".when",
                        "diagnosis rules cannot reference validation field '%s'" % key,
                    )
                if key.startswith("retry_count"):
                    self._error(
                        errors,
                        path + ".when",
                        "retry_count conditions belong in routing.rules, not diagnosis.rules",
                    )
            self._validate_diagnosis_condition_types(rule.get("when"), path + ".when", errors)

    def _validate_diagnosis_condition_types(
        self,
        node: Any,
        path: str,
        errors: List[ProfileValidationError],
    ) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                child_path = "%s.%s" % (path, key)
                if key in {"all", "any"}:
                    if not isinstance(value, list):
                        self._error(errors, child_path, "must be a list")
                    self._validate_diagnosis_condition_types(value, child_path, errors)
                    continue
                if key == "not":
                    self._validate_diagnosis_condition_types(value, child_path, errors)
                    continue
                if key in {"alert_issue_type", "alert_object", "alert_relation_type"}:
                    if not isinstance(value, str):
                        self._error(errors, child_path, "must be a string")
                    continue
                if key in {
                    "alert_issue_type_in",
                    "alert_object_in",
                    "alert_relation_type_in",
                    "missing_state_any",
                    "context_action_in",
                    "runtime_gripper_state_any",
                }:
                    if not _is_string_list(value):
                        self._error(errors, child_path, "must be a list of strings")
                    continue
                if key in {"context_action", "stage_name_pattern", "alert_severity", "alert_source"}:
                    if not isinstance(value, str):
                        self._error(errors, child_path, "must be a string")
                    continue
                if key in {"runtime_grasped_count_gte", "runtime_placed_count_gte"}:
                    if not isinstance(value, int) or isinstance(value, bool) or int(value) < 0:
                        self._error(errors, child_path, "must be a non-negative integer")
                    continue
                if key == "fault_type_history_has":
                    self._error(errors, child_path, "fault_type_history_has is reserved for a future diagnosis history store")
                    continue
                if key not in DIAGNOSIS_CONDITION_KEYS:
                    self._error(errors, child_path, "unsupported diagnosis condition '%s'" % key)
                    continue
                self._validate_diagnosis_condition_types(value, child_path, errors)
        elif isinstance(node, list):
            for idx, item in enumerate(node):
                self._validate_diagnosis_condition_types(item, "%s[%d]" % (path, idx), errors)

    def _validate_recovery_section(self, data: Dict[str, Any], errors: List[ProfileValidationError]) -> None:
        recovery = _section(data, "recovery")
        action_schemas = recovery.get("action_schemas", {}) or {}
        strategies = recovery.get("strategies", []) or []

        if not isinstance(action_schemas, dict):
            self._error(errors, "recovery.action_schemas", "must be a mapping")
            action_schemas = {}

        for action_name, schema in action_schemas.items():
            path = "recovery.action_schemas.%s" % action_name
            if not isinstance(schema, dict):
                self._error(errors, path, "action schema must be a mapping")
                continue
            params_schema = schema.get("params", {}) or {}
            if not isinstance(params_schema, dict):
                self._error(errors, path + ".params", "params must be a mapping")
            preconditions = schema.get("preconditions", []) or []
            if not isinstance(preconditions, list):
                self._error(errors, path + ".preconditions", "preconditions must be a list")
                continue
            for precondition in preconditions:
                if str(precondition) not in KNOWN_PRECONDITIONS:
                    self._error(errors, path + ".preconditions", "unknown precondition '%s'" % precondition)

        if not isinstance(strategies, list):
            self._error(errors, "recovery.strategies", "must be a list")
            return

        for idx, strategy in enumerate(strategies):
            path = "recovery.strategies[%d]" % idx
            if not isinstance(strategy, dict):
                self._error(errors, path, "strategy must be a mapping")
                continue
            if not str(strategy.get("id", "") or "").strip():
                self._error(errors, path + ".id", "id is required")
            actions = strategy.get("actions", []) or []
            if not isinstance(actions, list):
                self._error(errors, path + ".actions", "actions must be a list")
                continue
            for action_idx, action in enumerate(actions):
                self._validate_strategy_action(
                    action,
                    action_schemas,
                    "%s.actions[%d]" % (path, action_idx),
                    errors,
                )

    def _validate_strategy_action(
        self,
        action: Any,
        action_schemas: Dict[str, Any],
        path: str,
        errors: List[ProfileValidationError],
    ) -> None:
        if not isinstance(action, dict):
            self._error(errors, path, "action must be a mapping")
            return

        action_name = str(action.get("action", "") or "").strip()
        if not action_name:
            self._error(errors, path + ".action", "action name is required")
            return

        if action_name not in action_schemas:
            self._error(errors, path + ".action", "action '%s' is not in recovery.action_schemas" % action_name)
            return

        schema = action_schemas.get(action_name, {}) or {}
        params_schema = schema.get("params", {}) or {}
        params = action.get("params", {}) or {}

        if not isinstance(params, dict):
            self._error(errors, path + ".params", "params must be a mapping")
            return

        for param_name in params.keys():
            if param_name not in params_schema:
                self._error(errors, path + ".params.%s" % param_name, "parameter is not defined by action schema")

        for param_name, spec in params_schema.items():
            spec = spec or {}
            spec_path = "%s.params.%s" % (path, param_name)
            if not isinstance(spec, dict):
                self._error(errors, spec_path, "parameter schema must be a mapping")
                continue
            if bool(spec.get("required", False)) and param_name not in params and "default" not in spec:
                self._error(errors, spec_path, "required parameter is missing")
                continue
            if param_name not in params:
                continue
            self._validate_param_value(params[param_name], spec, spec_path, errors)

    def _validate_param_value(
        self,
        value: Any,
        spec: Dict[str, Any],
        path: str,
        errors: List[ProfileValidationError],
    ) -> None:
        if _is_expression(value):
            if bool(spec.get("allow_expression", False)):
                return
            self._error(errors, path, "expressions are not allowed for this parameter")
            return

        expected_type = str(spec.get("type", "") or "")
        if expected_type and not _type_ok(value, expected_type):
            self._error(errors, path, "expected type '%s', got '%s'" % (expected_type, type(value).__name__))
            return

        if _is_number(value):
            if spec.get("min") is not None and float(value) < float(spec.get("min")):
                self._error(errors, path, "value %.6g is below minimum %.6g" % (float(value), float(spec.get("min"))))
            if spec.get("max") is not None and float(value) > float(spec.get("max")):
                self._error(errors, path, "value %.6g is above maximum %.6g" % (float(value), float(spec.get("max"))))

    def _validate_routing_section(self, data: Dict[str, Any], errors: List[ProfileValidationError]) -> None:
        routing = _section(data, "routing")
        rules = routing.get("rules", []) or []
        strategies = self._defined_strategies(data)
        fault_types = self._defined_fault_types(data)

        if not isinstance(rules, list):
            self._error(errors, "routing.rules", "must be a list")
            return

        for idx, rule in enumerate(rules):
            path = "routing.rules[%d]" % idx
            if not isinstance(rule, dict):
                self._error(errors, path, "rule must be a mapping")
                continue

            if not str(rule.get("id", "") or "").strip():
                self._error(errors, path + ".id", "id is required")

            strategy = str(rule.get("strategy", "") or "").strip()
            if not strategy:
                self._error(errors, path + ".strategy", "strategy is required")
            elif strategy not in strategies:
                self._error(errors, path + ".strategy", "undefined strategy '%s'" % strategy)

            when = rule.get("when")
            for key in _iter_condition_keys(when):
                if key.startswith("scene_graph") or key.startswith("expected") or key in VALIDATION_ONLY_FIELDS:
                    self._error(errors, path + ".when", "routing cannot reference validation field '%s'" % key)
                elif key not in ROUTING_CONDITION_KEYS:
                    self._error(errors, path + ".when", "unsupported routing condition '%s'" % key)

            self._validate_routing_fault_refs(when, fault_types, path + ".when", errors)
            self._validate_routing_condition_types(when, path + ".when", errors)

    def _validate_routing_fault_refs(
        self,
        node: Any,
        fault_types: Set[str],
        path: str,
        errors: List[ProfileValidationError],
    ) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "fault_type":
                    if str(value) not in fault_types:
                        self._error(errors, path + ".fault_type", "undefined fault_type '%s'" % value)
                elif key == "fault_type_in":
                    for item in _as_list(value):
                        if str(item) not in fault_types:
                            self._error(errors, path + ".fault_type_in", "undefined fault_type '%s'" % item)
                else:
                    self._validate_routing_fault_refs(value, fault_types, path, errors)
        elif isinstance(node, list):
            for item in node:
                self._validate_routing_fault_refs(item, fault_types, path, errors)

    def _validate_routing_condition_types(
        self,
        node: Any,
        path: str,
        errors: List[ProfileValidationError],
    ) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                child_path = "%s.%s" % (path, key)
                if key in {"all", "any"}:
                    if not isinstance(value, list):
                        self._error(errors, child_path, "must be a list")
                    self._validate_routing_condition_types(value, child_path, errors)
                    continue
                if key == "not":
                    self._validate_routing_condition_types(value, child_path, errors)
                    continue
                if key in {"fault_type", "decision_kind", "severity", "alert_severity", "context_status", "context_action"}:
                    if not isinstance(value, str):
                        self._error(errors, child_path, "must be a string")
                    continue
                if key in {"fault_type_in", "decision_kind_in", "context_action_in"}:
                    if not _is_string_list(value):
                        self._error(errors, child_path, "must be a list of strings")
                    continue
                if key == "recoverable":
                    if not isinstance(value, bool):
                        self._error(errors, child_path, "must be a bool")
                    continue
                if key in {"retry_count_gte", "retry_count_lte", "retry_count_eq"}:
                    if not isinstance(value, int) or isinstance(value, bool) or int(value) < 0:
                        self._error(errors, child_path, "must be a non-negative integer")
                    continue
                if key == "context_step":
                    if not isinstance(value, int) or isinstance(value, bool) or int(value) < 1:
                        self._error(errors, child_path, "must be a positive integer")
                    continue
                if key == "context_step_in":
                    if not isinstance(value, list) or not all(isinstance(item, int) and not isinstance(item, bool) and item >= 1 for item in value):
                        self._error(errors, child_path, "must be a list of positive integers")
                    continue
                self._validate_routing_condition_types(value, child_path, errors)
        elif isinstance(node, list):
            for idx, item in enumerate(node):
                self._validate_routing_condition_types(item, "%s[%d]" % (path, idx), errors)


def validate_profile(profile: ProfileConfig) -> List[ProfileValidationError]:
    return ProfileValidator().validate(profile)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a merged task profile.")
    parser.add_argument("profile", nargs="?", default="block_building")
    parser.add_argument("--profile-root", default=None)
    args = parser.parse_args()

    profile = load_profile(args.profile, profile_root=args.profile_root)
    errors = validate_profile(profile)

    if errors:
        print("Profile validation failed: %s" % profile.profile_id)
        for error in errors:
            print("  - %s" % error)
        return 1

    print("Profile validation passed: %s" % profile.profile_id)
    print("Source files:")
    for source in profile.source_files:
        print("  - %s" % source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
