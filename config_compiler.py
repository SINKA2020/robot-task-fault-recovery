#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

from scene_graph_system.configuration.action_template_loader import (
    ActionTemplate,
    build_action_template_alias_map,
    get_default_action_template_root,
    load_all_action_templates,
)
from scene_graph_system.configuration.capability_loader import CapabilityConfig, load_capability
from scene_graph_system.planning.configurable_expected_state_builder import build_configured_process_supervision_registry
from scene_graph_system.planning.generate_expected_state_ai import build_process_stage_names_for_execution_step
from scene_graph_system.configuration.profile_validator import validate_profile
from scene_graph_system.configuration.rule_config_loader import ProfileConfig, load_profile
from scene_graph_system.planning.task_plan_loader import (
    build_internal_plan_from_file,
    get_default_action_plan_file,
    load_task_plan,
    summarize_internal_plan,
)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _string_set(value: Any) -> set:
    return {str(item).strip() for item in _as_list(value) if str(item).strip()}


def _section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = data.get(name, {}) or {}
    return value if isinstance(value, dict) else {}


def _relation_specs(rule: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for key in ("relations", "forbidden_relations"):
        for item in list(rule.get(key, []) or []):
            if isinstance(item, dict):
                yield item
    raw_state = dict(rule.get("expected_state", {}) or {})
    for key in ("relations", "forbidden_relations"):
        for item in list(raw_state.get(key, []) or []):
            if isinstance(item, dict):
                yield item


def _node_specs(rule: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for item in list(rule.get("nodes", []) or []):
        if isinstance(item, dict):
            yield item
    raw_state = dict(rule.get("expected_state", {}) or {})
    for item in list(raw_state.get("nodes", []) or []):
        if isinstance(item, dict):
            yield item


def _condition_operator_keys(node: Any) -> Iterable[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            key = str(key)
            if key in {"all", "any", "not"}:
                yield from _condition_operator_keys(value)
            else:
                yield key
    elif isinstance(node, list):
        for item in node:
            yield from _condition_operator_keys(item)


def _condition_value_for_operator(node: Any, operator: str) -> Iterable[Any]:
    if isinstance(node, dict):
        for key, value in node.items():
            if str(key) == operator:
                yield value
            yield from _condition_value_for_operator(value, operator)
    elif isinstance(node, list):
        for item in node:
            yield from _condition_value_for_operator(item, operator)


def _profile_planning(profile: ProfileConfig) -> Dict[str, Any]:
    return _section(profile.to_dict(), "planning")


def _capability_sets(capability: CapabilityConfig) -> Dict[str, set]:
    validation = capability.validation
    comparison = capability.comparison
    recovery = capability.recovery
    return {
        "operations": _string_set(validation.get("operations", [])),
        "states": _string_set(validation.get("node_states", [])),
        "relations": _string_set(validation.get("relations", [])),
        "custom_ops": _string_set(validation.get("custom_check_operators", [])),
        "object_match_modes": _string_set(comparison.get("object_match_modes", [])),
        "relation_match_modes": _string_set(comparison.get("relation_match_modes", [])),
        "recovery_actions": _string_set(recovery.get("executable_actions", [])),
    }


def _state_names_from_node(node: Dict[str, Any]) -> Iterable[str]:
    for key in ("required_states", "optional_states", "required_states_any", "states_any", "states_all"):
        for state in _as_list(node.get(key)):
            if str(state).strip():
                yield str(state).strip()


def _validate_profile_against_capabilities(
    profile: ProfileConfig,
    capability: CapabilityConfig,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    errors: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []
    caps = _capability_sets(capability)
    data = profile.to_dict()
    planning = _profile_planning(profile)
    mode = str(planning.get("validation_mode") or capability.validation.get("mode") or "constrained")

    validation = _section(data, "validation")
    comparison = validation.get("comparison", {}) or {}
    if isinstance(comparison, dict):
        object_mode = comparison.get("object_match_mode")
        if object_mode is not None and caps["object_match_modes"] and str(object_mode) not in caps["object_match_modes"]:
            errors.append({
                "path": "validation.comparison.object_match_mode",
                "message": "mode '%s' is not supported by capability '%s'" % (object_mode, capability.capability_id),
            })
        relation_mode = comparison.get("relation_match_mode")
        if relation_mode is not None and caps["relation_match_modes"] and str(relation_mode) not in caps["relation_match_modes"]:
            errors.append({
                "path": "validation.comparison.relation_match_mode",
                "message": "mode '%s' is not supported by capability '%s'" % (relation_mode, capability.capability_id),
            })

    if mode != "constrained":
        warnings.append({
            "path": "planning.validation_mode",
            "message": "advanced validation mode bypasses part of the semi-custom capability check",
        })
        return errors, warnings

    for group_name in ("overrides", "custom_checks"):
        for idx, rule in enumerate(list(validation.get(group_name, []) or [])):
            if not isinstance(rule, dict):
                continue
            rule_path = "validation.%s[%d]" % (group_name, idx)
            operation = str(rule.get("operation", "append_custom_check" if group_name == "custom_checks" else "") or "")
            if operation and caps["operations"] and operation not in caps["operations"]:
                errors.append({
                    "path": rule_path + ".operation",
                    "message": "operation '%s' is not in the supported operation list" % operation,
                })

            for node in _node_specs(rule):
                for state in _state_names_from_node(node):
                    if caps["states"] and state not in caps["states"]:
                        errors.append({
                            "path": rule_path + ".nodes",
                            "message": "node state '%s' is not supported by current capabilities" % state,
                        })

            for rel in _relation_specs(rule):
                relation = str(rel.get("relation", "") or "").strip()
                if relation and caps["relations"] and relation not in caps["relations"]:
                    errors.append({
                        "path": rule_path + ".relations",
                        "message": "relation '%s' is not supported by current capabilities" % relation,
                    })

            for operator in _condition_operator_keys(rule.get("when")):
                if caps["custom_ops"] and operator not in caps["custom_ops"]:
                    errors.append({
                        "path": rule_path + ".when",
                        "message": "condition operator '%s' is not supported in constrained validation mode" % operator,
                    })

            for spec in _condition_value_for_operator(rule.get("when"), "node_has_state"):
                if isinstance(spec, dict):
                    for state in _state_names_from_node(spec):
                        if caps["states"] and state not in caps["states"]:
                            errors.append({
                                "path": rule_path + ".when.node_has_state",
                                "message": "node state '%s' is not supported by current capabilities" % state,
                            })

            for spec in _condition_value_for_operator(rule.get("when"), "relation_exists"):
                if isinstance(spec, dict):
                    relation = str(spec.get("relation", "") or "").strip()
                    if relation and caps["relations"] and relation not in caps["relations"]:
                        errors.append({
                            "path": rule_path + ".when.relation_exists",
                            "message": "relation '%s' is not supported by current capabilities" % relation,
                        })

    recovery = _section(data, "recovery")
    action_schemas = dict(recovery.get("action_schemas", {}) or {})
    for action_name in action_schemas.keys():
        if caps["recovery_actions"] and str(action_name) not in caps["recovery_actions"]:
            errors.append({
                "path": "recovery.action_schemas.%s" % action_name,
                "message": "recovery action is not executable according to current capabilities",
            })

    return errors, warnings


def _validate_templates_against_capabilities(
    templates: List[ActionTemplate],
    capability: CapabilityConfig,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    errors: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []
    caps = _capability_sets(capability)

    for template in templates:
        prefix = "action_templates.%s" % template.template_id
        for stage in template.stages:
            stage_path = prefix + ".stages.%s" % str(stage.get("id", ""))
            for check in list(stage.get("validation_checks", []) or []):
                if not isinstance(check, dict):
                    continue
                operator = str(check.get("operator", "") or "").strip()
                if operator and caps["custom_ops"] and operator not in caps["custom_ops"]:
                    errors.append({
                        "path": stage_path + ".validation_checks",
                        "message": "operator '%s' is not supported by current capabilities" % operator,
                    })
                for state in _state_names_from_node(check):
                    if caps["states"] and state not in caps["states"]:
                        errors.append({
                            "path": stage_path + ".validation_checks",
                            "message": "state '%s' is not supported by current capabilities" % state,
                        })
                relation = str(check.get("relation", "") or "").strip()
                if relation and caps["relations"] and relation not in caps["relations"]:
                    errors.append({
                        "path": stage_path + ".validation_checks",
                        "message": "relation '%s' is not supported by current capabilities" % relation,
                    })

        for route in list((template.to_dict().get("recovery_defaults", {}) or {}).get("routes", []) or []):
            if not isinstance(route, dict):
                continue
            if not str(route.get("strategy", "") or "").strip():
                warnings.append({
                    "path": prefix + ".recovery_defaults.routes",
                    "message": "route '%s' has no strategy" % str(route.get("id", "")),
                })

    return errors, warnings


def _validate_task_plan_step_validation_against_capabilities(
    plan: Dict[str, Any],
    capability: CapabilityConfig,
) -> List[Dict[str, str]]:
    errors: List[Dict[str, str]] = []
    caps = _capability_sets(capability)
    for step_index, step in enumerate(list(plan.get("execution_steps", []) or [])):
        if not isinstance(step, dict):
            continue
        validation = step.get("validation", {}) or {}
        if not isinstance(validation, dict):
            continue
        step_path = "execution_steps[%d].validation" % step_index
        for group_name in ("overrides", "custom_checks"):
            for idx, rule in enumerate(list(validation.get(group_name, []) or [])):
                if not isinstance(rule, dict):
                    continue
                rule_path = "%s.%s[%d]" % (step_path, group_name, idx)
                operation = str(rule.get("operation", "append_custom_check" if group_name == "custom_checks" else "") or "")
                if operation and caps["operations"] and operation not in caps["operations"]:
                    errors.append({
                        "path": rule_path + ".operation",
                        "message": "operation '%s' is not in the supported operation list" % operation,
                    })
                for node in _node_specs(rule):
                    for state in _state_names_from_node(node):
                        if caps["states"] and state not in caps["states"]:
                            errors.append({
                                "path": rule_path + ".nodes",
                                "message": "node state '%s' is not supported by current capabilities" % state,
                            })
                for rel in _relation_specs(rule):
                    relation = str(rel.get("relation", "") or "").strip()
                    if relation and caps["relations"] and relation not in caps["relations"]:
                        errors.append({
                            "path": rule_path + ".relations",
                            "message": "relation '%s' is not supported by current capabilities" % relation,
                        })
                for operator in _condition_operator_keys(rule.get("when")):
                    if caps["custom_ops"] and operator not in caps["custom_ops"]:
                        errors.append({
                            "path": rule_path + ".when",
                            "message": "condition operator '%s' is not supported in constrained validation mode" % operator,
                        })
    return errors


def _load_plan(
    profile: ProfileConfig,
    *,
    task_plan_ref: Optional[str] = None,
    task_plan_root: Optional[str] = None,
    plan_file: Optional[str] = None,
) -> Dict[str, Any]:
    planning = _profile_planning(profile)
    chosen_ref = str(task_plan_ref or planning.get("task_plan_ref") or "").strip()
    if chosen_ref:
        return load_task_plan(chosen_ref, task_plan_root=task_plan_root)
    chosen_file = str(plan_file or "").strip()
    if chosen_file:
        return build_internal_plan_from_file(chosen_file)
    return build_internal_plan_from_file(get_default_action_plan_file())


def _template_for_step(
    step: Dict[str, Any],
    profile: ProfileConfig,
    plan: Dict[str, Any],
    alias_map: Dict[str, ActionTemplate],
) -> Optional[ActionTemplate]:
    action_type = str(step.get("action_type", "") or "").strip()
    profile_bindings = dict(_profile_planning(profile).get("action_template_bindings", {}) or {})
    plan_bindings = dict(plan.get("action_template_bindings", {}) or {})

    template_ref = (
        profile_bindings.get(action_type)
        or plan_bindings.get(action_type)
        or action_type
    )
    return alias_map.get(str(template_ref or "").strip())


def _expanded_step_preview(
    profile: ProfileConfig,
    plan: Dict[str, Any],
    alias_map: Dict[str, ActionTemplate],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    warnings: List[Dict[str, str]] = []
    strict_templates = bool(_profile_planning(profile).get("strict_action_templates", False))
    preview = []
    for step in list(plan.get("execution_steps", []) or []):
        action_type = str(step.get("action_type", "") or "").strip()
        template = _template_for_step(step, profile, plan, alias_map)
        if template is None:
            message = "action_type '%s' has no configured action template; legacy stage generation will be used" % action_type
            item = {"path": "execution_steps[%s].action_type" % str(step.get("step")), "message": message}
            warnings.append(item)
            if strict_templates:
                warnings[-1]["message"] = message + " (strict_action_templates=true)"
            preview.append({
                "step": step.get("step"),
                "target_object_id": step.get("target_object_id"),
                "action_type": action_type,
                "template_id": "",
                "stages": [],
            })
            continue

        stage_names = build_process_stage_names_for_execution_step(step)
        stages = []
        for stage in template.stages:
            stage_id = str(stage.get("id", "") or "").strip()
            stages.append({
                "stage_id": stage_id,
                "stage_name": stage_names.get(stage_id, "step_%s__%s" % (step.get("step"), stage_id)),
                "phases": list(stage.get("phases", []) or []),
                "validation_check_count": len(list(stage.get("validation_checks", []) or [])),
            })
        preview.append({
            "step": step.get("step"),
            "target_object_id": step.get("target_object_id"),
            "target_class": step.get("target_class"),
            "expected_support_object_id": step.get("expected_support_object_id"),
            "action_type": action_type,
            "template_id": template.template_id,
            "stages": stages,
        })
    return preview, warnings


def _registry_summary(registry: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    phase_count = 0
    expected_node_count = 0
    expected_relation_count = 0
    forbidden_relation_count = 0
    applied_overrides = set()
    custom_check_count = 0
    for bundle in registry.values():
        for state in list((bundle or {}).values()):
            if state is None:
                continue
            phase_count += 1
            expected_node_count += len(list(getattr(state, "expected_nodes", []) or []))
            expected_relation_count += len(list(getattr(state, "expected_relations", []) or []))
            forbidden_relation_count += len(list(getattr(state, "forbidden_relations", []) or []))
            metadata = dict(getattr(state, "metadata", {}) or {})
            profile_meta = dict(metadata.get("validation_profile", {}) or {})
            for item in list(profile_meta.get("applied_override_ids", []) or []):
                if str(item):
                    applied_overrides.add(str(item))
            custom_check_count += len(list(profile_meta.get("custom_checks", []) or []))
    return {
        "stage_count": len(registry),
        "phase_count": phase_count,
        "expected_node_count": expected_node_count,
        "expected_relation_count": expected_relation_count,
        "forbidden_relation_count": forbidden_relation_count,
        "applied_override_ids": sorted(applied_overrides),
        "custom_check_count": custom_check_count,
        "stage_names": sorted(registry.keys())[:20],
    }


def compile_profile_config(
    profile: ProfileConfig,
    *,
    task_plan_ref: Optional[str] = None,
    task_plan_root: Optional[str] = None,
    plan_file: Optional[str] = None,
    capability_ref: Optional[str] = None,
    capability_root: Optional[str] = None,
    action_template_root: Optional[str] = None,
) -> Dict[str, Any]:
    errors: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []

    profile_errors = validate_profile(profile)
    errors.extend({"path": err.path, "message": err.message} for err in profile_errors)

    planning = _profile_planning(profile)
    cap_ref = str(capability_ref or planning.get("capability_ref") or "scene_graph_capabilities").strip()
    capability = load_capability(cap_ref, capability_root=capability_root)
    templates = load_all_action_templates(action_template_root or get_default_action_template_root())
    alias_map = build_action_template_alias_map(templates)

    cap_errors, cap_warnings = _validate_profile_against_capabilities(profile, capability)
    tmpl_errors, tmpl_warnings = _validate_templates_against_capabilities(templates, capability)
    errors.extend(cap_errors)
    errors.extend(tmpl_errors)
    warnings.extend(cap_warnings)
    warnings.extend(tmpl_warnings)

    plan = _load_plan(
        profile,
        task_plan_ref=task_plan_ref,
        task_plan_root=task_plan_root,
        plan_file=plan_file,
    )
    errors.extend(_validate_task_plan_step_validation_against_capabilities(plan, capability))
    expanded_steps, step_warnings = _expanded_step_preview(profile, plan, alias_map)
    warnings.extend(step_warnings)

    registry_summary: Dict[str, Any] = {}
    try:
        registry = build_configured_process_supervision_registry(plan, profile=profile)
        registry_summary = _registry_summary(registry)
    except Exception as exc:
        errors.append({
            "path": "compiled.validation_registry",
            "message": str(exc),
        })

    recovery = profile.recovery
    routing = profile.routing
    result = {
        "ok": len(errors) == 0,
        "profile_id": profile.profile_id,
        "profile_sources": list(profile.source_files or []),
        "capability": {
            "ref": cap_ref,
            "capability_id": capability.capability_id,
            "source_file": capability.source_file,
            "validation": capability.validation,
            "comparison": capability.comparison,
            "recovery": capability.recovery,
        },
        "action_templates": [
            {
                "template_id": template.template_id,
                "aliases": template.aliases,
                "stage_count": len(template.stages),
                "source_file": template.source_file,
            }
            for template in templates
        ],
        "plan_summary": summarize_internal_plan(plan),
        "expanded_steps": expanded_steps,
        "validation_registry": registry_summary,
        "recovery_summary": {
            "action_schemas": sorted(dict(recovery.get("action_schemas", {}) or {}).keys()),
            "strategies": [
                str(item.get("id", "") or "")
                for item in list(recovery.get("strategies", []) or [])
                if isinstance(item, dict)
            ],
            "routing_rules": [
                str(item.get("id", "") or "")
                for item in list(routing.get("rules", []) or [])
                if isinstance(item, dict)
            ],
        },
        "errors": errors,
        "warnings": warnings,
    }
    return result


def compile_configuration(
    profile_name: str = "block_building",
    *,
    profile_root: Optional[str] = None,
    task_plan_ref: Optional[str] = None,
    task_plan_root: Optional[str] = None,
    plan_file: Optional[str] = None,
    capability_ref: Optional[str] = None,
    capability_root: Optional[str] = None,
    action_template_root: Optional[str] = None,
) -> Dict[str, Any]:
    profile = load_profile(profile_name, profile_root=profile_root)
    return compile_profile_config(
        profile,
        task_plan_ref=task_plan_ref,
        task_plan_root=task_plan_root,
        plan_file=plan_file,
        capability_ref=capability_ref,
        capability_root=capability_root,
        action_template_root=action_template_root,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile a profile, task plan, and action templates into a dry-run report.")
    parser.add_argument("--profile", default="block_building")
    parser.add_argument("--profile-root", default=None)
    parser.add_argument("--task-plan", default=None)
    parser.add_argument("--task-plan-root", default=None)
    parser.add_argument("--plan-file", default=None)
    parser.add_argument("--capability", default=None)
    parser.add_argument("--capability-root", default=None)
    parser.add_argument("--action-template-root", default=None)
    args = parser.parse_args()

    report = compile_configuration(
        profile_name=args.profile,
        profile_root=args.profile_root,
        task_plan_ref=args.task_plan,
        task_plan_root=args.task_plan_root,
        plan_file=args.plan_file,
        capability_ref=args.capability,
        capability_root=args.capability_root,
        action_template_root=args.action_template_root,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
