#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from scene_graph_system.resources import package_resource_path

import argparse
import copy
import fnmatch
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


from scene_graph_system.planning.expected_state import ExpectedNode, ExpectedRelation, ExpectedState
from scene_graph_system.validation.scene_graph_comparator import ComparisonIssue
from scene_graph_system.configuration.profile_validator import validate_profile
from scene_graph_system.configuration.rule_config_loader import ProfileConfig, load_profile, merge_profile_dicts


PHASES = ("pre", "run", "post")
PROFILE_METADATA_KEY = "validation_profile"


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _string_list(value: Any) -> List[str]:
    return [str(item) for item in _as_list(value) if str(item).strip()]


def _matches_stage(rule: Dict[str, Any], stage_name: str, expected_state: Optional[ExpectedState]) -> bool:
    pattern = str(rule.get("stage_pattern", "*") or "*").strip() or "*"
    names = [str(stage_name or "")]
    if expected_state is not None:
        names.append(str(getattr(expected_state, "stage_name", "") or ""))
    return any(fnmatch.fnmatchcase(name, pattern) for name in names)


def _matches_phase(rule: Dict[str, Any], phase: str) -> bool:
    configured = rule.get("phase", "*")
    if configured in (None, "", "*"):
        return True
    return str(configured).strip() == str(phase).strip()


def _metadata(state: ExpectedState) -> Dict[str, Any]:
    metadata = dict(getattr(state, "metadata", {}) or {})
    profile_meta = dict(metadata.get(PROFILE_METADATA_KEY, {}) or {})
    metadata[PROFILE_METADATA_KEY] = profile_meta
    state.metadata = metadata
    return profile_meta


def _note_rule_applied(state: ExpectedState, rule: Dict[str, Any]) -> None:
    profile_meta = _metadata(state)
    applied = list(profile_meta.get("applied_override_ids", []) or [])
    rule_id = str(rule.get("id", "") or "")
    if rule_id and rule_id not in applied:
        applied.append(rule_id)
    profile_meta["applied_override_ids"] = applied


def _make_node(raw: Dict[str, Any]) -> ExpectedNode:
    return ExpectedNode(
        node_id=str(raw.get("node_id", "") or ""),
        class_name=raw.get("class_name"),
        required_states=_string_list(raw.get("required_states", [])),
        required_states_any=_string_list(
            raw.get("required_states_any", raw.get("states_any", []))
        ),
        optional_states=_string_list(raw.get("optional_states", [])),
        required=bool(raw.get("required", True)),
        severity=str(raw.get("severity", "error") or "error"),
        note=str(raw.get("note", "") or ""),
    )


def _make_relation(raw: Dict[str, Any]) -> ExpectedRelation:
    return ExpectedRelation(
        subject_id=str(raw.get("subject_id", "") or ""),
        object_id=str(raw.get("object_id", "") or ""),
        relation=str(raw.get("relation", "") or ""),
        required=bool(raw.get("required", True)),
        severity=str(raw.get("severity", "error") or "error"),
        note=str(raw.get("note", "") or ""),
    )


def _merge_node(existing: ExpectedNode, incoming: ExpectedNode) -> ExpectedNode:
    required_states = list(getattr(existing, "required_states", []) or [])
    for state in list(getattr(incoming, "required_states", []) or []):
        if state not in required_states:
            required_states.append(state)

    required_states_any = list(getattr(existing, "required_states_any", []) or [])
    for state in list(getattr(incoming, "required_states_any", []) or []):
        if state not in required_states_any:
            required_states_any.append(state)

    optional_states = list(getattr(existing, "optional_states", []) or [])
    for state in list(getattr(incoming, "optional_states", []) or []):
        if state not in optional_states:
            optional_states.append(state)

    note = str(getattr(existing, "note", "") or "")
    incoming_note = str(getattr(incoming, "note", "") or "")
    if incoming_note and incoming_note not in note:
        note = (note + "; " + incoming_note).strip("; ")

    return ExpectedNode(
        node_id=existing.node_id,
        class_name=existing.class_name or incoming.class_name,
        required_states=required_states,
        required_states_any=required_states_any,
        optional_states=optional_states,
        required=bool(existing.required or incoming.required),
        severity=incoming.severity or existing.severity,
        note=note,
    )


def _append_nodes(state: ExpectedState, nodes: Iterable[Any]) -> None:
    by_id = {str(node.node_id): idx for idx, node in enumerate(state.expected_nodes)}
    for raw in list(nodes or []):
        if not isinstance(raw, dict) or not raw.get("node_id"):
            continue
        incoming = _make_node(raw)
        idx = by_id.get(str(incoming.node_id))
        if idx is None:
            by_id[str(incoming.node_id)] = len(state.expected_nodes)
            state.expected_nodes.append(incoming)
        else:
            state.expected_nodes[idx] = _merge_node(state.expected_nodes[idx], incoming)


def _append_relations(target: List[ExpectedRelation], relations: Iterable[Any]) -> None:
    existing = {rel.key() for rel in list(target or [])}
    for raw in list(relations or []):
        if not isinstance(raw, dict):
            continue
        if not raw.get("subject_id") or not raw.get("object_id") or not raw.get("relation"):
            continue
        incoming = _make_relation(raw)
        if incoming.key() in existing:
            continue
        target.append(incoming)
        existing.add(incoming.key())


def _append_expected(state: ExpectedState, rule: Dict[str, Any]) -> None:
    _append_nodes(state, rule.get("nodes", []))
    _append_relations(state.expected_relations, rule.get("relations", []))
    _append_relations(state.forbidden_relations, rule.get("forbidden_relations", []))
    _note_rule_applied(state, rule)


def _remove_nodes(state: ExpectedState, node_ids: Iterable[str]) -> None:
    targets = {str(item) for item in node_ids}
    if not targets:
        return
    state.expected_nodes = [node for node in state.expected_nodes if str(node.node_id) not in targets]


def _remove_relations(target: List[ExpectedRelation], raw_relations: Iterable[Any]) -> List[ExpectedRelation]:
    remove_keys = set()
    for raw in list(raw_relations or []):
        if not isinstance(raw, dict):
            continue
        subject = str(raw.get("subject_id", "") or "")
        obj = str(raw.get("object_id", "") or "")
        relation = str(raw.get("relation", "") or "")
        if subject and obj and relation:
            remove_keys.add((subject, obj, relation))
    if not remove_keys:
        return target
    return [rel for rel in target if rel.key() not in remove_keys]


def _disable_check(bundle: Dict[str, Optional[ExpectedState]], phase: str, rule: Dict[str, Any]) -> None:
    state = bundle.get(phase)
    if state is None:
        return
    if bool(rule.get("disable_phase", False)):
        bundle[phase] = None
        return

    _remove_nodes(state, _string_list(rule.get("node_ids", [])))
    _remove_nodes(state, [str(item.get("node_id")) for item in _as_list(rule.get("nodes", [])) if isinstance(item, dict)])
    state.expected_relations = _remove_relations(state.expected_relations, rule.get("relations", []))
    state.forbidden_relations = _remove_relations(state.forbidden_relations, rule.get("forbidden_relations", []))
    _note_rule_applied(state, rule)


def _relax_expected(state: ExpectedState, rule: Dict[str, Any]) -> None:
    profile_meta = _metadata(state)
    relaxed = list(profile_meta.get("relaxed_state_any", []) or [])

    for raw in list(rule.get("nodes", []) or []):
        if not isinstance(raw, dict):
            continue
        node_id = str(raw.get("node_id", "") or "")
        states_any = _string_list(raw.get("required_states_any", []))
        if not node_id or not states_any:
            continue
        relaxed.append({
            "rule_id": str(rule.get("id", "") or ""),
            "node_id": node_id,
            "required_states_any": states_any,
        })

    profile_meta["relaxed_state_any"] = relaxed
    _note_rule_applied(state, rule)


def _register_custom_check(state: ExpectedState, rule: Dict[str, Any]) -> None:
    profile_meta = _metadata(state)
    custom_checks = list(profile_meta.get("custom_checks", []) or [])
    custom_checks.append(copy.deepcopy(rule))
    profile_meta["custom_checks"] = custom_checks
    _note_rule_applied(state, rule)


def _make_state_from_rule(stage_name: str, rule: Dict[str, Any]) -> ExpectedState:
    raw_state = dict(rule.get("expected_state", {}) or {})
    return ExpectedState(
        stage_name=str(raw_state.get("stage_name") or stage_name),
        expected_nodes=[_make_node(item) for item in list(raw_state.get("nodes", rule.get("nodes", [])) or [])],
        expected_relations=[
            _make_relation(item)
            for item in list(raw_state.get("relations", rule.get("relations", [])) or [])
        ],
        forbidden_relations=[
            _make_relation(item)
            for item in list(raw_state.get("forbidden_relations", rule.get("forbidden_relations", [])) or [])
        ],
        description=str(raw_state.get("description", rule.get("note", "")) or ""),
        metadata={PROFILE_METADATA_KEY: {"applied_override_ids": [str(rule.get("id", "") or "")]}},
    )


def _apply_rule_to_phase(
    bundle: Dict[str, Optional[ExpectedState]],
    stage_name: str,
    phase: str,
    rule: Dict[str, Any],
) -> None:
    operation = str(rule.get("operation", "") or "").strip()
    if operation == "replace_phase":
        bundle[phase] = _make_state_from_rule(stage_name + "__" + phase, rule)
        return

    if operation == "disable_check":
        _disable_check(bundle, phase, rule)
        return

    state = bundle.get(phase)
    if state is None:
        return

    if operation == "append_expected":
        _append_expected(state, rule)
        return

    if operation == "append_custom_check":
        _register_custom_check(state, rule)
        return

    if operation == "relax":
        _relax_expected(state, rule)
        return


def apply_validation_overrides(
    base_registry: Dict[str, Dict[str, Optional[ExpectedState]]],
    validation_config: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, Optional[ExpectedState]]]:
    registry = copy.deepcopy(base_registry)
    validation_config = dict(validation_config or {})
    overrides = list(validation_config.get("overrides", []) or [])
    for custom_check in list(validation_config.get("custom_checks", []) or []):
        if isinstance(custom_check, dict):
            rule = copy.deepcopy(custom_check)
            rule.setdefault("operation", "append_custom_check")
            overrides.append(rule)

    for stage_name, bundle in registry.items():
        if not isinstance(bundle, dict):
            continue
        for rule in overrides:
            if not isinstance(rule, dict):
                continue
            if not bool(rule.get("enabled", True)):
                continue
            for phase in PHASES:
                state = bundle.get(phase)
                if not _matches_phase(rule, phase):
                    continue
                if not _matches_stage(rule, stage_name, state):
                    continue
                _apply_rule_to_phase(bundle, stage_name, phase, rule)
                if bundle.get(phase) is not None:
                    bundle[phase].validate()
    return registry


def _load_action_template_alias_map():
    try:
        from scene_graph_system.configuration.action_template_loader import build_action_template_alias_map, load_all_action_templates

        templates = load_all_action_templates()
        return build_action_template_alias_map(templates)
    except Exception:
        return {}


def _template_for_step(step: Dict[str, Any], profile: ProfileConfig, plan_dict: Dict[str, Any], alias_map: Dict[str, Any]):
    action_type = str(step.get("action_type", "") or "").strip()
    planning = profile.planning
    profile_bindings = dict(planning.get("action_template_bindings", {}) or {})
    plan_bindings = dict(plan_dict.get("action_template_bindings", {}) or {})
    template_ref = profile_bindings.get(action_type) or plan_bindings.get(action_type) or action_type
    return alias_map.get(str(template_ref or "").strip())


def _collect_action_template_validation_config(profile: ProfileConfig, plan_dict: Dict[str, Any]) -> Dict[str, Any]:
    alias_map = _load_action_template_alias_map()
    if not alias_map:
        return {}

    used_templates = {}
    for step in list(plan_dict.get("execution_steps", []) or []):
        if not isinstance(step, dict):
            continue
        template = _template_for_step(step, profile, plan_dict, alias_map)
        if template is not None:
            used_templates[getattr(template, "template_id", "")] = template

    validation_config: Dict[str, Any] = {}
    for template in used_templates.values():
        data = template.to_dict() if hasattr(template, "to_dict") else dict(getattr(template, "data", {}) or {})
        template_validation = {}
        overrides = list(data.get("validation_overrides", []) or [])
        custom_checks = list(data.get("validation_custom_checks", []) or [])
        if overrides:
            template_validation["overrides"] = overrides
        if custom_checks:
            template_validation["custom_checks"] = custom_checks
        if template_validation:
            validation_config = merge_profile_dicts(validation_config, template_validation)
    return validation_config


def _collect_step_validation_config(plan_dict: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from scene_graph_system.planning.generate_expected_state_ai import build_process_stage_names_for_execution_step
    except Exception:
        build_process_stage_names_for_execution_step = None

    validation_config: Dict[str, Any] = {}
    for step in list(plan_dict.get("execution_steps", []) or []):
        if not isinstance(step, dict):
            continue
        validation = step.get("validation", {}) or {}
        if not isinstance(validation, dict):
            continue
        step_overrides = []
        for rule in list(validation.get("overrides", []) or []):
            if not isinstance(rule, dict):
                continue
            rule = copy.deepcopy(rule)
            if not rule.get("stage_pattern") and build_process_stage_names_for_execution_step is not None:
                try:
                    stage_names = build_process_stage_names_for_execution_step(step)
                    prefix = str(stage_names.get("approach_source", "")).rsplit("__", 1)[0]
                    if prefix:
                        rule["stage_pattern"] = prefix + "__*"
                except Exception:
                    pass
            step_overrides.append(rule)
        step_custom_checks = list(validation.get("custom_checks", []) or [])
        step_validation = {}
        if step_overrides:
            step_validation["overrides"] = step_overrides
        if step_custom_checks:
            step_validation["custom_checks"] = step_custom_checks
        if step_validation:
            validation_config = merge_profile_dicts(validation_config, step_validation)
    return validation_config


def _stage_name_for_step_stage(step: Dict[str, Any], stage_id: str) -> str:
    try:
        from scene_graph_system.planning.generate_expected_state_ai import build_process_stage_names_for_execution_step

        stage_names = build_process_stage_names_for_execution_step(step)
        name = str(stage_names.get(stage_id, "") or "")
        if name:
            return name
    except Exception:
        pass
    target = str(step.get("target_object_id") or step.get("target_class") or "target")
    return "step_%s_%s__%s" % (step.get("step"), target, stage_id)


def _matching_steps_for_scope(scope: Dict[str, Any], profile: ProfileConfig, plan_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps = [step for step in list(plan_dict.get("execution_steps", []) or []) if isinstance(step, dict)]
    level = str(scope.get("level", "") or "").strip()
    if level == "step":
        wanted_step = scope.get("step")
        try:
            wanted_step = int(wanted_step)
        except Exception:
            wanted_step = None
        return [step for step in steps if int(step.get("step", -1)) == wanted_step]

    if level == "action":
        action_type = str(scope.get("action_type", "") or "").strip()
        if not action_type:
            return steps
        return [step for step in steps if str(step.get("action_type", "") or "").strip() == action_type]

    return steps


def _expand_scoped_rule(rule: Dict[str, Any], profile: ProfileConfig, plan_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    scope = rule.get("scope")
    if not isinstance(scope, dict):
        return [rule]

    stage_id = str(scope.get("stage_id") or rule.get("stage_id") or "").strip()
    level = str(scope.get("level", "") or "").strip()
    if not stage_id or level not in {"action", "step"}:
        return [rule]

    expanded = []
    for step in _matching_steps_for_scope(scope, profile, plan_dict):
        copy_rule = copy.deepcopy(rule)
        if copy_rule.get("stage_pattern"):
            copy_rule.setdefault("legacy_stage_pattern", copy_rule.get("stage_pattern"))
        copy_rule["stage_pattern"] = _stage_name_for_step_stage(step, stage_id)
        copy_rule.setdefault("level", level)
        if level == "step":
            copy_rule.setdefault("step", step.get("step"))
        expanded.append(copy_rule)
    if expanded:
        return expanded

    fallback = copy.deepcopy(rule)
    if fallback.get("stage_pattern"):
        fallback.setdefault("legacy_stage_pattern", fallback.get("stage_pattern"))
    fallback["stage_pattern"] = "*__%s" % stage_id
    return [fallback]


def _expand_validation_scopes(
    validation_config: Dict[str, Any],
    profile: ProfileConfig,
    plan_dict: Dict[str, Any],
) -> Dict[str, Any]:
    if not plan_dict:
        return validation_config
    out = copy.deepcopy(validation_config)
    for group_name in ("overrides", "custom_checks"):
        expanded: List[Dict[str, Any]] = []
        for rule in list(out.get(group_name, []) or []):
            if isinstance(rule, dict):
                expanded.extend(_expand_scoped_rule(rule, profile, plan_dict))
        if expanded:
            out[group_name] = expanded
    return out


def build_effective_validation_config(profile: ProfileConfig, plan_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Compose validation config in the intended priority order:

    action template defaults < task profile validation < task-plan step overrides.
    """
    plan_dict = dict(plan_dict or {})
    template_validation = _collect_action_template_validation_config(profile, plan_dict)
    effective = merge_profile_dicts(template_validation, profile.validation)
    step_validation = _collect_step_validation_config(plan_dict)
    if step_validation:
        effective = merge_profile_dicts(effective, step_validation)
    return _expand_validation_scopes(effective, profile, plan_dict)


def build_configured_process_supervision_registry(
    plan_dict: Dict[str, Any],
    profile: Optional[ProfileConfig] = None,
    profile_name: str = "block_building",
    profile_root: Optional[str] = None,
) -> Dict[str, Dict[str, Optional[ExpectedState]]]:
    from scene_graph_system.planning.generate_expected_state_ai import build_process_supervision_registry_from_execution_plan

    base_registry = build_process_supervision_registry_from_execution_plan(plan_dict)
    if profile is None:
        profile = load_profile(profile_name, profile_root=profile_root)
        errors = validate_profile(profile)
        if errors:
            raise ValueError("invalid profile %s: %s" % (profile.profile_id, "; ".join(str(e) for e in errors)))
    validation_config = build_effective_validation_config(profile, plan_dict)
    return apply_validation_overrides(base_registry, validation_config)


def apply_validation_overrides_to_state(
    expected_state: ExpectedState,
    validation_config: Optional[Dict[str, Any]],
    phase: str = "post",
    stage_name: Optional[str] = None,
) -> ExpectedState:
    """
    Apply validation Profile overrides to a single legacy ExpectedState.

    This keeps the Phase-2 integration available for the older step-level
    closed-loop runner, while the richer process registry path remains the
    primary supervision interface.
    """
    registry_stage = str(stage_name or getattr(expected_state, "stage_name", "") or "")
    if not registry_stage:
        return expected_state

    phase = str(phase or "post").strip() or "post"
    registry = {registry_stage: {phase: expected_state}}
    configured = apply_validation_overrides(registry, validation_config)
    return configured.get(registry_stage, {}).get(phase) or expected_state


def build_configured_expected_state_for_execution_step(
    step_dict: Dict[str, Any],
    resolved_object_id: Optional[str] = None,
    profile: Optional[ProfileConfig] = None,
    profile_name: str = "block_building",
    profile_root: Optional[str] = None,
    phase: str = "post",
) -> ExpectedState:
    from scene_graph_system.planning.generate_expected_state_ai import build_expected_state_for_execution_step

    base_state = build_expected_state_for_execution_step(step_dict, resolved_object_id=resolved_object_id)
    if profile is None:
        profile = load_profile(profile_name, profile_root=profile_root)
        errors = validate_profile(profile)
        if errors:
            raise ValueError("invalid profile %s: %s" % (profile.profile_id, "; ".join(str(e) for e in errors)))
    validation_config = build_effective_validation_config(profile, {"execution_steps": [step_dict]})
    return apply_validation_overrides_to_state(base_state, validation_config, phase=phase)


def _issue_matches_relaxed_state(issue: Any, entry: Dict[str, Any]) -> bool:
    if str(getattr(issue, "issue_type", "") or "") != "state_mismatch":
        return False
    node_id = str(getattr(issue, "node_id", "") or "").strip()
    if node_id != str(entry.get("node_id", "") or "").strip():
        return False
    observed = set(_string_list(getattr(issue, "observed", []) or []))
    states_any = set(_string_list(entry.get("required_states_any", [])))
    if not observed or not states_any:
        return False
    return bool(observed & states_any)


def _refresh_result_passed(result: Any) -> None:
    result.passed = not any(
        str(getattr(issue, "severity", "") or "") in {"error", "critical"}
        for issue in list(getattr(result, "issues", []) or [])
    )


def _find_graph_node(graph: Any, node_id: str):
    node_id = str(node_id or "")
    if not node_id or graph is None:
        return None
    get_node = getattr(graph, "get_node", None)
    if callable(get_node):
        node = get_node(node_id)
        if node is not None:
            return node
    for node in list(getattr(graph, "nodes", []) or []):
        try:
            if str(node.uid()) == node_id:
                return node
        except Exception:
            pass
    return None


def _graph_relation_exists(graph: Any, spec: Dict[str, Any]) -> bool:
    if graph is None or not isinstance(spec, dict):
        return True
    subject_id = str(spec.get("subject_id", spec.get("subject", "")) or "")
    object_id = str(spec.get("object_id", spec.get("object", "")) or "")
    relation = str(spec.get("relation", "") or "")
    if not subject_id or not object_id or not relation:
        return True

    get_relations = getattr(graph, "get_relations", None)
    relations = get_relations() if callable(get_relations) else []
    for edge in relations or []:
        try:
            if (
                str(edge.start.uid()) == subject_id
                and str(edge.end.uid()) == object_id
                and str(edge.edge_type) == relation
            ):
                return True
        except Exception:
            continue
    return False


def _node_has_state(graph: Any, spec: Dict[str, Any]) -> bool:
    if not isinstance(spec, dict):
        return True
    node_id = str(spec.get("node_id", spec.get("node", "")) or "")
    node = _find_graph_node(graph, node_id)
    if node is None:
        return False
    observed = set(_string_list(getattr(node, "states", []) or []))
    states_any = set(_string_list(spec.get("states_any", spec.get("state_any", []))))
    states_all = set(_string_list(spec.get("states_all", spec.get("state_all", []))))
    if states_any and not (observed & states_any):
        return False
    if states_all and not states_all.issubset(observed):
        return False
    return True


def _custom_condition_passes(condition: Any, graph: Any) -> bool:
    if condition in (None, {}, []):
        return True
    if isinstance(condition, list):
        return all(_custom_condition_passes(item, graph) for item in condition)
    if not isinstance(condition, dict):
        return True
    if "all" in condition:
        return all(_custom_condition_passes(item, graph) for item in _as_list(condition.get("all")))
    if "any" in condition:
        return any(_custom_condition_passes(item, graph) for item in _as_list(condition.get("any")))
    if "not" in condition:
        return not _custom_condition_passes(condition.get("not"), graph)
    if "node_has_state" in condition:
        return _node_has_state(graph, condition.get("node_has_state"))
    if "relation_exists" in condition:
        return _graph_relation_exists(graph, condition.get("relation_exists"))
    return True


def _make_custom_issue(rule: Dict[str, Any]) -> ComparisonIssue:
    issue_cfg = dict(rule.get("issues_on_fail", {}) or {})
    rule_id = str(rule.get("id", "") or "")
    return ComparisonIssue(
        issue_type=str(issue_cfg.get("issue_type", "custom_validation_failed") or "custom_validation_failed"),
        severity=str(issue_cfg.get("severity", "error") or "error"),
        message=str(issue_cfg.get("message", "custom validation failed") or "custom validation failed"),
        node_id=issue_cfg.get("node_id"),
        subject_id=issue_cfg.get("subject_id"),
        object_id=issue_cfg.get("object_id"),
        relation=issue_cfg.get("relation"),
        expected=issue_cfg.get("expected"),
        observed=issue_cfg.get("observed"),
        metadata={
            "source": "configurable_expected_state_builder",
            "custom_check_id": rule_id,
            **dict(issue_cfg.get("metadata", {}) or {}),
        },
    )


def _apply_custom_checks(expected_state: ExpectedState, result: Any, graph: Any) -> None:
    metadata = dict(getattr(expected_state, "metadata", {}) or {})
    profile_meta = dict(metadata.get(PROFILE_METADATA_KEY, {}) or {})
    custom_checks = list(profile_meta.get("custom_checks", []) or [])
    for rule in custom_checks:
        if not isinstance(rule, dict) or not bool(rule.get("enabled", True)):
            continue
        if _custom_condition_passes(rule.get("when"), graph):
            continue
        result.add_issue(_make_custom_issue(rule))


def apply_comparison_result_overrides(expected_state: Optional[ExpectedState], result: Any) -> Any:
    if expected_state is None or result is None:
        return result

    metadata = dict(getattr(expected_state, "metadata", {}) or {})
    profile_meta = dict(metadata.get(PROFILE_METADATA_KEY, {}) or {})
    relaxed = list(profile_meta.get("relaxed_state_any", []) or [])
    if not relaxed:
        return result

    kept = []
    suppressed = []
    for issue in list(getattr(result, "issues", []) or []):
        matched_entry = None
        for entry in relaxed:
            if _issue_matches_relaxed_state(issue, entry):
                matched_entry = entry
                break
        if matched_entry is None:
            kept.append(issue)
        else:
            suppressed.append({
                "issue_type": getattr(issue, "issue_type", ""),
                "node_id": getattr(issue, "node_id", None),
                "missing_state": dict(getattr(issue, "metadata", {}) or {}).get("missing_state"),
                "rule_id": matched_entry.get("rule_id"),
            })

    if suppressed:
        result.issues = kept
        _refresh_result_passed(result)
        profile_meta["suppressed_issues"] = suppressed
        metadata[PROFILE_METADATA_KEY] = profile_meta
        expected_state.metadata = metadata
    return result


def compare_configured_bundle_state(stage_name, kind, bundle, graph, comparator):
    expected_state = bundle.get(kind)
    if expected_state is None:
        return None, None
    result = compare_configured_expected_state(expected_state, graph, comparator)
    return expected_state, result


def compare_configured_expected_state(expected_state, graph, comparator):
    result = comparator.compare(expected_state, graph)
    _apply_custom_checks(expected_state, result, graph)
    result = apply_comparison_result_overrides(expected_state, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a configured supervision registry summary.")
    parser.add_argument("--plan-file", default=package_resource_path("action.py"))
    parser.add_argument("--profile", default="block_building")
    parser.add_argument("--profile-root", default=None)
    args = parser.parse_args()

    from scene_graph_system.planning.task_plan_adapter import build_internal_plan_from_file

    plan = build_internal_plan_from_file(args.plan_file)
    profile = load_profile(args.profile, profile_root=args.profile_root)
    errors = validate_profile(profile)
    if errors:
        for error in errors:
            print(error)
        return 1

    registry = build_configured_process_supervision_registry(
        plan,
        profile=profile,
        profile_name=args.profile,
        profile_root=args.profile_root,
    )
    summary = {
        "profile_id": profile.profile_id,
        "stage_count": len(registry),
        "source_files": list(profile.source_files),
        "stages": sorted(registry.keys())[:10],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
