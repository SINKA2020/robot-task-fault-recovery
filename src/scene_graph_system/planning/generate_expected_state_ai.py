#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from scene_graph_system.resources import package_resource_path

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


from scene_graph_system.planning.expected_state import ExpectedNode, ExpectedRelation, ExpectedState, make_on_top_expected_state
from scene_graph_system.robot.gripper_stage_expectations import get_expected_gripper_states
from scene_graph_system.scene_graph.object_id_utils import canonical_object_id, object_class_from_id, normalize_object_class_name


# -------------------------
# helpers
# -------------------------

def _severity_rank(severity):
    return {"warning": 1, "error": 2, "critical": 3}.get(str(severity), 2)


def _merge_severity(a, b):
    return a if _severity_rank(a) >= _severity_rank(b) else b


def _lookup_gripper_states(full_stage_name):
    """
    从公共规则表查询夹爪预期状态。

    full_stage_name 格式: "approach_source__pre"
    解析为 base_stage="approach_source", phase="pre" 后查表。
    查不到返回 None。
    """
    if "__" in full_stage_name:
        base, phase = full_stage_name.rsplit("__", 1)
        required = get_expected_gripper_states(base, phase)
        if required is not None:
            return list(required)
    return None


def _dedup_expected_nodes(nodes):
    merged = {}
    order = []

    for node in list(nodes or []):
        node_id = canonical_object_id(getattr(node, "node_id", ""))
        if not node_id:
            continue

        if node_id not in merged:
            order.append(node_id)
            merged[node_id] = ExpectedNode(
                node_id=node_id,
                class_name=getattr(node, "class_name", None),
                required_states=list(getattr(node, "required_states", []) or []),
                required_states_any=list(getattr(node, "required_states_any", []) or []),
                optional_states=list(getattr(node, "optional_states", []) or []),
                required=bool(getattr(node, "required", True)),
                severity=getattr(node, "severity", "error"),
                note=getattr(node, "note", ""),
            )
            continue

        old = merged[node_id]
        required_states = list(old.required_states)
        for state in list(getattr(node, "required_states", []) or []):
            if state not in required_states:
                required_states.append(state)

        required_states_any = list(getattr(old, "required_states_any", []) or [])
        for state in list(getattr(node, "required_states_any", []) or []):
            if state not in required_states_any:
                required_states_any.append(state)

        optional_states = list(old.optional_states)
        for state in list(getattr(node, "optional_states", []) or []):
            if state not in optional_states:
                optional_states.append(state)

        old_note = str(getattr(old, "note", "") or "")
        new_note = str(getattr(node, "note", "") or "")
        if new_note and new_note not in old_note:
            note = (old_note + "; " + new_note).strip("; ")
        else:
            note = old_note

        merged[node_id] = ExpectedNode(
            node_id=node_id,
            class_name=old.class_name or getattr(node, "class_name", None),
            required_states=required_states,
            required_states_any=required_states_any,
            optional_states=optional_states,
            required=bool(old.required or getattr(node, "required", True)),
            severity=_merge_severity(old.severity, getattr(node, "severity", "error")),
            note=note,
        )

    return [merged[node_id] for node_id in order]


def _dedup_expected_relations(relations):
    merged = {}
    order = []

    for rel in list(relations or []):
        subject_id = canonical_object_id(getattr(rel, "subject_id", ""))
        object_id = canonical_object_id(getattr(rel, "object_id", ""))
        relation = str(getattr(rel, "relation", "")).strip()
        if not subject_id or not object_id or not relation:
            continue

        key = (subject_id, object_id, relation)
        if key not in merged:
            order.append(key)
            merged[key] = ExpectedRelation(
                subject_id=subject_id,
                object_id=object_id,
                relation=relation,
                required=bool(getattr(rel, "required", True)),
                severity=getattr(rel, "severity", "error"),
                note=getattr(rel, "note", ""),
            )
            continue

        old = merged[key]
        old_note = str(getattr(old, "note", "") or "")
        new_note = str(getattr(rel, "note", "") or "")
        if new_note and new_note not in old_note:
            note = (old_note + "; " + new_note).strip("; ")
        else:
            note = old_note

        merged[key] = ExpectedRelation(
            subject_id=subject_id,
            object_id=object_id,
            relation=relation,
            required=bool(old.required or getattr(rel, "required", True)),
            severity=_merge_severity(old.severity, getattr(rel, "severity", "error")),
            note=note,
        )

    return [merged[key] for key in order]


# -------------------------
# persistent structure constraints
# -------------------------

def _merge_persistent_structure_constraints(
    expected_state,
    persistent_nodes=None,
    persistent_relations=None,
    persistent_forbidden_relations=None,
):
    if expected_state is None:
        return None

    persistent_nodes = list(persistent_nodes or [])
    persistent_relations = list(persistent_relations or [])
    persistent_forbidden_relations = list(persistent_forbidden_relations or [])

    if not persistent_nodes and not persistent_relations and not persistent_forbidden_relations:
        return expected_state

    metadata = dict(getattr(expected_state, "metadata", {}) or {})
    metadata["persistent_structure_enabled"] = True
    metadata["persistent_node_count"] = len(persistent_nodes)
    metadata["persistent_relation_count"] = len(persistent_relations)
    metadata["persistent_forbidden_relation_count"] = len(persistent_forbidden_relations)

    description = getattr(expected_state, "description", "") or ""
    suffix = " 持续约束：前序已完成的搭建结构应在当前阶段保持不变。"
    if suffix.strip() not in description:
        description = (description + suffix).strip()

    return ExpectedState(
        stage_name=expected_state.stage_name,
        expected_nodes=_dedup_expected_nodes(
            list(getattr(expected_state, "expected_nodes", []) or []) + persistent_nodes
        ),
        expected_relations=_dedup_expected_relations(
            list(getattr(expected_state, "expected_relations", []) or []) + persistent_relations
        ),
        forbidden_relations=_dedup_expected_relations(
            list(getattr(expected_state, "forbidden_relations", []) or [])
            + persistent_forbidden_relations
        ),
        description=description,
        metadata=metadata,
    )


def _merge_persistent_structure_into_bundle(
    bundle,
    persistent_nodes=None,
    persistent_relations=None,
    persistent_forbidden_relations=None,
):
    if not persistent_nodes and not persistent_relations and not persistent_forbidden_relations:
        return bundle

    merged_bundle = {}

    for stage_name, state_bundle in bundle.items():
        merged_state_bundle = {}

        for kind, expected_state in state_bundle.items():
            merged_state_bundle[kind] = _merge_persistent_structure_constraints(
                expected_state,
                persistent_nodes=persistent_nodes,
                persistent_relations=persistent_relations,
                persistent_forbidden_relations=persistent_forbidden_relations,
            )

        merged_bundle[stage_name] = merged_state_bundle

    return merged_bundle


def _collect_persistent_structure_constraints_before_step(execution_steps, current_index, resolved_lookup=None):
    resolved_lookup = dict(resolved_lookup or {})

    persistent_nodes = []
    persistent_relations = []
    persistent_forbidden_relations = []

    for prev_index, prev_step in enumerate(list(execution_steps or [])[:current_index]):
        prev_target_object_id = resolved_lookup.get(prev_index) or prev_step.get("target_object_id")
        if not prev_target_object_id:
            continue

        upper_id = canonical_object_id(prev_target_object_id)

        upper_class = normalize_object_class_name(
            prev_step.get("target_class")
            or prev_step.get("source_class")
            or object_class_from_id(upper_id)
        )

        persistent_nodes.append(
            ExpectedNode(
                node_id=upper_id,
                class_name=upper_class,
                required=True,
                severity="critical",
                note="前序已完成搭建结构中的目标节点，应在后续步骤中持续存在。",
            )
        )

        support_id = (
            prev_step.get("expected_support_object_id")
            or prev_step.get("target_support_object_id")
            or prev_step.get("on")
        )

        if not support_id:
            continue

        lower_id = canonical_object_id(support_id)

        lower_class = normalize_object_class_name(
            prev_step.get("support_class")
            or prev_step.get("target_support_class")
            or object_class_from_id(lower_id)
        )

        persistent_nodes.append(
            ExpectedNode(
                node_id=lower_id,
                class_name=lower_class,
                required=True,
                severity="critical",
                note="前序已完成搭建结构中的支撑节点，应在后续步骤中持续存在。",
            )
        )

        persistent_relations.append(
            ExpectedRelation(
                subject_id=upper_id,
                object_id=lower_id,
                relation="on",
                required=True,
                severity="critical",
                note="persistent_built_structure: 前序搭建结果应在当前阶段持续保持。",
            )
        )

        persistent_forbidden_relations.append(
            ExpectedRelation(
                subject_id=upper_id,
                object_id=lower_id,
                relation="intersecting",
                required=False,
                severity="critical",
                note="persistent_built_structure: 已完成支撑关系不应退化为 intersecting。",
            )
        )

    return (
        _dedup_expected_nodes(persistent_nodes),
        _dedup_expected_relations(persistent_relations),
        _dedup_expected_relations(persistent_forbidden_relations),
    )


# -------------------------
# process stage names
# -------------------------

def build_process_stage_names_for_execution_step(step_dict):
    task_step = int(step_dict.get("step", 1))
    target_object_id = step_dict.get("target_object_id")
    target_class = normalize_object_class_name(step_dict.get("target_class") or object_class_from_id(target_object_id or "object"))
    target_name = canonical_object_id(target_object_id) if target_object_id else target_class
    prefix = f"step_{task_step}_{target_name}"
    return {
        "approach_source": f"{prefix}__approach_source",
        "descend_to_grasp": f"{prefix}__descend_to_grasp",
        "close_gripper": f"{prefix}__close_gripper",
        "lift_object": f"{prefix}__lift_object",
        "move_to_target_above": f"{prefix}__move_to_target_above",
        "descend_to_place": f"{prefix}__descend_to_place",
        "open_gripper": f"{prefix}__open_gripper",
        "leave_target": f"{prefix}__leave_target",
    }


# -------------------------
# per-stage expected state builders
# -------------------------

def _build_open_precondition_expected_state(stage_name, description, target_object_id=None, target_class=None):
    gripper_states = _lookup_gripper_states(stage_name) or ["open"]
    expected_nodes = [
        ExpectedNode(node_id="gripper", class_name="gripper", required_states=gripper_states, severity="error")
    ]
    if target_object_id:
        expected_nodes.append(ExpectedNode(node_id=target_object_id, class_name=target_class, severity="warning", required=False))
    return ExpectedState(stage_name=stage_name, expected_nodes=expected_nodes, description=description, metadata={"source": "process_supervision_rule", "kind": "pre"})


def _build_open_runtime_expected_state(stage_name, description):
    gripper_states = _lookup_gripper_states(stage_name) or ["open"]
    return ExpectedState(
        stage_name=stage_name,
        expected_nodes=[ExpectedNode(node_id="gripper", class_name="gripper", required_states=gripper_states, severity="warning")],
        description=description,
        metadata={"source": "process_supervision_rule", "kind": "run"},
    )


def _build_runtime_gripper_monitor_expected_state(stage_name, description, target_object_id=None, target_class=None):
    gripper_states = _lookup_gripper_states(stage_name) or ["closed", "holding"]
    expected_nodes = [
        ExpectedNode(node_id="gripper", class_name="gripper", required_states=gripper_states, severity="critical")
    ]
    if target_object_id:
        expected_nodes.append(
            ExpectedNode(
                node_id=target_object_id,
                class_name=target_class,
                required_states=["grasped"],
                required=False,
                severity="warning",
            )
        )
    return ExpectedState(stage_name=stage_name, expected_nodes=expected_nodes, description=description, metadata={"source": "runtime_monitor_rule"})


def _build_runtime_release_monitor_expected_state(stage_name, description):
    gripper_states = _lookup_gripper_states(stage_name) or ["open"]
    return ExpectedState(
        stage_name=stage_name,
        expected_nodes=[ExpectedNode(node_id="gripper", class_name="gripper", required_states=gripper_states, severity="error")],
        description=description,
        metadata={"source": "runtime_monitor_rule"},
    )


def _build_holding_expected_state(stage_name, description, target_object_id=None, target_class=None):
    return _build_runtime_gripper_monitor_expected_state(stage_name, description, target_object_id=target_object_id, target_class=target_class)


def build_expected_state_for_execution_step(step_dict, resolved_object_id=None):
    target_object_id = resolved_object_id or step_dict.get("target_object_id")
    target_class = normalize_object_class_name(step_dict.get("target_class") or object_class_from_id(target_object_id or "object"))
    stage_name = step_dict.get("stage_name") or f"step_{int(step_dict.get('step', 1))}_catch_and_place_{target_class}"
    expected_support_object_id = step_dict.get("expected_support_object_id") or step_dict.get("target_support_object_id")
    expected_nodes = [ExpectedNode(node_id="gripper", class_name="gripper", required_states=["open"], severity="error")]

    if target_object_id:
        normalized_object_id = canonical_object_id(target_object_id)
        target_class = normalize_object_class_name(step_dict.get("target_class") or object_class_from_id(normalized_object_id))
    else:
        normalized_object_id = None

    if expected_support_object_id and normalized_object_id:
        support_id = canonical_object_id(expected_support_object_id)
        return make_on_top_expected_state(
            stage_name=stage_name,
            upper_id=normalized_object_id,
            lower_id=support_id,
            upper_class=target_class,
            lower_class=object_class_from_id(support_id),
            description=step_dict.get("comment", "执行后目标物体应位于期望支撑物体上方，且夹爪打开。"),
            extra_expected_nodes=expected_nodes,
        )

    target_nodes = list(expected_nodes)
    if normalized_object_id:
        target_nodes.insert(0, ExpectedNode(node_id=normalized_object_id, class_name=target_class, required_states=[], severity="error"))

    return ExpectedState(
        stage_name=stage_name,
        expected_nodes=target_nodes,
        description=step_dict.get("comment", "执行后应能重新观察到目标物体，且夹爪处于打开状态。"),
        metadata={"source": "execution_plan_rule", "target_class": target_class, "resolved_object_id": normalized_object_id},
    )


def _build_place_result_expected_state(stage_name, step_dict, resolved_object_id=None):
    step_copy = dict(step_dict)
    step_copy["stage_name"] = stage_name
    return build_expected_state_for_execution_step(step_copy, resolved_object_id=resolved_object_id)


# -------------------------
# process supervision bundle & registry
# -------------------------

def build_process_supervision_bundle_for_execution_step(
    step_dict,
    resolved_object_id=None,
    persistent_nodes=None,
    persistent_relations=None,
    persistent_forbidden_relations=None,
):
    target_object_id = resolved_object_id or step_dict.get("target_object_id")
    normalized_object_id = canonical_object_id(target_object_id) if target_object_id else None
    target_class = normalize_object_class_name(step_dict.get("target_class") or object_class_from_id(normalized_object_id or "object"))
    stage_names = build_process_stage_names_for_execution_step(step_dict)

    bundle = {
        stage_names["approach_source"]: {
            "pre": _build_open_precondition_expected_state(stage_names["approach_source"] + "__pre", "接近抓取位前，夹爪应保持打开。", normalized_object_id, target_class),
            "run": _build_open_runtime_expected_state(stage_names["approach_source"] + "__run", "接近抓取位过程中，夹爪应保持打开。"),
            "post": _build_open_runtime_expected_state(stage_names["approach_source"] + "__post", "到达预抓取位后，夹爪仍应保持打开。"),
        },
        stage_names["descend_to_grasp"]: {
            "pre": _build_open_precondition_expected_state(stage_names["descend_to_grasp"] + "__pre", "下探抓取前，夹爪应保持打开。", normalized_object_id, target_class),
            "run": _build_open_runtime_expected_state(stage_names["descend_to_grasp"] + "__run", "下探抓取过程中，夹爪应保持打开。"),
            "post": _build_open_runtime_expected_state(stage_names["descend_to_grasp"] + "__post", "下探到抓取位后，夹爪仍应保持打开。"),
        },
        stage_names["close_gripper"]: {
            "pre": _build_open_precondition_expected_state(stage_names["close_gripper"] + "__pre", "闭夹爪前，夹爪应保持打开。", normalized_object_id, target_class),
            "run": None,
            "post": _build_holding_expected_state(stage_names["close_gripper"] + "__post", "闭夹爪后应处于 closed/holding。", normalized_object_id, target_class),
        },
        stage_names["lift_object"]: {
            "pre": _build_holding_expected_state(stage_names["lift_object"] + "__pre", "抬升前应已经夹持目标物体。", normalized_object_id, target_class),
            "run": _build_holding_expected_state(stage_names["lift_object"] + "__run", "抬升过程中夹爪应持续保持 closed/holding。", normalized_object_id, target_class),
            "post": _build_holding_expected_state(stage_names["lift_object"] + "__post", "抬升完成后夹爪应持续保持 closed/holding。", normalized_object_id, target_class),
        },
        stage_names["move_to_target_above"]: {
            "pre": _build_holding_expected_state(stage_names["move_to_target_above"] + "__pre", "搬运到目标上方前应已经夹持目标物体。", normalized_object_id, target_class),
            "run": _build_holding_expected_state(stage_names["move_to_target_above"] + "__run", "搬运过程中夹爪应持续保持 closed/holding。", normalized_object_id, target_class),
            "post": _build_holding_expected_state(stage_names["move_to_target_above"] + "__post", "到达目标上方后夹爪应持续保持 closed/holding。", normalized_object_id, target_class),
        },
        stage_names["descend_to_place"]: {
            "pre": _build_holding_expected_state(stage_names["descend_to_place"] + "__pre", "下放前应已经夹持目标物体。", normalized_object_id, target_class),
            "run": _build_holding_expected_state(stage_names["descend_to_place"] + "__run", "下放过程中夹爪应持续保持 closed/holding。", normalized_object_id, target_class),
            "post": _build_holding_expected_state(stage_names["descend_to_place"] + "__post", "到达放置位后夹爪应仍处于闭合夹持状态。", normalized_object_id, target_class),
        },
        stage_names["open_gripper"]: {
            "pre": _build_holding_expected_state(stage_names["open_gripper"] + "__pre", "释放前应仍夹持目标物体。", normalized_object_id, target_class),
            "run": None,
            "post": _build_runtime_release_monitor_expected_state(stage_names["open_gripper"] + "__post", "释放后夹爪应打开。"),
        },
        stage_names["leave_target"]: {
            "pre": None,
            "run": None,
            "post": _build_place_result_expected_state(stage_names["leave_target"] + "__post", step_dict, resolved_object_id=normalized_object_id),
        },
    }

    return _merge_persistent_structure_into_bundle(
        bundle,
        persistent_nodes=persistent_nodes,
        persistent_relations=persistent_relations,
        persistent_forbidden_relations=persistent_forbidden_relations,
    )


def build_process_supervision_registry_from_execution_plan(plan_dict, resolved_object_ids=None):
    resolved_lookup = dict(resolved_object_ids or {})
    process_registry = {}

    execution_steps = list(plan_dict.get("execution_steps", []) or [])

    for index, step_dict in enumerate(execution_steps):
        resolved_object_id = resolved_lookup.get(index) or step_dict.get("target_object_id")

        (
            persistent_nodes,
            persistent_relations,
            persistent_forbidden_relations,
        ) = _collect_persistent_structure_constraints_before_step(
            execution_steps,
            current_index=index,
            resolved_lookup=resolved_lookup,
        )

        process_registry.update(
            build_process_supervision_bundle_for_execution_step(
                step_dict,
                resolved_object_id=resolved_object_id,
                persistent_nodes=persistent_nodes,
                persistent_relations=persistent_relations,
                persistent_forbidden_relations=persistent_forbidden_relations,
            )
        )

    return process_registry


# -------------------------
# debug / main
# -------------------------

def _print_expected_state_detail(expected_state, indent="    "):
    if expected_state is None:
        print(f"{indent}None")
        return

    print(f"{indent}stage_name: {expected_state.stage_name}")

    print(f"{indent}nodes:")
    nodes = list(getattr(expected_state, "expected_nodes", []) or [])
    if not nodes:
        print(f"{indent}  - 无")
    else:
        for node in nodes:
            states = list(getattr(node, "required_states", []) or [])
            if states:
                print(f"{indent}  - {node.node_id}({node.class_name})[{', '.join(states)}]")
            else:
                print(f"{indent}  - {node.node_id}({node.class_name})")

    print(f"{indent}relations:")
    relations = list(getattr(expected_state, "expected_relations", []) or [])
    if not relations:
        print(f"{indent}  - 无")
    else:
        for rel in relations:
            print(f"{indent}  - {rel.subject_id} --{rel.relation}--> {rel.object_id} [{rel.severity}]")

    forbidden = list(getattr(expected_state, "forbidden_relations", []) or [])
    if forbidden:
        print(f"{indent}forbidden_relations:")
        for rel in forbidden:
            print(f"{indent}  - {rel.subject_id} --{rel.relation}--> {rel.object_id} [{rel.severity}]")


def main():
    from scene_graph_system.planning.task_plan_adapter import build_internal_plan_from_file

    plan = build_internal_plan_from_file(package_resource_path("action.py"))

    print("")
    print("execution_steps:")
    for step in plan.get("execution_steps", []) or []:
        print(
            f"  step={step.get('step')} "
            f"target={step.get('target_object_id')}({step.get('target_class')}) "
            f"target_place={step.get('target_place')} "
            f"support={step.get('expected_support_object_id')}"
        )

    registry = build_process_supervision_registry_from_execution_plan(plan)

    print("")
    print("注册的过程监督阶段详情：")

    for stage_name, bundle in registry.items():
        print("")
        print(f"- {stage_name}")

        for kind in ("pre", "run", "post"):
            print(f"  {kind}:")
            _print_expected_state_detail(bundle.get(kind), indent="    ")


if __name__ == "__main__":
    main()
