#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from scene_graph_system.resources import package_resource_path

import argparse
import importlib.util
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional

import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

from scene_graph_system.resources import find_package_resource, get_package_root


def _abs(path: str) -> str:
    return os.path.abspath(path)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _float_list(value: Any) -> List[float]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for item in value:
        try:
            out.append(float(item))
        except Exception:
            return []
    return out


def _read_yaml_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("task plan YAML must contain a mapping: %s" % path)
    return data


def _is_within(path: str, root: str) -> bool:
    path_abs = _abs(path)
    root_abs = _abs(root)
    return path_abs == root_abs or path_abs.startswith(root_abs + os.sep)


def _safe_ref(ref: str) -> str:
    ref = str(ref or "").strip().replace("\\", "/")
    ref = ref[:-5] if ref.endswith(".yaml") else ref
    ref = ref[:-4] if ref.endswith(".yml") else ref
    ref = ref.strip("/")
    if not ref:
        raise ValueError("task plan ref is required")
    if ref.startswith("/") or ".." in ref.split("/"):
        raise ValueError("invalid task plan ref: %s" % ref)
    return ref


def get_default_task_plan_root() -> str:
    resource = find_package_resource(os.path.join("config", "task_plans"))
    if resource:
        return _abs(resource)

    package_root = get_package_root()
    if package_root:
        return _abs(os.path.join(package_root, "config", "task_plans"))

    return _abs(package_resource_path("config/task_plans"))


def get_default_action_plan_file() -> str:
    resource = find_package_resource(os.path.join("scripts", "action.py"))
    if resource and os.path.isfile(resource):
        return _abs(resource)

    package_root = get_package_root()
    if package_root:
        for rel in (
            os.path.join("resources", "action.py"),
            os.path.join("scripts", "action.py"),
        ):
            candidate = _abs(os.path.join(package_root, rel))
            if os.path.isfile(candidate):
                return candidate

    return _abs(package_resource_path("action.py"))


def _candidate_task_plan_paths(ref: str, task_plan_root: str) -> List[str]:
    ref = str(ref or "").strip()
    if not ref:
        return []
    candidates = []
    if os.path.isabs(ref) or os.path.exists(ref):
        candidates.append(ref)
    base_ref = ref if ref.endswith((".yaml", ".yml")) else ref + ".yaml"
    candidates.extend([
        os.path.join(task_plan_root, ref),
        os.path.join(task_plan_root, base_ref),
    ])
    return [_abs(path) for path in candidates]


def resolve_task_plan_path(ref: str, task_plan_root: Optional[str] = None) -> str:
    root = _abs(task_plan_root or get_default_task_plan_root())
    for path in _candidate_task_plan_paths(ref, root):
        if os.path.isfile(path):
            if not (os.path.isabs(ref) or os.path.exists(ref)) and not _is_within(path, root):
                continue
            return path
    raise FileNotFoundError("task plan not found: %s under %s" % (ref, root))


def _load_python_plan_file(plan_file: str) -> Dict[str, Any]:
    spec = importlib.util.spec_from_file_location("generated_action_module_for_task_plan_loader", plan_file)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load action plan module: %s" % plan_file)

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    closed_loop_plan = getattr(mod, "CLOSED_LOOP_PLAN", None)
    if isinstance(closed_loop_plan, dict):
        return dict(closed_loop_plan)

    out: Dict[str, Any] = {}
    for attr in ("TASK_GOAL", "OBJECT_SPECS", "ACTION_PLAN", "task_queue"):
        if hasattr(mod, attr):
            out[attr] = getattr(mod, attr)

    for func_name in ("get_closed_loop_plan", "get_action_plan", "get_task_queue"):
        func = getattr(mod, func_name, None)
        if not callable(func):
            continue
        result = func()
        if func_name == "get_closed_loop_plan" and isinstance(result, dict):
            return dict(result)
        if func_name == "get_action_plan" and isinstance(result, list):
            out["ACTION_PLAN"] = result
        if func_name == "get_task_queue" and isinstance(result, list):
            out["task_queue"] = result

    return out


def _object_class_from_id(object_id: Any, object_specs: Dict[str, Any]) -> str:
    object_id = str(object_id or "").strip()
    if object_id and object_id in object_specs:
        return str(object_specs[object_id] or "").strip()
    if "_" in object_id:
        return object_id.rsplit("_", 1)[0]
    return object_id or "object"


def _normalize_step(
    raw: Dict[str, Any],
    *,
    index: int,
    object_specs: Dict[str, Any],
    default_pose_tolerance: float,
) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None

    step_num = _safe_int(raw.get("step", raw.get("id", index + 1)), index + 1)
    if step_num <= 0:
        return None

    target_object_id = (
        raw.get("target_object_id")
        or raw.get("object_id")
        or raw.get("target")
        or raw.get("name")
    )

    target_class = (
        raw.get("target_class")
        or raw.get("source_class")
        or raw.get("class_name")
        or raw.get("object_class")
        or raw.get("category")
        or _object_class_from_id(target_object_id, object_specs)
    )

    target_place = (
        raw.get("target_place")
        or raw.get("place")
        or raw.get("target_pose")
        or raw.get("position")
        or []
    )

    expected_support_object_id = (
        raw.get("expected_support_object_id")
        or raw.get("target_support_object_id")
        or raw.get("support_object_id")
        or raw.get("on")
    )

    action_type = str(raw.get("action_type") or raw.get("template") or "catch_and_place").strip()

    normalized = {
        "step": step_num,
        "action_type": action_type,
        "target_object_id": target_object_id,
        "target_class": str(target_class or "").strip() or "object",
        "source_class": str(raw.get("source_class") or target_class or "").strip() or "object",
        "class_name": str(raw.get("class_name") or target_class or "").strip() or "object",
        "target_place": _float_list(target_place) or target_place,
        "source_pose": _float_list(raw.get("source_pose")) or raw.get("source_pose"),
        "expected_support_object_id": expected_support_object_id,
        "support_class": raw.get("support_class"),
        "pose_tolerance": float(raw.get("pose_tolerance", default_pose_tolerance)),
        "stage_name": raw.get("stage_name"),
        "comment": str(raw.get("comment", "") or ""),
    }

    for optional_key in (
        "validation",
        "recovery",
        "routing",
        "expected_relations",
        "expected_nodes",
        "constraints",
    ):
        if optional_key in raw:
            normalized[optional_key] = raw.get(optional_key)

    return normalized


def _extract_steps(data: Dict[str, Any]) -> List[Any]:
    for key in ("execution_steps", "steps", "ACTION_PLAN", "task_queue"):
        value = data.get(key)
        if isinstance(value, list) and value:
            return value
    return []


def build_internal_plan(data: Any, default_pose_tolerance: float = 0.05) -> Dict[str, Any]:
    if isinstance(data, list):
        data = {"execution_steps": data}
    if not isinstance(data, dict):
        raise ValueError("plan data must be a mapping or a list of steps")

    object_specs = dict(data.get("object_specs") or data.get("OBJECT_SPECS") or {})
    scene_object_specs = dict(
        data.get("scene_object_specs")
        or data.get("SCENE_OBJECT_SPECS")
        or object_specs
        or {}
    )
    raw_steps = _extract_steps(data)
    steps = []
    for index, raw in enumerate(raw_steps):
        step = _normalize_step(
            raw,
            index=index,
            object_specs=object_specs,
            default_pose_tolerance=default_pose_tolerance,
        )
        if step is not None:
            steps.append(step)

    steps = sorted(steps, key=lambda item: int(item.get("step", 0)))
    if not steps:
        raise ValueError("no executable steps found in task plan")

    metadata = dict(data.get("plan_metadata", {}) or {})
    if data.get("task_plan_id") is not None:
        metadata["task_plan_id"] = str(data.get("task_plan_id"))
    if data.get("source") is not None:
        metadata["source"] = str(data.get("source"))

    return {
        "task_goal": str(data.get("task_goal") or data.get("TASK_GOAL") or ""),
        "object_specs": object_specs,
        "scene_object_specs": scene_object_specs,
        "execution_steps": steps,
        "plan_metadata": metadata,
        "action_template_bindings": dict(data.get("action_template_bindings", {}) or {}),
    }


def build_internal_plan_from_file(plan_file: str, default_pose_tolerance: float = 0.05) -> Dict[str, Any]:
    plan_file = _abs(str(plan_file or "").strip())
    if not os.path.isfile(plan_file):
        raise FileNotFoundError("action plan file not found: %s" % plan_file)

    if plan_file.endswith((".yaml", ".yml")):
        data = _read_yaml_file(plan_file)
        source_kind = "task_plan_yaml"
    elif plan_file.endswith(".py"):
        data = _load_python_plan_file(plan_file)
        source_kind = "python_action_plan"
    else:
        raise ValueError("unsupported plan file type: %s" % plan_file)

    plan = build_internal_plan(data, default_pose_tolerance=default_pose_tolerance)
    plan["plan_file"] = plan_file
    plan["source"] = source_kind
    plan.setdefault("plan_metadata", {})["plan_file"] = plan_file
    plan.setdefault("plan_metadata", {})["source_kind"] = source_kind
    return plan


def load_task_plan(
    ref: str,
    task_plan_root: Optional[str] = None,
    default_pose_tolerance: float = 0.05,
) -> Dict[str, Any]:
    path = resolve_task_plan_path(_safe_ref(ref), task_plan_root=task_plan_root)
    plan = build_internal_plan_from_file(path, default_pose_tolerance=default_pose_tolerance)
    plan.setdefault("plan_metadata", {})["task_plan_ref"] = _safe_ref(ref)
    return plan


def summarize_internal_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    steps = list(plan.get("execution_steps", []) or [])
    action_types = sorted({str(step.get("action_type", "") or "") for step in steps if step.get("action_type")})
    target_classes = sorted({str(step.get("target_class", "") or "") for step in steps if step.get("target_class")})
    support_count = sum(1 for step in steps if step.get("expected_support_object_id"))
    return {
        "task_goal": str(plan.get("task_goal", "") or ""),
        "step_count": len(steps),
        "action_types": action_types,
        "target_classes": target_classes,
        "support_relation_steps": support_count,
        "source": str(plan.get("source", "") or ""),
        "plan_file": str(plan.get("plan_file", "") or ""),
        "plan_metadata": dict(plan.get("plan_metadata", {}) or {}),
        "object_specs": dict(plan.get("object_specs", {}) or {}),
        "scene_object_specs": dict(plan.get("scene_object_specs", {}) or {}),
        "steps": [
            {
                "step": step.get("step"),
                "action_type": step.get("action_type"),
                "target_object_id": step.get("target_object_id"),
                "target_class": step.get("target_class"),
                "expected_support_object_id": step.get("expected_support_object_id"),
            }
            for step in steps
        ],
    }


def list_task_plans(task_plan_root: Optional[str] = None) -> List[Dict[str, Any]]:
    root = _abs(task_plan_root or get_default_task_plan_root())
    if not os.path.isdir(root):
        return []
    out = []
    for walk_root, _, files in os.walk(root):
        for filename in files:
            if not filename.endswith((".yaml", ".yml")):
                continue
            path = _abs(os.path.join(walk_root, filename))
            if not _is_within(path, root):
                continue
            rel = os.path.relpath(path, root).replace("\\", "/")
            ref = rel.rsplit(".", 1)[0]
            try:
                raw = _read_yaml_file(path)
                summary = summarize_internal_plan(build_internal_plan(raw))
            except Exception as exc:
                raw = {}
                summary = {"error": str(exc), "step_count": 0}
            out.append({
                "ref": ref,
                "task_plan_id": str(raw.get("task_plan_id", "") or ref),
                "description": str(raw.get("description", "") or ""),
                "path": path,
                "summary": summary,
            })
    return sorted(out, key=lambda item: item["ref"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Load and print a normalized task plan.")
    parser.add_argument("plan", nargs="?", default="")
    parser.add_argument("--task-plan-root", default=None)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    if not args.plan:
        plans = list_task_plans(args.task_plan_root)
        print(json.dumps(plans, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if os.path.isfile(args.plan):
        plan = build_internal_plan_from_file(args.plan)
    else:
        plan = load_task_plan(args.plan, task_plan_root=args.task_plan_root)
    payload = summarize_internal_plan(plan) if args.summary else plan
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
