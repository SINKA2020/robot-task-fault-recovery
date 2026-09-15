#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from scene_graph_system.resources import package_resource_path

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

from scene_graph_system.resources import find_package_resource, get_package_root


@dataclass
class ActionTemplate:
    data: Dict[str, Any]
    source_file: str = ""
    template_root: str = ""

    @property
    def template_id(self) -> str:
        return str(self.data.get("template_id", "") or "")

    @property
    def aliases(self) -> List[str]:
        return [str(item) for item in list(self.data.get("action_type_aliases", []) or []) if str(item).strip()]

    @property
    def stages(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in list(self.data.get("stages", []) or []) if isinstance(item, dict)]

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.data)


def _abs(path: str) -> str:
    return os.path.abspath(path)


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
        raise ValueError("action template ref is required")
    if ref.startswith("/") or ".." in ref.split("/"):
        raise ValueError("invalid action template ref: %s" % ref)
    return ref


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("action template YAML must contain a mapping: %s" % path)
    return data


def get_default_action_template_root() -> str:
    resource = find_package_resource(os.path.join("config", "action_templates"))
    if resource:
        return _abs(resource)

    package_root = get_package_root()
    if package_root:
        return _abs(os.path.join(package_root, "config", "action_templates"))

    return _abs(package_resource_path("config/action_templates"))


def _candidate_paths(ref: str, template_root: str) -> List[str]:
    ref = str(ref or "").strip()
    if not ref:
        return []
    candidates = []
    if os.path.isabs(ref) or os.path.exists(ref):
        candidates.append(ref)
    base_ref = ref if ref.endswith((".yaml", ".yml")) else ref + ".yaml"
    candidates.extend([
        os.path.join(template_root, ref),
        os.path.join(template_root, base_ref),
    ])
    return [_abs(path) for path in candidates]


def resolve_action_template_path(ref: str, template_root: Optional[str] = None) -> str:
    root = _abs(template_root or get_default_action_template_root())
    for path in _candidate_paths(ref, root):
        if os.path.isfile(path):
            if not (os.path.isabs(ref) or os.path.exists(ref)) and not _is_within(path, root):
                continue
            return path
    raise FileNotFoundError("action template not found: %s under %s" % (ref, root))


def validate_action_template_data(data: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    template_id = str(data.get("template_id", "") or "").strip()
    if not template_id:
        errors.append("template_id is required")

    aliases = data.get("action_type_aliases", []) or []
    if not isinstance(aliases, list):
        errors.append("action_type_aliases must be a list")

    stages = data.get("stages", []) or []
    if not isinstance(stages, list) or not stages:
        errors.append("stages must be a non-empty list")
        return errors

    seen = set()
    for idx, stage in enumerate(stages):
        path = "stages[%d]" % idx
        if not isinstance(stage, dict):
            errors.append("%s must be a mapping" % path)
            continue
        stage_id = str(stage.get("id", "") or "").strip()
        if not stage_id:
            errors.append("%s.id is required" % path)
        elif stage_id in seen:
            errors.append("duplicate stage id: %s" % stage_id)
        seen.add(stage_id)
        phases = stage.get("phases", []) or []
        if not isinstance(phases, list):
            errors.append("%s.phases must be a list" % path)
    return errors


def load_action_template(ref: str, template_root: Optional[str] = None) -> ActionTemplate:
    root = _abs(template_root or get_default_action_template_root())
    path = resolve_action_template_path(_safe_ref(ref), template_root=root)
    data = _read_yaml(path)
    errors = validate_action_template_data(data)
    if errors:
        raise ValueError("invalid action template %s: %s" % (ref, "; ".join(errors)))
    return ActionTemplate(data=data, source_file=path, template_root=root)


def list_action_templates(template_root: Optional[str] = None) -> List[Dict[str, Any]]:
    root = _abs(template_root or get_default_action_template_root())
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
                data = _read_yaml(path)
                errors = validate_action_template_data(data)
            except Exception as exc:
                data = {}
                errors = [str(exc)]
            out.append({
                "ref": ref,
                "template_id": str(data.get("template_id", "") or ref),
                "description": str(data.get("description", "") or ""),
                "aliases": list(data.get("action_type_aliases", []) or []),
                "stage_count": len(list(data.get("stages", []) or [])),
                "path": path,
                "ok": len(errors) == 0,
                "errors": errors,
            })
    return sorted(out, key=lambda item: item["ref"])


def load_all_action_templates(template_root: Optional[str] = None) -> List[ActionTemplate]:
    templates = []
    for item in list_action_templates(template_root):
        if item.get("ok"):
            templates.append(load_action_template(str(item["ref"]), template_root=template_root))
    return templates


def build_action_template_alias_map(templates: List[ActionTemplate]) -> Dict[str, ActionTemplate]:
    out: Dict[str, ActionTemplate] = {}
    for template in templates:
        keys = [template.template_id] + template.aliases
        for key in keys:
            key = str(key or "").strip()
            if key:
                out[key] = template
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Load action templates.")
    parser.add_argument("template", nargs="?", default="")
    parser.add_argument("--template-root", default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list or not args.template:
        payload = list_action_templates(args.template_root)
    else:
        payload = load_action_template(args.template, args.template_root).to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
