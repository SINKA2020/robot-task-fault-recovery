#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from scene_graph_system.resources import package_resource_path

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

from scene_graph_system.resources import find_package_resource, get_package_root


@dataclass
class CapabilityConfig:
    data: Dict[str, Any]
    source_file: str = ""
    capability_root: str = ""
    errors: List[str] = field(default_factory=list)

    @property
    def capability_id(self) -> str:
        return str(self.data.get("capability_id", "") or "")

    @property
    def validation(self) -> Dict[str, Any]:
        return dict(self.data.get("validation", {}) or {})

    @property
    def comparison(self) -> Dict[str, Any]:
        return dict(self.data.get("comparison", {}) or {})

    @property
    def diagnosis(self) -> Dict[str, Any]:
        return dict(self.data.get("diagnosis", {}) or {})

    @property
    def recovery(self) -> Dict[str, Any]:
        return dict(self.data.get("recovery", {}) or {})

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
        raise ValueError("capability ref is required")
    if ref.startswith("/") or ".." in ref.split("/"):
        raise ValueError("invalid capability ref: %s" % ref)
    return ref


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("capability YAML must contain a mapping: %s" % path)
    return data


def get_default_capability_root() -> str:
    resource = find_package_resource(os.path.join("config", "capabilities"))
    if resource:
        return _abs(resource)

    package_root = get_package_root()
    if package_root:
        return _abs(os.path.join(package_root, "config", "capabilities"))

    return _abs(package_resource_path("config/capabilities"))


def _candidate_paths(ref: str, capability_root: str) -> List[str]:
    ref = str(ref or "").strip()
    if not ref:
        return []
    candidates = []
    if os.path.isabs(ref) or os.path.exists(ref):
        candidates.append(ref)
    base_ref = ref if ref.endswith((".yaml", ".yml")) else ref + ".yaml"
    candidates.extend([
        os.path.join(capability_root, ref),
        os.path.join(capability_root, base_ref),
    ])
    return [_abs(path) for path in candidates]


def resolve_capability_path(ref: str, capability_root: Optional[str] = None) -> str:
    root = _abs(capability_root or get_default_capability_root())
    for path in _candidate_paths(ref, root):
        if os.path.isfile(path):
            if not (os.path.isabs(ref) or os.path.exists(ref)) and not _is_within(path, root):
                continue
            return path
    raise FileNotFoundError("capability not found: %s under %s" % (ref, root))


def validate_capability_data(data: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not str(data.get("capability_id", "") or "").strip():
        errors.append("capability_id is required")

    validation = data.get("validation", {}) or {}
    if not isinstance(validation, dict):
        errors.append("validation must be a mapping")
        validation = {}

    for path in (
        "phases",
        "operations",
        "node_states",
        "relations",
        "custom_check_operators",
        "issue_types",
    ):
        value = validation.get(path, [])
        if value is not None and not isinstance(value, list):
            errors.append("validation.%s must be a list" % path)

    comparison = data.get("comparison", {}) or {}
    if comparison and not isinstance(comparison, dict):
        errors.append("comparison must be a mapping")

    recovery = data.get("recovery", {}) or {}
    if recovery and not isinstance(recovery, dict):
        errors.append("recovery must be a mapping")

    return errors


def load_capability(
    ref: str = "scene_graph_capabilities",
    capability_root: Optional[str] = None,
) -> CapabilityConfig:
    root = _abs(capability_root or get_default_capability_root())
    path = resolve_capability_path(_safe_ref(ref), capability_root=root)
    data = _read_yaml(path)
    errors = validate_capability_data(data)
    if errors:
        raise ValueError("invalid capability %s: %s" % (ref, "; ".join(errors)))
    return CapabilityConfig(data=data, source_file=path, capability_root=root)


def list_capabilities(capability_root: Optional[str] = None) -> List[Dict[str, Any]]:
    root = _abs(capability_root or get_default_capability_root())
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
                errors = validate_capability_data(data)
            except Exception as exc:
                data = {}
                errors = [str(exc)]
            out.append({
                "ref": ref,
                "capability_id": str(data.get("capability_id", "") or ref),
                "description": str(data.get("description", "") or ""),
                "path": path,
                "ok": len(errors) == 0,
                "errors": errors,
            })
    return sorted(out, key=lambda item: item["ref"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Load a capability registry.")
    parser.add_argument("capability", nargs="?", default="scene_graph_capabilities")
    parser.add_argument("--capability-root", default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        payload = list_capabilities(args.capability_root)
    else:
        payload = load_capability(args.capability, args.capability_root).to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
