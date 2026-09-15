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
class ProfileConfig:
    data: Dict[str, Any]
    source_files: List[str] = field(default_factory=list)
    profile_root: str = ""

    @property
    def profile_id(self) -> str:
        return str(self.data.get("profile_id", "") or "")

    @property
    def fault_types(self) -> List[Dict[str, Any]]:
        return list(self.data.get("fault_types", []) or [])

    @property
    def planning(self) -> Dict[str, Any]:
        return dict(self.data.get("planning", {}) or {})

    @property
    def validation(self) -> Dict[str, Any]:
        return dict(self.data.get("validation", {}) or {})

    @property
    def diagnosis(self) -> Dict[str, Any]:
        return dict(self.data.get("diagnosis", {}) or {})

    @property
    def recovery(self) -> Dict[str, Any]:
        return dict(self.data.get("recovery", {}) or {})

    @property
    def routing(self) -> Dict[str, Any]:
        return dict(self.data.get("routing", {}) or {})

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.data)


def get_default_profile_root() -> str:
    resource = find_package_resource(os.path.join("config", "task_profiles"))
    if resource:
        return os.path.abspath(resource)

    package_root = get_package_root()
    if package_root:
        return os.path.abspath(os.path.join(package_root, "config", "task_profiles"))

    return os.path.abspath(package_resource_path("config/task_profiles"))


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("profile file must contain a YAML mapping: %s" % path)
    return data


def _candidate_paths(ref: str, profile_root: str) -> List[str]:
    ref = str(ref or "").strip()
    if not ref:
        return []

    candidates = []
    if os.path.isabs(ref) or os.path.exists(ref):
        candidates.append(ref)

    base_ref = ref if ref.endswith((".yaml", ".yml")) else ref + ".yaml"
    candidates.extend([
        os.path.join(profile_root, ref),
        os.path.join(profile_root, base_ref),
        os.path.join(profile_root, "defaults", ref),
        os.path.join(profile_root, "defaults", base_ref),
    ])
    return [os.path.abspath(path) for path in candidates]


def resolve_profile_path(ref: str, profile_root: Optional[str] = None) -> str:
    root = os.path.abspath(profile_root or get_default_profile_root())
    for path in _candidate_paths(ref, root):
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("profile not found: %s under %s" % (ref, root))


def _list_items_are_id_maps(items: Any) -> bool:
    if not isinstance(items, list) or not items:
        return False
    for item in items:
        if not isinstance(item, dict) or "id" not in item:
            return False
    return True


def _merge_id_lists(base: List[Dict[str, Any]], override: List[Dict[str, Any]], path: str) -> List[Dict[str, Any]]:
    out = [copy.deepcopy(item) for item in base]
    index_by_id = {str(item.get("id")): i for i, item in enumerate(out)}

    for item in override:
        item_id = str(item.get("id"))
        if item_id in index_by_id:
            idx = index_by_id[item_id]
            out[idx] = merge_profile_dicts(out[idx], item, path="%s[%s]" % (path, item_id))
        else:
            index_by_id[item_id] = len(out)
            out.append(copy.deepcopy(item))

    return out


def merge_profile_dicts(base: Any, override: Any, path: str = "") -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        out = copy.deepcopy(base)
        for key, value in override.items():
            child_path = "%s.%s" % (path, key) if path else str(key)
            if key in out:
                out[key] = merge_profile_dicts(out[key], value, path=child_path)
            else:
                out[key] = copy.deepcopy(value)
        return out

    if _list_items_are_id_maps(base) and _list_items_are_id_maps(override):
        return _merge_id_lists(base, override, path)

    return copy.deepcopy(override)


def _load_profile_recursive(
    ref: str,
    profile_root: str,
    stack: List[str],
    source_files: List[str],
) -> Dict[str, Any]:
    path = resolve_profile_path(ref, profile_root)
    if path in stack:
        cycle = " -> ".join(stack + [path])
        raise ValueError("profile inheritance cycle detected: %s" % cycle)

    raw = _read_yaml(path)
    merged: Dict[str, Any] = {}

    inherits = raw.get("inherits", []) or []
    if isinstance(inherits, str):
        inherits = [inherits]
    if not isinstance(inherits, list):
        raise ValueError("inherits must be a string or list in %s" % path)

    for parent_ref in inherits:
        parent = _load_profile_recursive(str(parent_ref), profile_root, stack + [path], source_files)
        merged = merge_profile_dicts(merged, parent)

    merged = merge_profile_dicts(merged, raw)
    if path not in source_files:
        source_files.append(path)
    return merged


def load_profile(ref: str, profile_root: Optional[str] = None) -> ProfileConfig:
    root = os.path.abspath(profile_root or get_default_profile_root())
    source_files: List[str] = []
    data = _load_profile_recursive(ref, root, [], source_files)
    return ProfileConfig(data=data, source_files=source_files, profile_root=root)


def main() -> int:
    parser = argparse.ArgumentParser(description="Load and print a merged task profile.")
    parser.add_argument("profile", nargs="?", default="block_building")
    parser.add_argument("--profile-root", default=None)
    parser.add_argument("--sources", action="store_true", help="print source files instead of merged profile")
    args = parser.parse_args()

    profile = load_profile(args.profile, profile_root=args.profile_root)
    if args.sources:
        for path in profile.source_files:
            print(path)
        return 0

    print(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
