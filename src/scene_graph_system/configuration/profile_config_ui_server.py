#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from scene_graph_system.resources import package_resource_path

import argparse
import copy
import json
import mimetypes
import os
import posixpath
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

from scene_graph_system.resources import get_package_root
from scene_graph_system.configuration.action_template_loader import (
    get_default_action_template_root,
    list_action_templates,
    load_action_template,
)
from scene_graph_system.configuration.capability_loader import (
    get_default_capability_root,
    list_capabilities,
    load_capability,
)
from scene_graph_system.configuration.config_compiler import compile_profile_config
from scene_graph_system.configuration.profile_validator import validate_profile
from scene_graph_system.configuration.rule_config_loader import (
    ProfileConfig,
    get_default_profile_root,
    load_profile,
    merge_profile_dicts,
    resolve_profile_path,
)
from scene_graph_system.planning.task_plan_loader import (
    build_internal_plan,
    get_default_task_plan_root,
    list_task_plans,
    load_task_plan,
    resolve_task_plan_path,
    summarize_internal_plan,
)


class ProfileSaveConflict(RuntimeError):
    pass


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
        raise ValueError("profile ref is required")
    if ref.startswith("/") or ".." in ref.split("/"):
        raise ValueError("invalid profile ref: %s" % ref)
    return ref


def _safe_config_ref(ref: str, label: str = "config") -> str:
    ref = str(ref or "").strip().replace("\\", "/")
    ref = ref[:-5] if ref.endswith(".yaml") else ref
    ref = ref[:-4] if ref.endswith(".yml") else ref
    ref = ref.strip("/")
    if not ref:
        raise ValueError("%s ref is required" % label)
    if ref.startswith("/") or ".." in ref.split("/"):
        raise ValueError("invalid %s ref: %s" % (label, ref))
    return ref


def _profile_path_for_write(ref: str, profile_root: str) -> str:
    safe_ref = _safe_ref(ref)
    path = _abs(os.path.join(profile_root, safe_ref + ".yaml"))
    if not _is_within(path, profile_root):
        raise ValueError("profile path escapes profile root")
    return path


def _task_plan_path_for_write(ref: str, task_plan_root: str) -> str:
    safe_ref = _safe_config_ref(ref, "task plan")
    path = _abs(os.path.join(task_plan_root, safe_ref + ".yaml"))
    if not _is_within(path, task_plan_root):
        raise ValueError("task plan path escapes task plan root")
    return path


def profile_file_version(ref: str, profile_root: str) -> Dict[str, Any]:
    path = resolve_profile_path(_safe_ref(ref), profile_root=profile_root)
    if not _is_within(path, profile_root):
        raise ValueError("profile path escapes profile root")
    stat = os.stat(path)
    return {
        "path": path,
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1000000000))),
        "size": int(stat.st_size),
    }


def task_plan_file_version(ref: str, task_plan_root: str) -> Dict[str, Any]:
    path = resolve_task_plan_path(_safe_config_ref(ref, "task plan"), task_plan_root=task_plan_root)
    if not _is_within(path, task_plan_root):
        raise ValueError("task plan path escapes task plan root")
    stat = os.stat(path)
    return {
        "path": path,
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1000000000))),
        "size": int(stat.st_size),
    }


def _expected_mtime_from_body(body: Dict[str, Any]) -> Optional[int]:
    expected = body.get("expected_file_version", None)
    if isinstance(expected, dict):
        expected = expected.get("mtime_ns")
    if expected in (None, ""):
        return None
    try:
        return int(expected)
    except Exception:
        raise ValueError("expected_file_version.mtime_ns must be an integer")


def _find_static_root(explicit: Optional[str] = None) -> str:
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.append(package_resource_path("web/profile_editor"))
    package_root = get_package_root()
    if package_root:
        candidates.append(os.path.join(package_root, "web", "profile_editor"))
    for candidate in candidates:
        candidate = _abs(candidate)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError("profile editor static directory not found")


def _read_yaml_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping")
    return data


def _load_raw_yaml(ref: str, profile_root: str) -> str:
    path = resolve_profile_path(_safe_ref(ref), profile_root=profile_root)
    if not _is_within(path, profile_root):
        raise ValueError("profile path escapes profile root")
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _load_raw_profile_data(ref: str, profile_root: str) -> Dict[str, Any]:
    path = resolve_profile_path(_safe_ref(ref), profile_root=profile_root)
    if not _is_within(path, profile_root):
        raise ValueError("profile path escapes profile root")
    return _read_yaml_file(path)


def _load_raw_task_plan_yaml(ref: str, task_plan_root: str) -> str:
    path = resolve_task_plan_path(_safe_config_ref(ref, "task plan"), task_plan_root=task_plan_root)
    if not _is_within(path, task_plan_root):
        raise ValueError("task plan path escapes task plan root")
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def merge_profile_yaml(yaml_text: str, profile_root: str) -> ProfileConfig:
    raw = yaml.safe_load(str(yaml_text or "")) or {}
    if not isinstance(raw, dict):
        raise ValueError("profile YAML must contain a mapping")
    return merge_profile_data(raw, profile_root)


def merge_profile_data(raw_data: Dict[str, Any], profile_root: str) -> ProfileConfig:
    if not isinstance(raw_data, dict):
        raise ValueError("profile draft must contain a mapping")

    merged: Dict[str, Any] = {}
    raw = copy.deepcopy(raw_data)
    inherits = raw.get("inherits", []) or []
    if isinstance(inherits, str):
        inherits = [inherits]
    if not isinstance(inherits, list):
        raise ValueError("inherits must be a string or list")

    source_files: List[str] = []
    for parent_ref in inherits:
        parent = load_profile(str(parent_ref), profile_root=profile_root)
        merged = merge_profile_dicts(merged, parent.to_dict())
        source_files.extend(list(parent.source_files or []))

    merged = merge_profile_dicts(merged, raw)
    return ProfileConfig(data=merged, source_files=source_files + ["<editor>"], profile_root=profile_root)


def render_profile_yaml(raw_data: Dict[str, Any]) -> str:
    if not isinstance(raw_data, dict):
        raise ValueError("profile draft must contain a mapping")
    return yaml.safe_dump(copy.deepcopy(raw_data), allow_unicode=True, sort_keys=False)


def parse_profile_yaml(yaml_text: str) -> Dict[str, Any]:
    raw = yaml.safe_load(str(yaml_text or "")) or {}
    if not isinstance(raw, dict):
        raise ValueError("profile YAML must contain a mapping")
    return raw


def parse_task_plan_yaml(yaml_text: str) -> Dict[str, Any]:
    raw = yaml.safe_load(str(yaml_text or "")) or {}
    if not isinstance(raw, dict):
        raise ValueError("task plan YAML must contain a mapping")
    return raw


def render_task_plan_yaml(raw_data: Dict[str, Any]) -> str:
    if not isinstance(raw_data, dict):
        raise ValueError("task plan draft must contain a mapping")
    return yaml.safe_dump(copy.deepcopy(raw_data), allow_unicode=True, sort_keys=False)


def validate_task_plan_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        plan = build_internal_plan(raw_data)
        summary = summarize_internal_plan(plan)
        return {
            "ok": True,
            "errors": [],
            "plan": plan,
            "summary": summary,
            "yaml": render_task_plan_yaml(raw_data),
        }
    except Exception as exc:
        return {
            "ok": False,
            "errors": [{"path": "task_plan", "message": str(exc)}],
            "summary": {"step_count": 0, "error": str(exc)},
            "yaml": render_task_plan_yaml(raw_data) if isinstance(raw_data, dict) else "",
        }


def validate_task_plan_yaml(yaml_text: str) -> Dict[str, Any]:
    raw = parse_task_plan_yaml(yaml_text)
    return validate_task_plan_data(raw)


def validate_profile_yaml(yaml_text: str, profile_root: str) -> Dict[str, Any]:
    profile = merge_profile_yaml(yaml_text, profile_root)
    return validate_profile_config(profile)


def validate_profile_config(profile: ProfileConfig) -> Dict[str, Any]:
    errors = validate_profile(profile)
    return {
        "ok": len(errors) == 0,
        "profile_id": profile.profile_id,
        "errors": [{"path": err.path, "message": err.message} for err in errors],
        "summary": summarize_profile(profile.to_dict()),
        "merged": profile.to_dict(),
    }


def validate_profile_data(raw_data: Dict[str, Any], profile_root: str) -> Dict[str, Any]:
    profile = merge_profile_data(raw_data, profile_root)
    result = validate_profile_config(profile)
    result["yaml"] = render_profile_yaml(raw_data)
    return result


def compile_profile_yaml(
    yaml_text: str,
    profile_root: str,
    *,
    task_plan_ref: Optional[str] = None,
    task_plan_root: Optional[str] = None,
    capability_ref: Optional[str] = None,
    capability_root: Optional[str] = None,
    action_template_root: Optional[str] = None,
) -> Dict[str, Any]:
    profile = merge_profile_yaml(yaml_text, profile_root)
    return compile_profile_config(
        profile,
        task_plan_ref=task_plan_ref,
        task_plan_root=task_plan_root,
        capability_ref=capability_ref,
        capability_root=capability_root,
        action_template_root=action_template_root,
    )


def compile_profile_data(
    raw_data: Dict[str, Any],
    profile_root: str,
    *,
    task_plan_ref: Optional[str] = None,
    task_plan_root: Optional[str] = None,
    capability_ref: Optional[str] = None,
    capability_root: Optional[str] = None,
    action_template_root: Optional[str] = None,
) -> Dict[str, Any]:
    profile = merge_profile_data(raw_data, profile_root)
    return compile_profile_config(
        profile,
        task_plan_ref=task_plan_ref,
        task_plan_root=task_plan_root,
        capability_ref=capability_ref,
        capability_root=capability_root,
        action_template_root=action_template_root,
    )


def compile_reference_error(exc: Exception) -> Dict[str, Any]:
    return {
        "ok": False,
        "errors": [
            {
                "path": "compiled.configuration",
                "message": str(exc),
            }
        ],
        "warnings": [],
    }


def summarize_profile(data: Dict[str, Any]) -> Dict[str, Any]:
    planning = data.get("planning", {}) or {}
    validation = data.get("validation", {}) or {}
    diagnosis = data.get("diagnosis", {}) or {}
    recovery = data.get("recovery", {}) or {}
    routing = data.get("routing", {}) or {}
    return {
        "profile_id": str(data.get("profile_id", "") or ""),
        "task_plan_ref": str(planning.get("task_plan_ref", "") or ""),
        "capability_ref": str(planning.get("capability_ref", "") or ""),
        "action_template_bindings": len(dict(planning.get("action_template_bindings", {}) or {})),
        "fault_types": len(list(data.get("fault_types", []) or [])),
        "validation_overrides": len(list(validation.get("overrides", []) or [])),
        "validation_custom_checks": len(list(validation.get("custom_checks", []) or [])),
        "diagnosis_rules": len(list(diagnosis.get("rules", []) or [])),
        "action_schemas": len(dict(recovery.get("action_schemas", {}) or {})),
        "recovery_strategies": len(list(recovery.get("strategies", []) or [])),
        "routing_rules": len(list(routing.get("rules", []) or [])),
    }


def list_profiles(profile_root: str) -> List[Dict[str, Any]]:
    out = []
    for root, _, files in os.walk(profile_root):
        for filename in files:
            if not filename.endswith((".yaml", ".yml")):
                continue
            path = _abs(os.path.join(root, filename))
            if not _is_within(path, profile_root):
                continue
            rel = os.path.relpath(path, profile_root).replace("\\", "/")
            ref = rel.rsplit(".", 1)[0]
            try:
                raw = _read_yaml_file(path)
            except Exception:
                raw = {}
            out.append({
                "ref": ref,
                "profile_id": str(raw.get("profile_id", "") or ref),
                "description": str(raw.get("description", "") or ""),
                "path": path,
                "is_default": ref.startswith("defaults/"),
            })
    return sorted(out, key=lambda item: (bool(item["is_default"]), item["ref"]))


def write_profile_yaml(
    ref: str,
    yaml_text: str,
    profile_root: str,
    *,
    create_backup: bool = True,
    expected_mtime_ns: Optional[int] = None,
) -> Dict[str, str]:
    path = _profile_path_for_write(ref, profile_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if expected_mtime_ns is not None and os.path.exists(path):
        stat = os.stat(path)
        current_mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1000000000)))
        if current_mtime_ns != int(expected_mtime_ns):
            raise ProfileSaveConflict(
                "profile file changed on disk after it was loaded; reload before saving: %s" % path
            )
    backup_path = ""
    if os.path.exists(path) and create_backup:
        backup_path = path + ".bak.%s" % time.strftime("%Y%m%d_%H%M%S")
        with open(path, "r", encoding="utf-8") as src, open(backup_path, "w", encoding="utf-8") as dst:
            dst.write(src.read())
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(yaml_text)
        if yaml_text and not yaml_text.endswith("\n"):
            handle.write("\n")
    version = profile_file_version(ref, profile_root)
    return {
        "path": path,
        "backup_path": backup_path,
        "mtime_ns": str(version["mtime_ns"]),
        "size": str(version["size"]),
    }


def write_task_plan_yaml(
    ref: str,
    yaml_text: str,
    task_plan_root: str,
    *,
    create_backup: bool = True,
    expected_mtime_ns: Optional[int] = None,
) -> Dict[str, str]:
    path = _task_plan_path_for_write(ref, task_plan_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if expected_mtime_ns is not None and os.path.exists(path):
        stat = os.stat(path)
        current_mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1000000000)))
        if current_mtime_ns != int(expected_mtime_ns):
            raise ProfileSaveConflict(
                "task plan file changed on disk after it was loaded; reload before saving: %s" % path
            )
    backup_path = ""
    if os.path.exists(path) and create_backup:
        backup_path = path + ".bak.%s" % time.strftime("%Y%m%d_%H%M%S")
        with open(path, "r", encoding="utf-8") as src, open(backup_path, "w", encoding="utf-8") as dst:
            dst.write(src.read())
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(yaml_text)
        if yaml_text and not yaml_text.endswith("\n"):
            handle.write("\n")
    version = task_plan_file_version(ref, task_plan_root)
    return {
        "path": path,
        "backup_path": backup_path,
        "mtime_ns": str(version["mtime_ns"]),
        "size": str(version["size"]),
    }


class ProfileConfigRequestHandler(BaseHTTPRequestHandler):
    server_version = "ProfileConfigUI/1.0"

    def _json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = 400) -> None:
        self._json({"ok": False, "error": str(message)}, status=status)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        data = json.loads(raw or "{}")
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        return data

    def _draft_from_body(self, body: Dict[str, Any]) -> Dict[str, Any]:
        draft = body.get("draft", body.get("profile", body.get("data")))
        if not isinstance(draft, dict):
            raise ValueError("request body must include a draft object")
        return draft

    @property
    def profile_root(self) -> str:
        return self.server.profile_root  # type: ignore[attr-defined]

    @property
    def static_root(self) -> str:
        return self.server.static_root  # type: ignore[attr-defined]

    @property
    def task_plan_root(self) -> str:
        return self.server.task_plan_root  # type: ignore[attr-defined]

    @property
    def action_template_root(self) -> str:
        return self.server.action_template_root  # type: ignore[attr-defined]

    @property
    def capability_root(self) -> str:
        return self.server.capability_root  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            if path == "/api/health":
                self._json({
                    "ok": True,
                    "profile_root": self.profile_root,
                    "task_plan_root": self.task_plan_root,
                    "action_template_root": self.action_template_root,
                    "capability_root": self.capability_root,
                })
                return
            if path == "/api/profiles":
                self._json({"ok": True, "profiles": list_profiles(self.profile_root)})
                return
            if path == "/api/capabilities":
                self._json({"ok": True, "capabilities": list_capabilities(self.capability_root)})
                return
            if path == "/api/capability":
                ref = _safe_config_ref((query.get("ref") or ["scene_graph_capabilities"])[0], "capability")
                capability = load_capability(ref, capability_root=self.capability_root)
                self._json({
                    "ok": True,
                    "ref": ref,
                    "data": capability.to_dict(),
                    "source_file": capability.source_file,
                })
                return
            if path == "/api/action-templates":
                self._json({"ok": True, "templates": list_action_templates(self.action_template_root)})
                return
            if path == "/api/action-template":
                ref = _safe_config_ref((query.get("ref") or ["pick_place"])[0], "action template")
                template = load_action_template(ref, template_root=self.action_template_root)
                self._json({
                    "ok": True,
                    "ref": ref,
                    "data": template.to_dict(),
                    "source_file": template.source_file,
                })
                return
            if path == "/api/task-plans":
                self._json({"ok": True, "task_plans": list_task_plans(self.task_plan_root)})
                return
            if path == "/api/task-plan":
                ref = _safe_config_ref((query.get("ref") or ["block_building_demo"])[0], "task plan")
                plan = load_task_plan(ref, task_plan_root=self.task_plan_root)
                self._json({
                    "ok": True,
                    "ref": ref,
                    "plan": plan,
                    "summary": summarize_internal_plan(plan),
                })
                return
            if path == "/api/task-plan/raw":
                ref = _safe_config_ref((query.get("ref") or ["block_building_demo"])[0], "task plan")
                yaml_text = _load_raw_task_plan_yaml(ref, self.task_plan_root)
                self._json({
                    "ok": True,
                    "ref": ref,
                    "yaml": yaml_text,
                    "file_version": task_plan_file_version(ref, self.task_plan_root),
                })
                return
            if path == "/api/profile":
                ref = (query.get("ref") or [""])[0]
                merged = (query.get("merged") or ["0"])[0] in {"1", "true", "yes"}
                safe_ref = _safe_ref(ref)
                if merged:
                    profile = load_profile(safe_ref, profile_root=self.profile_root)
                    yaml_text = yaml.safe_dump(profile.to_dict(), allow_unicode=True, sort_keys=False)
                    data = profile.to_dict()
                    source_files = list(profile.source_files or [])
                    file_version = profile_file_version(safe_ref, self.profile_root)
                else:
                    yaml_text = _load_raw_yaml(ref, self.profile_root)
                    data = _load_raw_profile_data(ref, self.profile_root)
                    source_files = [resolve_profile_path(safe_ref, profile_root=self.profile_root)]
                    file_version = profile_file_version(safe_ref, self.profile_root)
                self._json({
                    "ok": True,
                    "ref": safe_ref,
                    "merged": merged,
                    "yaml": yaml_text,
                    "data": data,
                    "summary": summarize_profile(data),
                    "source_files": source_files,
                    "file_version": file_version,
                })
                return
            self._serve_static(path)
        except ProfileSaveConflict as exc:
            self._error(str(exc), status=409)
        except ValueError as exc:
            self._error(str(exc), status=400)
        except Exception as exc:
            self._error(str(exc), status=500)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            body = self._read_json_body()
            if parsed.path == "/api/profile-draft/parse-yaml":
                draft = parse_profile_yaml(str(body.get("yaml", "") or ""))
                result = validate_profile_data(draft, self.profile_root)
                self._json({
                    "ok": result["ok"],
                    "draft": draft,
                    "summary": summarize_profile(draft),
                    "validation": result,
                    "errors": result.get("errors", []),
                }, status=200 if result["ok"] else 422)
                return
            if parsed.path == "/api/profile-draft/render-yaml":
                draft = self._draft_from_body(body)
                yaml_text = render_profile_yaml(draft)
                self._json({
                    "ok": True,
                    "yaml": yaml_text,
                    "summary": summarize_profile(draft),
                })
                return
            if parsed.path == "/api/profile-draft/validate":
                draft = self._draft_from_body(body)
                result = validate_profile_data(draft, self.profile_root)
                self._json(result, status=200 if result["ok"] else 422)
                return
            if parsed.path in {"/api/profile-draft/compile", "/api/profile-draft/dry-run"}:
                draft = self._draft_from_body(body)
                task_plan_ref = str(body.get("task_plan_ref", "") or "").strip() or None
                capability_ref = str(body.get("capability_ref", "") or "").strip() or None
                try:
                    result = compile_profile_data(
                        draft,
                        self.profile_root,
                        task_plan_ref=task_plan_ref,
                        task_plan_root=self.task_plan_root,
                        capability_ref=capability_ref,
                        capability_root=self.capability_root,
                        action_template_root=self.action_template_root,
                    )
                except (FileNotFoundError, ValueError) as exc:
                    result = compile_reference_error(exc)
                result["yaml"] = render_profile_yaml(draft)
                self._json(result, status=200 if result["ok"] else 422)
                return
            if parsed.path == "/api/profile-draft/save":
                ref = _safe_ref(str(body.get("ref", "") or ""))
                draft = self._draft_from_body(body)
                validation = validate_profile_data(draft, self.profile_root)
                if not validation["ok"]:
                    self._json(validation, status=422)
                    return

                compile_result: Optional[Dict[str, Any]] = None
                if bool(body.get("require_compile", True)):
                    task_plan_ref = str(body.get("task_plan_ref", "") or "").strip() or None
                    capability_ref = str(body.get("capability_ref", "") or "").strip() or None
                    try:
                        compile_result = compile_profile_data(
                            draft,
                            self.profile_root,
                            task_plan_ref=task_plan_ref,
                            task_plan_root=self.task_plan_root,
                            capability_ref=capability_ref,
                            capability_root=self.capability_root,
                            action_template_root=self.action_template_root,
                        )
                    except (FileNotFoundError, ValueError) as exc:
                        compile_result = compile_reference_error(exc)
                    if not compile_result["ok"]:
                        self._json({
                            "ok": False,
                            "validation": validation,
                            "compile": compile_result,
                            "errors": compile_result.get("errors", []),
                        }, status=422)
                        return

                yaml_text = render_profile_yaml(draft)
                write_info = write_profile_yaml(
                    ref,
                    yaml_text,
                    self.profile_root,
                    create_backup=bool(body.get("create_backup", True)),
                    expected_mtime_ns=_expected_mtime_from_body(body),
                )
                self._json({
                    "ok": True,
                    "ref": ref,
                    "path": write_info["path"],
                    "backup_path": write_info["backup_path"],
                    "file_version": {
                        "path": write_info["path"],
                        "mtime_ns": int(write_info["mtime_ns"]),
                        "size": int(write_info["size"]),
                    },
                    "yaml": yaml_text,
                    "validation": validation,
                    "compile": compile_result,
                })
                return
            if parsed.path == "/api/task-plan/parse-yaml":
                draft = parse_task_plan_yaml(str(body.get("yaml", "") or ""))
                result = validate_task_plan_data(draft)
                self._json({
                    "ok": result["ok"],
                    "draft": draft,
                    "summary": result.get("summary", {}),
                    "validation": result,
                    "errors": result.get("errors", []),
                }, status=200 if result["ok"] else 422)
                return
            if parsed.path == "/api/task-plan/render-yaml":
                draft = body.get("draft", body.get("task_plan", body.get("data")))
                if not isinstance(draft, dict):
                    raise ValueError("request body must include a task plan draft object")
                yaml_text = render_task_plan_yaml(draft)
                self._json({
                    "ok": True,
                    "yaml": yaml_text,
                    "summary": validate_task_plan_data(draft).get("summary", {}),
                })
                return
            if parsed.path == "/api/task-plan/validate":
                result = validate_task_plan_yaml(str(body.get("yaml", "") or ""))
                self._json(result, status=200 if result["ok"] else 422)
                return
            if parsed.path == "/api/task-plan/save":
                ref = _safe_config_ref(str(body.get("ref", "") or ""), "task plan")
                yaml_text = str(body.get("yaml", "") or "")
                result = validate_task_plan_yaml(yaml_text)
                if not result["ok"]:
                    self._json(result, status=422)
                    return
                write_info = write_task_plan_yaml(
                    ref,
                    yaml_text,
                    self.task_plan_root,
                    create_backup=bool(body.get("create_backup", True)),
                    expected_mtime_ns=_expected_mtime_from_body(body),
                )
                self._json({
                    "ok": True,
                    "ref": ref,
                    "path": write_info["path"],
                    "backup_path": write_info["backup_path"],
                    "file_version": {
                        "path": write_info["path"],
                        "mtime_ns": int(write_info["mtime_ns"]),
                        "size": int(write_info["size"]),
                    },
                    "yaml": render_task_plan_yaml(parse_task_plan_yaml(yaml_text)),
                    "validation": result,
                })
                return
            if parsed.path == "/api/validate":
                result = validate_profile_yaml(str(body.get("yaml", "") or ""), self.profile_root)
                self._json(result, status=200 if result["ok"] else 422)
                return
            if parsed.path in {"/api/compile", "/api/dry-run"}:
                task_plan_ref = str(body.get("task_plan_ref", "") or "").strip() or None
                capability_ref = str(body.get("capability_ref", "") or "").strip() or None
                try:
                    result = compile_profile_yaml(
                        str(body.get("yaml", "") or ""),
                        self.profile_root,
                        task_plan_ref=task_plan_ref,
                        task_plan_root=self.task_plan_root,
                        capability_ref=capability_ref,
                        capability_root=self.capability_root,
                        action_template_root=self.action_template_root,
                    )
                except (FileNotFoundError, ValueError) as exc:
                    result = compile_reference_error(exc)
                self._json(result, status=200 if result["ok"] else 422)
                return
            if parsed.path == "/api/save":
                ref = _safe_ref(str(body.get("ref", "") or ""))
                yaml_text = str(body.get("yaml", "") or "")
                result = validate_profile_yaml(yaml_text, self.profile_root)
                if not result["ok"]:
                    self._json(result, status=422)
                    return

                compile_result: Optional[Dict[str, Any]] = None
                if bool(body.get("require_compile", True)):
                    task_plan_ref = str(body.get("task_plan_ref", "") or "").strip() or None
                    capability_ref = str(body.get("capability_ref", "") or "").strip() or None
                    try:
                        compile_result = compile_profile_yaml(
                            yaml_text,
                            self.profile_root,
                            task_plan_ref=task_plan_ref,
                            task_plan_root=self.task_plan_root,
                            capability_ref=capability_ref,
                            capability_root=self.capability_root,
                            action_template_root=self.action_template_root,
                        )
                    except (FileNotFoundError, ValueError) as exc:
                        compile_result = compile_reference_error(exc)
                    if not compile_result["ok"]:
                        self._json({
                            "ok": False,
                            "validation": result,
                            "compile": compile_result,
                            "errors": compile_result.get("errors", []),
                        }, status=422)
                        return

                write_info = write_profile_yaml(
                    ref,
                    yaml_text,
                    self.profile_root,
                    create_backup=bool(body.get("create_backup", True)),
                    expected_mtime_ns=_expected_mtime_from_body(body),
                )
                self._json({
                    "ok": True,
                    "ref": ref,
                    "path": write_info["path"],
                    "backup_path": write_info["backup_path"],
                    "file_version": {
                        "path": write_info["path"],
                        "mtime_ns": int(write_info["mtime_ns"]),
                        "size": int(write_info["size"]),
                    },
                    "validation": result,
                    "compile": compile_result,
                })
                return
            self._error("unknown endpoint", status=404)
        except ProfileSaveConflict as exc:
            self._error(str(exc), status=409)
        except ValueError as exc:
            self._error(str(exc), status=400)
        except Exception as exc:
            self._error(str(exc), status=500)

    def _serve_static(self, request_path: str) -> None:
        if request_path in {"", "/"}:
            request_path = "/index.html"
        normalized = posixpath.normpath(unquote(request_path)).lstrip("/")
        if normalized.startswith("../"):
            self._error("invalid static path", status=400)
            return
        path = _abs(os.path.join(self.static_root, normalized))
        if not _is_within(path, self.static_root) or not os.path.isfile(path):
            self._error("not found", status=404)
            return
        mime, _ = mimetypes.guess_type(path)
        with open(path, "rb") as handle:
            body = handle.read()
        self.send_response(200)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[profile-ui] " + (fmt % args) + "\n")


def run_server(
    host: str,
    port: int,
    profile_root: str,
    static_root: Optional[str] = None,
    task_plan_root: Optional[str] = None,
    action_template_root: Optional[str] = None,
    capability_root: Optional[str] = None,
) -> None:
    profile_root = _abs(profile_root or get_default_profile_root())
    static_root = _find_static_root(static_root)
    task_plan_root = _abs(task_plan_root or get_default_task_plan_root())
    action_template_root = _abs(action_template_root or get_default_action_template_root())
    capability_root = _abs(capability_root or get_default_capability_root())
    server = ThreadingHTTPServer((host, int(port)), ProfileConfigRequestHandler)
    server.profile_root = profile_root  # type: ignore[attr-defined]
    server.static_root = static_root  # type: ignore[attr-defined]
    server.task_plan_root = task_plan_root  # type: ignore[attr-defined]
    server.action_template_root = action_template_root  # type: ignore[attr-defined]
    server.capability_root = capability_root  # type: ignore[attr-defined]
    print("Profile configuration UI: http://%s:%d" % (host, int(port)))
    print("profile_root=%s" % profile_root)
    print("task_plan_root=%s" % task_plan_root)
    print("action_template_root=%s" % action_template_root)
    print("capability_root=%s" % capability_root)
    print("static_root=%s" % static_root)
    server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the local task-profile configuration UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--profile-root", default=get_default_profile_root())
    parser.add_argument("--static-root", default=None)
    parser.add_argument("--task-plan-root", default=get_default_task_plan_root())
    parser.add_argument("--action-template-root", default=get_default_action_template_root())
    parser.add_argument("--capability-root", default=get_default_capability_root())
    args = parser.parse_args()
    run_server(
        args.host,
        args.port,
        args.profile_root,
        args.static_root,
        args.task_plan_root,
        args.action_template_root,
        args.capability_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
