from __future__ import annotations

import re
from typing import Optional


_INSTANCE_PATTERN = re.compile(r"^\s*([A-Za-z][A-Za-z0-9]*?)[\s_\-]*([0-9]+)\s*$")


def normalize_object_class_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def canonical_object_id(name: str, fallback_index: Optional[int] = None) -> str:
    raw = str(name).strip()
    match = _INSTANCE_PATTERN.match(raw)
    if match:
        class_name = normalize_object_class_name(match.group(1))
        instance_index = int(match.group(2))
        return f"{class_name}_{instance_index}"

    class_name = normalize_object_class_name(raw)
    if fallback_index is not None:
        return f"{class_name}_{int(fallback_index)}"
    return class_name


def has_explicit_instance_id(name: str) -> bool:
    return _INSTANCE_PATTERN.match(str(name).strip()) is not None


def object_class_from_id(object_id: str) -> str:
    normalized = normalize_object_class_name(object_id)
    match = re.match(r"^(.*?)(?:_([0-9]+))?$", normalized)
    if match is None:
        return normalized
    class_name = match.group(1) or normalized
    return class_name