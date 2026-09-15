from __future__ import annotations


_RELATION_ALIASES = {
    "on top of": "on",
    "on_top_of": "on",
    "on": "on",
    "supports": "supports",
    "aligned with": "aligned_with",
    "aligned_with": "aligned_with",
    "intersecting": "intersecting",
    "adjacent to": "adjacent_to",
    "adjacent_to": "adjacent_to",
    "inside": "inside",
    "above": "above",
    "below": "below",
    "on the right of": "on_the_right_of",
    "on_the_right_of": "on_the_right_of",
    "on the left of": "on_the_left_of",
    "on_the_left_of": "on_the_left_of",
    "blocking": "blocking",
}


def normalize_relation_name(relation: str) -> str:
    raw = str(relation).strip().lower().replace("-", "_")
    raw = " ".join(raw.split())
    return _RELATION_ALIASES.get(raw, raw.replace(" ", "_"))