from __future__ import annotations

from typing import Dict, FrozenSet, Optional, Set, Tuple


# Required states use all-of semantics.  Carrying stages intentionally require
# only ``holding``: mechanical ``closed`` without holding evidence is an empty
# grasp and must not pass grasp validation.
GRIPPER_EXPECTATIONS: Dict[Tuple[str, str], Optional[FrozenSet[str]]] = {
    ("approach_source", "pre"): frozenset({"open"}),
    ("approach_source", "run"): frozenset({"open"}),
    ("approach_source", "post"): frozenset({"open"}),
    ("descend_to_grasp", "pre"): frozenset({"open"}),
    ("descend_to_grasp", "run"): frozenset({"open"}),
    ("descend_to_grasp", "post"): frozenset({"open"}),
    ("close_gripper", "pre"): frozenset({"open"}),
    # No run check while the jaws are moving.
    ("close_gripper", "post"): frozenset({"holding"}),
    ("lift_object", "pre"): frozenset({"holding"}),
    ("lift_object", "run"): frozenset({"holding"}),
    ("lift_object", "post"): frozenset({"holding"}),
    ("move_to_target_above", "pre"): frozenset({"holding"}),
    ("move_to_target_above", "run"): frozenset({"holding"}),
    ("move_to_target_above", "post"): frozenset({"holding"}),
    ("descend_to_place", "pre"): frozenset({"holding"}),
    ("descend_to_place", "run"): frozenset({"holding"}),
    ("descend_to_place", "post"): frozenset({"holding"}),
    ("open_gripper", "pre"): frozenset({"holding"}),
    # No run check while the jaws are moving.
    ("open_gripper", "post"): frozenset({"open"}),
    # leave_target intentionally has no gripper-state rule.
}


def get_expected_gripper_states(stage_name: str, phase: str) -> Optional[FrozenSet[str]]:
    return GRIPPER_EXPECTATIONS.get((str(stage_name), str(phase)))


def check_gripper_states_for_stage(
    stage_name: str,
    phase: str,
    observed_states: Set[str],
) -> Tuple[bool, str]:
    required = get_expected_gripper_states(stage_name, phase)
    if required is None:
        return True, ""

    observed = set(observed_states or set())
    missing = sorted(state for state in required if state not in observed)
    if missing:
        return (
            False,
            "stage=%s phase=%s missing=%s observed=%s"
            % (stage_name, phase, missing, sorted(observed)),
        )
    return True, ""
