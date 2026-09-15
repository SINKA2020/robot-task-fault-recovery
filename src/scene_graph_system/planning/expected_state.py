from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from scene_graph_system.scene_graph.relation_utils import normalize_relation_name


# ============================================================
# Expected-state primitives
# ============================================================

_VALID_SEVERITIES = {"warning", "error", "critical"}


def _dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


@dataclass
class ExpectedNode:
    """
    Expected node definition for a task stage.

    Attributes:
        node_id:
            The expected node identifier. In the minimal version, this should
            match the observed SceneGraph node uid() exactly.
        class_name:
            Optional expected class/category name, compared against Node.name.
        required_states:
            States that must exist in the observed node.states.
        required_states_any:
            Alternative states where at least one must exist.
        optional_states:
            States that are acceptable but not mandatory.
        required:
            Whether this node must exist.
        severity:
            warning / error / critical
        note:
            Optional human-readable note.
    """
    node_id: str
    class_name: Optional[str] = None
    required_states: List[str] = field(default_factory=list)
    required_states_any: List[str] = field(default_factory=list)
    optional_states: List[str] = field(default_factory=list)
    required: bool = True
    severity: str = "error"
    note: str = ""

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("ExpectedNode.node_id must be non-empty")
        if self.severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"Invalid severity '{self.severity}', must be one of {_VALID_SEVERITIES}"
            )
        self.required_states = _dedup_keep_order(list(self.required_states))
        self.required_states_any = _dedup_keep_order(list(self.required_states_any))
        self.optional_states = _dedup_keep_order(list(self.optional_states))


@dataclass
class ExpectedRelation:
    """
    Expected relation constraint.

    Example:
        cube_3 --on--> cuboid_1
    """
    subject_id: str
    object_id: str
    relation: str
    required: bool = True
    severity: str = "error"
    note: str = ""

    def __post_init__(self) -> None:
        if not self.subject_id:
            raise ValueError("ExpectedRelation.subject_id must be non-empty")
        if not self.object_id:
            raise ValueError("ExpectedRelation.object_id must be non-empty")
        if not self.relation:
            raise ValueError("ExpectedRelation.relation must be non-empty")
        self.relation = normalize_relation_name(self.relation)
        if self.severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"Invalid severity '{self.severity}', must be one of {_VALID_SEVERITIES}"
            )

    def key(self) -> Tuple[str, str, str]:
        return (self.subject_id, self.object_id, self.relation)

    def to_readable(self) -> str:
        return f"{self.subject_id} --{self.relation}--> {self.object_id}"


@dataclass
class ExpectedState:
    """
    Expected state for one task stage / action step.
    """
    stage_name: str

    expected_nodes: List[ExpectedNode] = field(default_factory=list)
    expected_relations: List[ExpectedRelation] = field(default_factory=list)
    forbidden_relations: List[ExpectedRelation] = field(default_factory=list)

    description: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stage_name:
            raise ValueError("ExpectedState.stage_name must be non-empty")

    def get_expected_node(self, node_id: str) -> Optional[ExpectedNode]:
        for node in self.expected_nodes:
            if node.node_id == node_id:
                return node
        return None

    def all_node_ids(self) -> List[str]:
        return [node.node_id for node in self.expected_nodes]

    def all_required_node_ids(self) -> List[str]:
        return [node.node_id for node in self.expected_nodes if node.required]

    def validate(self) -> None:
        """
        Basic self-check:
        - no duplicate expected node ids
        - no duplicate expected relations
        - no duplicate forbidden relations
        """
        node_ids = set()
        for node in self.expected_nodes:
            if node.node_id in node_ids:
                raise ValueError(f"Duplicate expected node_id: {node.node_id}")
            node_ids.add(node.node_id)

        rel_keys = set()
        for rel in self.expected_relations:
            if rel.key() in rel_keys:
                raise ValueError(f"Duplicate expected relation: {rel.key()}")
            rel_keys.add(rel.key())

        forbidden_keys = set()
        for rel in self.forbidden_relations:
            if rel.key() in forbidden_keys:
                raise ValueError(f"Duplicate forbidden relation: {rel.key()}")
            forbidden_keys.add(rel.key())

    def to_summary(self) -> Dict[str, object]:
        return {
            "stage_name": self.stage_name,
            "description": self.description,
            "expected_node_count": len(self.expected_nodes),
            "expected_relation_count": len(self.expected_relations),
            "forbidden_relation_count": len(self.forbidden_relations),
            "metadata": dict(self.metadata),
        }

    def summary_dict(self) -> Dict[str, object]:
        return self.to_summary()


# ============================================================
# Registry
# ============================================================

class ExpectedStateRegistry:
    """
    Simple registry mapping action/stage name -> ExpectedState.
    """

    def __init__(self) -> None:
        self._states: Dict[str, ExpectedState] = {}

    def register(self, action_name: str, expected_state: ExpectedState) -> None:
        if not action_name:
            raise ValueError("action_name must be non-empty")
        expected_state.validate()
        self._states[action_name] = expected_state

    def get(self, action_name: str) -> ExpectedState:
        if action_name not in self._states:
            raise KeyError(f"Expected state not found for action: {action_name}")
        return self._states[action_name]

    def has(self, action_name: str) -> bool:
        return action_name in self._states

    def keys(self) -> Sequence[str]:
        return tuple(self._states.keys())

    def clear(self) -> None:
        self._states.clear()

    def count(self) -> int:
        return len(self._states)

    def to_summary(self) -> Dict[str, object]:
        return {
            "count": self.count(),
            "keys": list(self._states.keys()),
        }

    def summary_dict(self) -> Dict[str, object]:
        summary = self.to_summary()
        return {
            "count": summary["count"],
            "actions": list(summary["keys"]),
        }


# ============================================================
# Convenience factory helpers
# ============================================================

def make_on_top_expected_state(
    stage_name: str,
    upper_id: str,
    lower_id: str,
    upper_class: Optional[str] = None,
    lower_class: Optional[str] = None,
    relation_severity: str = "error",
    intersect_severity: str = "critical",
    description: str = "",
    extra_expected_nodes: Optional[Sequence[ExpectedNode]] = None,
) -> ExpectedState:
    """
    Minimal helper for common block-stacking state:
        upper on lower
        upper must not intersect lower
    """
    state = ExpectedState(
        stage_name=stage_name,
        expected_nodes=[
            ExpectedNode(node_id=upper_id, class_name=upper_class),
            ExpectedNode(node_id=lower_id, class_name=lower_class),
            *list(extra_expected_nodes or []),
        ],
        expected_relations=[
            ExpectedRelation(
                subject_id=upper_id,
                object_id=lower_id,
                relation="on",
                required=True,
                severity=relation_severity,
            )
        ],
        forbidden_relations=[
            ExpectedRelation(
                subject_id=upper_id,
                object_id=lower_id,
                relation="intersecting",
                required=False,
                severity=intersect_severity,
            )
        ],
        description=description,
    )
    state.validate()
    return state
