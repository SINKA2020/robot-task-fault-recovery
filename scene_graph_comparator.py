from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from scene_graph_system.planning.expected_state import ExpectedNode, ExpectedRelation, ExpectedState
from scene_graph_system.scene_graph.scene_graph import Node, SceneGraph


_VALID_SEVERITIES = {"warning", "error", "critical"}


@dataclass
class ComparisonIssue:
    issue_type: str
    severity: str
    message: str

    node_id: Optional[str] = None
    subject_id: Optional[str] = None
    object_id: Optional[str] = None
    relation: Optional[str] = None

    expected: Optional[object] = None
    observed: Optional[object] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"Invalid severity '{self.severity}', must be one of {_VALID_SEVERITIES}"
            )


@dataclass
class ComparisonResult:
    stage_name: str
    passed: bool = True

    matched_nodes: List[str] = field(default_factory=list)
    matched_relations: List[str] = field(default_factory=list)
    issues: List[ComparisonIssue] = field(default_factory=list)

    def add_issue(self, issue: ComparisonIssue) -> None:
        self.issues.append(issue)
        if issue.severity in ("error", "critical"):
            self.passed = False

    def issue_count_by_type(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for issue in self.issues:
            out[issue.issue_type] = out.get(issue.issue_type, 0) + 1
        return out

    def issue_count_by_severity(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for issue in self.issues:
            out[issue.severity] = out.get(issue.severity, 0) + 1
        return out

    def highest_severity(self) -> str:
        order = {"warning": 1, "error": 2, "critical": 3}
        best = "ok"
        best_score = 0
        for issue in self.issues:
            score = order.get(issue.severity, 0)
            if score > best_score:
                best_score = score
                best = issue.severity
        return best

    def summary_dict(self) -> Dict[str, object]:
        return {
            "stage_name": self.stage_name,
            "passed": self.passed,
            "highest_severity": self.highest_severity(),
            "matched_nodes": list(self.matched_nodes),
            "matched_relations": list(self.matched_relations),
            "issue_count": len(self.issues),
            "issue_types": self.issue_count_by_type(),
            "issue_severity": self.issue_count_by_severity(),
        }

    def pretty_print(self) -> str:
        lines = [
            f"[ComparisonResult] stage={self.stage_name} "
            f"passed={self.passed} highest_severity={self.highest_severity()}"
        ]

        if self.matched_nodes:
            lines.append("Matched nodes:")
            for node_id in self.matched_nodes:
                lines.append(f"  - {node_id}")

        if self.matched_relations:
            lines.append("Matched relations:")
            for rel in self.matched_relations:
                lines.append(f"  - {rel}")

        if self.issues:
            lines.append("Issues:")
            for issue in self.issues:
                lines.append(f"  - [{issue.severity}] {issue.issue_type}: {issue.message}")
        else:
            lines.append("Issues: none")

        return "\n".join(lines)


@dataclass
class SceneGraphComparatorConfig:
    check_unexpected_nodes: bool = False
    allow_class_name_fallback: bool = True
    treat_warning_as_failure: bool = False

    enable_build_validation_filter: bool = True

    build_object_classes: Tuple[str, ...] = (
        "cube",
        "cuboid",
        "arch",
        "triangle",
    )

    manipulation_required_states: Tuple[str, ...] = (
        "grasped",
        "currently_manipulated",
    )


class SceneGraphComparator:
    """
    安全版类别级比较器。

    核心修复：
      1. gripper 永远从完整图中精确匹配，不走搭建区过滤；
      2. gripper 状态明确不匹配时输出 state_mismatch；时间对齐证据未知时
         输出 warning 级 state_unavailable，不能伪造成抓取失败；
      3. 普通积木仍按类别验证，避免 ID 漂移导致误报；
      4. 关系按类别验证，但只在有效搭建区节点上搜索。
    """

    def __init__(self, config: Optional[SceneGraphComparatorConfig] = None) -> None:
        self.config = config or SceneGraphComparatorConfig()

    def compare(self, expected_state: ExpectedState, observed_graph: SceneGraph) -> ComparisonResult:
        expected_state.validate()
        result = ComparisonResult(stage_name=expected_state.stage_name)

        expected_class_by_id = self._expected_class_by_id(expected_state)
        used_node_ids = set()

        # 1. 先检查 gripper，确保 gripper state_mismatch 不被结构故障掩盖。
        for exp_node in list(getattr(expected_state, "expected_nodes", []) or []):
            if self._is_gripper_expected_node(exp_node):
                self._compare_gripper_node(exp_node, observed_graph, result)

        # 2. 再检查普通节点。
        for exp_node in list(getattr(expected_state, "expected_nodes", []) or []):
            if self._is_gripper_expected_node(exp_node):
                continue
            self._compare_object_node_by_class(exp_node, observed_graph, used_node_ids, result)

        # 3. 类别级关系。
        self._compare_relations_by_class(expected_state, observed_graph, expected_class_by_id, result)

        # 4. 类别级 forbidden relation。
        self._compare_forbidden_relations_by_class(expected_state, observed_graph, expected_class_by_id, result)

        if self.config.check_unexpected_nodes:
            self._compare_unexpected_nodes(expected_state, observed_graph, result)

        if self.config.treat_warning_as_failure:
            if any(issue.severity == "warning" for issue in result.issues):
                result.passed = False

        return result

    # ============================================================
    # Helpers
    # ============================================================

    def _expected_class_by_id(self, expected_state: ExpectedState) -> Dict[str, str]:
        out = {}
        for node in getattr(expected_state, "expected_nodes", []) or []:
            node_id = str(getattr(node, "node_id", "") or "")
            class_name = getattr(node, "class_name", None)
            if node_id and class_name:
                out[node_id] = str(class_name)
        return out

    def _get_nodes(self, graph: SceneGraph) -> List[Node]:
        return list(getattr(graph, "nodes", []) or [])

    def _get_relations(self, graph: SceneGraph):
        try:
            return list(graph.get_relations() or [])
        except Exception:
            return []

    def _is_gripper_expected_node(self, exp_node: ExpectedNode) -> bool:
        node_id = str(getattr(exp_node, "node_id", "") or "").strip().lower()
        class_name = str(getattr(exp_node, "class_name", "") or "").strip().lower()
        return node_id == "gripper" or class_name == "gripper"

    def _find_gripper_node(self, graph: SceneGraph) -> Optional[Node]:
        for node in self._get_nodes(graph):
            if str(node.uid()).lower() == "gripper" or str(node.name).lower() == "gripper":
                return node
        return None

    def _required_states(self, exp_node: ExpectedNode) -> List[str]:
        return list(getattr(exp_node, "required_states", []) or [])

    def _required_states_any(self, exp_node: ExpectedNode) -> List[str]:
        return list(getattr(exp_node, "required_states_any", []) or [])

    def _is_valid_build_node(self, node: Node) -> bool:
        if node is None:
            return False
        if getattr(node, "name", None) == "gripper":
            return False
        if not self.config.enable_build_validation_filter:
            return True

        states = set(getattr(node, "states", []) or [])
        if "in_build_region" not in states:
            return False
        if "grasped" in states:
            return False
        if "currently_manipulated" in states:
            return False
        if "exclude_from_build_validation" in states:
            return False
        return True

    def _requires_manipulation_state(self, exp_node: ExpectedNode) -> bool:
        states = set(self._required_states(exp_node))
        return bool(states & set(self.config.manipulation_required_states))

    # ============================================================
    # Node comparison
    # ============================================================

    def _compare_gripper_node(self, exp_node: ExpectedNode, graph: SceneGraph, result: ComparisonResult) -> None:
        obs_node = self._find_gripper_node(graph)

        if obs_node is None:
            if getattr(exp_node, "required", True):
                result.add_issue(ComparisonIssue(
                    issue_type="missing_node",
                    severity=getattr(exp_node, "severity", "error"),
                    message="Expected gripper node is missing.",
                    node_id=getattr(exp_node, "node_id", "gripper"),
                    expected="gripper",
                    observed=None,
                    metadata={"match_mode": "gripper_exact"},
                ))
            return

        observed_states = set(getattr(obs_node, "states", []) or [])
        result.matched_nodes.append(f"{getattr(exp_node, 'node_id', 'gripper')}->{obs_node.uid()}[gripper]")

        if not observed_states or "unknown" in observed_states:
            attributes = dict(getattr(obs_node, "attributes", {}) or {})
            result.add_issue(ComparisonIssue(
                issue_type="state_unavailable",
                severity="warning",
                message=(
                    "Time-aligned gripper state is unavailable; defer fault "
                    "classification instead of treating it as a mismatch."
                ),
                node_id="gripper",
                expected={
                    "required_states": self._required_states(exp_node),
                    "required_states_any": self._required_states_any(exp_node),
                },
                observed=sorted(observed_states),
                metadata={
                    "match_mode": "gripper_exact",
                    "alignment_valid": attributes.get("gripper_alignment_valid"),
                    "alignment_reason": attributes.get("gripper_alignment_reason"),
                    "sample_seq": attributes.get("gripper_sample_seq"),
                },
            ))
            return

        for req_state in self._required_states(exp_node):
            if req_state not in observed_states:
                result.add_issue(ComparisonIssue(
                    issue_type="state_mismatch",
                    severity=getattr(exp_node, "severity", "error"),
                    message=(
                        f"Node 'gripper' missing required state '{req_state}'. "
                        f"Observed states={sorted(observed_states)}"
                    ),
                    node_id="gripper",
                    expected=req_state,
                    observed=sorted(observed_states),
                    metadata={
                        "match_mode": "gripper_exact",
                        "missing_state": req_state,
                        "observed_node_id": obs_node.uid(),
                    },
                ))

        required_any = self._required_states_any(exp_node)
        if required_any and not (set(required_any) & observed_states):
            result.add_issue(ComparisonIssue(
                issue_type="state_mismatch",
                severity=getattr(exp_node, "severity", "error"),
                message=(
                    "Node 'gripper' missing every alternative required state "
                    f"{required_any}. Observed states={sorted(observed_states)}"
                ),
                node_id="gripper",
                expected=list(required_any),
                observed=sorted(observed_states),
                metadata={
                    "match_mode": "gripper_exact_any",
                    "missing_states_any": list(required_any),
                    "observed_node_id": obs_node.uid(),
                },
            ))

    def _compare_object_node_by_class(
        self,
        exp_node: ExpectedNode,
        graph: SceneGraph,
        used_node_ids: set,
        result: ComparisonResult,
    ) -> None:
        expected_class = getattr(exp_node, "class_name", None)
        candidates = []

        for node in self._get_nodes(graph):
            if node.uid() in used_node_ids:
                continue
            if expected_class is not None and node.name != expected_class:
                continue

            if self._requires_manipulation_state(exp_node):
                # 被夹持/操作中的物体不要求在搭建区。
                candidates.append(node)
            else:
                if self._is_valid_build_node(node):
                    candidates.append(node)

        # 优先选择满足 required states 的候选。
        for node in candidates:
            observed_states = set(getattr(node, "states", []) or [])
            required_any = self._required_states_any(exp_node)
            all_ok = all(req in observed_states for req in self._required_states(exp_node))
            any_ok = (not required_any) or bool(set(required_any) & observed_states)
            if all_ok and any_ok:
                used_node_ids.add(node.uid())
                result.matched_nodes.append(f"{exp_node.node_id}->{node.uid()}[class]")
                return

        if getattr(exp_node, "required", True):
            result.add_issue(ComparisonIssue(
                issue_type="missing_node",
                severity=getattr(exp_node, "severity", "error"),
                message=(
                    f"Expected class '{exp_node.class_name}' for logical node "
                    f"'{exp_node.node_id}' is missing or lacks required states "
                    f"{self._required_states(exp_node)}."
                ),
                node_id=exp_node.node_id,
                expected={
                    "logical_node_id": exp_node.node_id,
                    "class_name": exp_node.class_name,
                    "required_states": self._required_states(exp_node),
                    "required_states_any": self._required_states_any(exp_node),
                },
                observed=None,
                metadata={"match_mode": "class_only", "id_ignored": True},
            ))

    # ============================================================
    # Relation comparison
    # ============================================================

    def _relation_uses_build_filter(self, subj_class: str, obj_class: str) -> bool:
        if subj_class == "gripper" or obj_class == "gripper":
            return False
        return bool(self.config.enable_build_validation_filter)

    def _find_relation_by_class(self, graph: SceneGraph, subject_class: str, object_class: str, relation: str):
        use_build_filter = self._relation_uses_build_filter(subject_class, object_class)

        for edge in self._get_relations(graph):
            if getattr(edge, "edge_type", None) != relation:
                continue
            start = edge.start
            end = edge.end
            if start is None or end is None:
                continue
            if start.name != subject_class or end.name != object_class:
                continue
            if use_build_filter:
                if not self._is_valid_build_node(start) or not self._is_valid_build_node(end):
                    continue
            return edge
        return None

    def _compare_relations_by_class(
        self,
        expected_state: ExpectedState,
        graph: SceneGraph,
        expected_class_by_id: Dict[str, str],
        result: ComparisonResult,
    ) -> None:
        for exp_rel in getattr(expected_state, "expected_relations", []) or []:
            subject_class = expected_class_by_id.get(str(exp_rel.subject_id))
            object_class = expected_class_by_id.get(str(exp_rel.object_id))

            if not subject_class or not object_class:
                result.add_issue(ComparisonIssue(
                    issue_type="relation_endpoint_class_missing",
                    severity=getattr(exp_rel, "severity", "error"),
                    message=(
                        f"Cannot verify relation by class because endpoint class is missing: "
                        f"{exp_rel.subject_id} --{exp_rel.relation}--> {exp_rel.object_id}"
                    ),
                    subject_id=exp_rel.subject_id,
                    object_id=exp_rel.object_id,
                    relation=exp_rel.relation,
                    metadata={"id_ignored": True, "match_mode": "class_relation"},
                ))
                continue

            edge = self._find_relation_by_class(graph, subject_class, object_class, exp_rel.relation)
            if edge is not None:
                result.matched_relations.append(
                    f"{exp_rel.subject_id}->{edge.start.uid()}({subject_class}) "
                    f"--{exp_rel.relation}--> "
                    f"{exp_rel.object_id}->{edge.end.uid()}({object_class}) [class]"
                )
                continue

            result.add_issue(ComparisonIssue(
                issue_type="missing_relation",
                severity=getattr(exp_rel, "severity", "error"),
                message=(
                    f"Missing class-level relation: "
                    f"{subject_class} --{exp_rel.relation}--> {object_class}. "
                    f"Logical IDs ignored: "
                    f"{exp_rel.subject_id} --{exp_rel.relation}--> {exp_rel.object_id}."
                ),
                subject_id=exp_rel.subject_id,
                object_id=exp_rel.object_id,
                relation=exp_rel.relation,
                expected=f"{subject_class} --{exp_rel.relation}--> {object_class}",
                observed=None,
                metadata={
                    "id_ignored": True,
                    "match_mode": "class_relation",
                    "subject_class": subject_class,
                    "object_class": object_class,
                },
            ))

    def _compare_forbidden_relations_by_class(
        self,
        expected_state: ExpectedState,
        graph: SceneGraph,
        expected_class_by_id: Dict[str, str],
        result: ComparisonResult,
    ) -> None:
        for forbid_rel in getattr(expected_state, "forbidden_relations", []) or []:
            subject_class = expected_class_by_id.get(str(forbid_rel.subject_id))
            object_class = expected_class_by_id.get(str(forbid_rel.object_id))
            if not subject_class or not object_class:
                continue

            edge = self._find_relation_by_class(graph, subject_class, object_class, forbid_rel.relation)
            if edge is None:
                continue

            result.add_issue(ComparisonIssue(
                issue_type="forbidden_relation",
                severity=getattr(forbid_rel, "severity", "critical"),
                message=(
                    f"Forbidden class-level relation detected: "
                    f"{subject_class} --{forbid_rel.relation}--> {object_class}. "
                    f"Observed as {edge.start.uid()} --{forbid_rel.relation}--> {edge.end.uid()}."
                ),
                subject_id=forbid_rel.subject_id,
                object_id=forbid_rel.object_id,
                relation=forbid_rel.relation,
                expected=None,
                observed=f"{edge.start.uid()} --{forbid_rel.relation}--> {edge.end.uid()}",
                metadata={
                    "id_ignored": True,
                    "match_mode": "class_forbidden_relation",
                    "subject_class": subject_class,
                    "object_class": object_class,
                    "observed_subject_id": edge.start.uid(),
                    "observed_object_id": edge.end.uid(),
                },
            ))

    def _compare_unexpected_nodes(self, expected_state: ExpectedState, graph: SceneGraph, result: ComparisonResult) -> None:
        # 类别级验证下暂不启用 unexpected node 强约束，避免 ID 漂移造成大量 warning。
        return
