#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


TOPIC_DEVIATION_ALERT = "/fault/deviation_alert"
TOPIC_DIAGNOSIS_REPORT = "/fault/diagnosis_report"
TOPIC_RECOVERY_PLAN = "/recovery/plan"
TOPIC_RECOVERY_FEEDBACK = "/recovery/feedback"
TOPIC_RECOVERY_RESOLUTION = "/recovery/resolution"
TOPIC_EXECUTION_CONTROL = "/task_execution/control"
TOPIC_EXECUTION_RECOVERY_READY = "/task_execution/recovery_ready"
TOPIC_RECOVERY_RUNTIME_CONTEXT = "/recovery/runtime_context"

CMD_PAUSE_EXECUTION = "pause"
CMD_RESUME_EXECUTION = "resume"
CMD_ABORT_EXECUTION = "abort"
CMD_HANDOVER_TO_RECOVERY = "handover_to_recovery"
CMD_MARK_CURRENT_TASK_DONE = "mark_current_task_done"


def new_event_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class DeviationAlert:
    alert_id: str = field(default_factory=lambda: new_event_id("alert"))
    request_id: str = ""
    source: str = "validation_monitor"
    run_id: str = ""
    attempt_id: Optional[int] = None
    stage_seq: Optional[int] = None
    barrier_id: str = ""
    stage_name: str = ""
    execution_step: Optional[int] = None
    severity: str = "error"
    comparison_summary: Dict[str, Any] = field(default_factory=dict)
    issues: List[Dict[str, Any]] = field(default_factory=list)
    expected_state_brief: str = ""
    graph_brief: str = ""
    runtime_summary: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "DeviationAlert":
        return cls(**json.loads(raw))


@dataclass
class DiagnosisReport:
    diagnosis_id: str = field(default_factory=lambda: new_event_id("diag"))
    alert_id: str = ""
    run_id: str = ""
    attempt_id: Optional[int] = None
    stage_seq: Optional[int] = None
    barrier_id: str = ""
    stage_name: str = ""
    execution_step: Optional[int] = None
    severity: str = "error"
    fault_type: str = "unknown_fault"
    root_cause: str = ""
    confidence: float = 0.5
    retryable: bool = True
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "DiagnosisReport":
        return cls(**json.loads(raw))


@dataclass
class RecoveryAction:
    action_type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryPlan:
    plan_id: str = field(default_factory=lambda: new_event_id("plan"))
    diagnosis_id: str = ""
    alert_id: str = ""
    run_id: str = ""
    attempt_id: Optional[int] = None
    stage_seq: Optional[int] = None
    barrier_id: str = ""
    stage_name: str = ""
    execution_step: Optional[int] = None
    strategy_name: str = ""
    actions: List[Dict[str, Any]] = field(default_factory=list)
    max_retry: int = 1
    abort_if_failed: bool = True
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "RecoveryPlan":
        return cls(**json.loads(raw))


@dataclass
class RecoveryFeedback:
    feedback_id: str = field(default_factory=lambda: new_event_id("feedback"))
    plan_id: str = ""
    run_id: str = ""
    attempt_id: Optional[int] = None
    stage_seq: Optional[int] = None
    barrier_id: str = ""
    stage_name: str = ""
    execution_step: Optional[int] = None
    status: str = "running"   # started / running / succeeded / failed
    current_action: str = ""
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "RecoveryFeedback":
        return cls(**json.loads(raw))


@dataclass
class RecoveryResolution:
    resolution_id: str = field(default_factory=lambda: new_event_id("resolution"))
    source: str = "validation_monitor"
    run_id: str = ""
    attempt_id: Optional[int] = None
    stage_seq: Optional[int] = None
    barrier_id: str = ""
    stage_name: str = ""
    execution_step: Optional[int] = None
    passed: bool = False
    compare_kind: str = "post"
    comparison_summary: Dict[str, Any] = field(default_factory=dict)
    runtime_summary: Dict[str, Any] = field(default_factory=dict)
    graph_brief: str = ""
    message: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "RecoveryResolution":
        return cls(**json.loads(raw))


@dataclass
class RecoveryRuntimeContext:
    run_id: str = ""
    attempt_id: Optional[int] = None
    stage_seq: Optional[int] = None
    actuation_seq: Optional[int] = None
    step: Optional[int] = None
    target_object_id: Optional[str] = None
    target_class: str = ""
    target_place: List[float] = field(default_factory=list)
    source_pose: Optional[List[float]] = None
    # 抓取时记录的安全返回位姿，格式均为 [x,y,z,qx,qy,qz,qw]
    source_return_approach_pose: Optional[List[float]] = None
    source_return_release_pose: Optional[List[float]] = None
    source_return_lift_pose: Optional[List[float]] = None
    gripper_command_id: str = ""
    gripper_command_stamp: float = 0.0
    action_type: str = ""
    expected_support_object_id: Optional[str] = None
    comment: str = ""
    status: str = "idle"   # idle / active / recovery / completed / aborted
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "RecoveryRuntimeContext":
        data = json.loads(raw)
        # 兼容执行端临时扩展字段：只保留 dataclass 定义的字段。
        valid = set(cls.__dataclass_fields__.keys())
        data = {k: v for k, v in data.items() if k in valid}
        return cls(**data)
