#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ROS-independent validation transaction protocol and state machine.

This module deliberately contains no rospy imports.  Execution, perception,
scene-graph, validation, and recovery nodes share these definitions without
pulling ROS node initialization into unit tests.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


TOPIC_VALIDATION_REQUEST = "/task_execution/validation_request"
TOPIC_OBSERVATION_REQUEST = "/scene_graph/observation_request"
TOPIC_VALIDATION_RESULT = "/task_execution/validation_result"

ERROR_CODE_VISION_UNAVAILABLE = "vision_unavailable"


class ValidationStatus:
    ACCEPTED = "accepted"
    WAITING_SCENE = "waiting_scene"
    PASSED = "passed"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    BUSY = "busy"
    INVALID = "invalid"

    TERMINAL = frozenset(
        {
            PASSED,
            FAILED,
            UNAVAILABLE,
            TIMEOUT,
            CANCELLED,
            BUSY,
            INVALID,
        }
    )


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("%s is required" % field_name)
    return text


def _positive_int(value: Any, field_name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("%s must be positive" % field_name)
    return parsed


@dataclass(frozen=True)
class CaptureInterval:
    """Conservative ROS-time interval containing the physical frame capture."""

    capture_stamp: float
    lower_bound: float
    upper_bound: float
    uncertainty_sec: float
    timestamp_source: str
    source_timestamp: Optional[float] = None
    source_timestamp_domain: str = "unknown"

    def __post_init__(self) -> None:
        lower = float(self.lower_bound)
        upper = float(self.upper_bound)
        stamp = float(self.capture_stamp)
        uncertainty = float(self.uncertainty_sec)
        if uncertainty < 0.0:
            raise ValueError("uncertainty_sec must be non-negative")
        if lower > stamp or stamp > upper:
            raise ValueError("capture_stamp must lie inside capture interval")
        object.__setattr__(self, "capture_stamp", stamp)
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "uncertainty_sec", uncertainty)
        object.__setattr__(self, "timestamp_source", str(self.timestamp_source or "unknown"))
        object.__setattr__(
            self,
            "source_timestamp",
            None if self.source_timestamp is None else float(self.source_timestamp),
        )
        object.__setattr__(
            self,
            "source_timestamp_domain",
            str(self.source_timestamp_domain or "unknown"),
        )

    @classmethod
    def from_host_receive(
        cls,
        *,
        host_receive_stamp: float,
        uncertainty_sec: float,
        timestamp_source: str = "host_receive",
        source_timestamp: Optional[float] = None,
        source_timestamp_domain: str = "unknown",
    ) -> "CaptureInterval":
        uncertainty = float(uncertainty_sec)
        if uncertainty < 0.0:
            raise ValueError("uncertainty_sec must be non-negative")
        upper = float(host_receive_stamp)
        lower = upper - uncertainty
        return cls(
            capture_stamp=(lower + upper) / 2.0,
            lower_bound=lower,
            upper_bound=upper,
            uncertainty_sec=uncertainty,
            timestamp_source=timestamp_source,
            source_timestamp=source_timestamp,
            source_timestamp_domain=source_timestamp_domain,
        )

    def to_dict(self, prefix: str = "capture_") -> Dict[str, Any]:
        return {
            "%sstamp" % prefix: self.capture_stamp,
            "%slower_bound" % prefix: self.lower_bound,
            "%supper_bound" % prefix: self.upper_bound,
            "%suncertainty_sec" % prefix: self.uncertainty_sec,
            "%stimestamp_source" % prefix: self.timestamp_source,
            "source_timestamp": self.source_timestamp,
            "source_timestamp_domain": self.source_timestamp_domain,
        }


@dataclass(frozen=True)
class TemporalContext:
    run_id: str
    attempt_id: int
    stage_seq: int
    stage_name: str
    barrier_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _required_text(self.run_id, "run_id"))
        object.__setattr__(self, "attempt_id", _positive_int(self.attempt_id, "attempt_id"))
        object.__setattr__(self, "stage_seq", _positive_int(self.stage_seq, "stage_seq"))
        object.__setattr__(self, "stage_name", _required_text(self.stage_name, "stage_name"))
        object.__setattr__(self, "barrier_id", _required_text(self.barrier_id, "barrier_id"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalContext":
        if not isinstance(data, dict):
            raise ValueError("temporal context must be an object")
        return cls(
            run_id=data.get("run_id", ""),
            attempt_id=data.get("attempt_id", 0),
            stage_seq=data.get("stage_seq", 0),
            stage_name=data.get("stage_name", ""),
            barrier_id=data.get("barrier_id", ""),
        )


def _context_from_payload(data: Dict[str, Any]) -> TemporalContext:
    nested = data.get("context")
    return TemporalContext.from_dict(nested if isinstance(nested, dict) else data)


@dataclass(frozen=True)
class ValidationRequest:
    request_id: str
    context: TemporalContext
    policy: str
    not_before: float
    required_distinct_frames: int
    timeout_sec: float
    created_stamp: float
    gripper_command_id: str = ""
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _required_text(self.request_id, "request_id"))
        if not isinstance(self.context, TemporalContext):
            raise ValueError("context must be TemporalContext")
        object.__setattr__(self, "policy", _required_text(self.policy, "policy"))
        object.__setattr__(self, "not_before", float(self.not_before))
        object.__setattr__(
            self,
            "required_distinct_frames",
            _positive_int(self.required_distinct_frames, "required_distinct_frames"),
        )
        timeout = float(self.timeout_sec)
        if timeout <= 0.0:
            raise ValueError("timeout_sec must be positive")
        object.__setattr__(self, "timeout_sec", timeout)
        object.__setattr__(self, "created_stamp", float(self.created_stamp))
        object.__setattr__(
            self,
            "gripper_command_id",
            str(self.gripper_command_id or "").strip(),
        )
        if int(self.schema_version) != 1:
            raise ValueError("unsupported validation request schema_version")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "schema_version": int(self.schema_version),
            "request_id": self.request_id,
            "policy": self.policy,
            "not_before": self.not_before,
            "required_distinct_frames": self.required_distinct_frames,
            "timeout_sec": self.timeout_sec,
            "created_stamp": self.created_stamp,
            "gripper_command_id": self.gripper_command_id,
            "context": self.context.to_dict(),
        }
        payload.update(self.context.to_dict())
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationRequest":
        if not isinstance(data, dict):
            raise ValueError("validation request must be an object")
        return cls(
            schema_version=data.get("schema_version", 1),
            request_id=data.get("request_id", ""),
            context=_context_from_payload(data),
            policy=data.get("policy", ""),
            not_before=data.get("not_before", 0.0),
            required_distinct_frames=data.get("required_distinct_frames", 1),
            timeout_sec=data.get("timeout_sec", 0.0),
            created_stamp=data.get("created_stamp", 0.0),
            gripper_command_id=data.get("gripper_command_id", ""),
        )

    @classmethod
    def from_json(cls, raw: str) -> "ValidationRequest":
        return cls.from_dict(json.loads(str(raw)))


@dataclass(frozen=True)
class ValidationResult:
    request_id: str
    context: TemporalContext
    status: str
    scene_versions: List[int] = field(default_factory=list)
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _required_text(self.request_id, "request_id"))
        if not isinstance(self.context, TemporalContext):
            raise ValueError("context must be TemporalContext")
        status = str(self.status or "").strip().lower()
        valid_statuses = ValidationStatus.TERMINAL | {
            ValidationStatus.ACCEPTED,
            ValidationStatus.WAITING_SCENE,
        }
        if status not in valid_statuses:
            raise ValueError("unsupported validation status: %s" % status)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "scene_versions", [int(x) for x in self.scene_versions])
        object.__setattr__(self, "message", str(self.message or ""))
        object.__setattr__(self, "timestamp", float(self.timestamp))
        if int(self.schema_version) != 1:
            raise ValueError("unsupported validation result schema_version")

    @property
    def terminal(self) -> bool:
        return self.status in ValidationStatus.TERMINAL

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "schema_version": int(self.schema_version),
            "request_id": self.request_id,
            "status": self.status,
            "scene_versions": list(self.scene_versions),
            "message": self.message,
            "timestamp": self.timestamp,
            "context": self.context.to_dict(),
        }
        payload.update(self.context.to_dict())
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationResult":
        if not isinstance(data, dict):
            raise ValueError("validation result must be an object")
        return cls(
            schema_version=data.get("schema_version", 1),
            request_id=data.get("request_id", ""),
            context=_context_from_payload(data),
            status=data.get("status", ""),
            scene_versions=data.get("scene_versions", []),
            message=data.get("message", ""),
            timestamp=data.get("timestamp", time.time()),
        )

    @classmethod
    def from_json(cls, raw: str) -> "ValidationResult":
        return cls.from_dict(json.loads(str(raw)))

    @classmethod
    def for_request(
        cls,
        request: ValidationRequest,
        *,
        status: str,
        scene_versions: Optional[List[int]] = None,
        message: str = "",
        timestamp: Optional[float] = None,
    ) -> "ValidationResult":
        return cls(
            request_id=request.request_id,
            context=request.context,
            status=status,
            scene_versions=list(scene_versions or []),
            message=message,
            timestamp=time.time() if timestamp is None else float(timestamp),
        )


def snapshot_matches_request(
    request: ValidationRequest,
    snapshot: Dict[str, Any],
) -> Tuple[bool, str]:
    if not isinstance(snapshot, dict):
        return False, "missing_scene_graph_snapshot"
    if not bool(snapshot.get("transaction_eligible", False)):
        return False, "snapshot_not_transaction_eligible"

    expected = request.context
    comparisons = (
        ("run_id", expected.run_id),
        ("attempt_id", expected.attempt_id),
        ("stage_seq", expected.stage_seq),
        ("stage_name", expected.stage_name),
        ("barrier_id", expected.barrier_id),
    )
    for key, expected_value in comparisons:
        actual = snapshot.get(key)
        if key in {"attempt_id", "stage_seq"}:
            try:
                actual = int(actual)
            except (TypeError, ValueError):
                return False, "missing_%s" % key
        else:
            actual = str(actual or "").strip()
        if actual != expected_value:
            return False, "%s_mismatch" % key

    try:
        scene_version = int(snapshot.get("scene_version"))
    except (TypeError, ValueError):
        return False, "missing_scene_version"
    if scene_version <= 0:
        return False, "missing_scene_version"

    try:
        capture_lower_bound = float(snapshot.get("capture_lower_bound"))
    except (TypeError, ValueError):
        return False, "missing_capture_lower_bound"
    if capture_lower_bound < request.not_before:
        return False, "capture_before_not_before"

    return True, ""


def result_matches_request(
    request: ValidationRequest,
    result: ValidationResult,
) -> Tuple[bool, str]:
    if request.request_id != result.request_id:
        return False, "request_id_mismatch"
    expected = request.context
    actual = result.context
    for key in ("run_id", "attempt_id", "stage_seq", "stage_name", "barrier_id"):
        if getattr(expected, key) != getattr(actual, key):
            return False, "%s_mismatch" % key
    return True, ""


def evaluate_gripper_evidence_for_request(
    request: ValidationRequest,
    evidence: Optional[Dict[str, Any]],
    *,
    min_stable_count: int,
) -> Tuple[str, str]:
    """Pure tri-state check for direct grasp-commit evidence."""
    if request.policy != "grasp_commit":
        return "passed", "policy_does_not_require_direct_gripper_evidence"
    if not request.gripper_command_id:
        return "waiting", "missing_request_gripper_command_id"
    if not isinstance(evidence, dict) or not evidence:
        return "waiting", "missing_gripper_evidence"
    if str(evidence.get("command_id", "") or "") != request.gripper_command_id:
        return "waiting", "gripper_command_id_mismatch"
    if str(evidence.get("command", "") or "") != "close":
        return "waiting", "gripper_command_type_mismatch"
    if str(evidence.get("run_id", "") or "") != request.context.run_id:
        return "waiting", "gripper_run_id_mismatch"
    try:
        if int(evidence.get("attempt_id")) != request.context.attempt_id:
            return "waiting", "gripper_attempt_id_mismatch"
        if float(evidence.get("sample_end_stamp")) < request.not_before:
            return "waiting", "gripper_sample_before_not_before"
        stable_count = int(evidence.get("stable_count", 0))
    except (TypeError, ValueError):
        return "waiting", "invalid_gripper_evidence_fields"
    if not bool(evidence.get("valid", False)):
        return "waiting", "gripper_evidence_invalid"
    if stable_count < max(1, int(min_stable_count)):
        return "waiting", "gripper_evidence_unstable"
    states = set(evidence.get("states", []) or [])
    if not states or "unknown" in states:
        return "waiting", "gripper_state_unknown"
    if "holding" in states:
        return "passed", "gripper_holding_confirmed"
    return "failed", "stable_gripper_evidence_missing_holding"


def event_context_matches(
    current: Dict[str, Any],
    event: Dict[str, Any],
    *,
    allow_legacy: bool = False,
) -> Tuple[bool, str]:
    current_run = str((current or {}).get("run_id", "") or "").strip()
    event_run = str((event or {}).get("run_id", "") or "").strip()
    try:
        current_attempt = int((current or {}).get("attempt_id"))
    except (TypeError, ValueError):
        current_attempt = None
    try:
        event_attempt = int((event or {}).get("attempt_id"))
    except (TypeError, ValueError):
        event_attempt = None
    try:
        current_stage_seq = int((current or {}).get("stage_seq"))
    except (TypeError, ValueError):
        current_stage_seq = None
    try:
        event_stage_seq = int((event or {}).get("stage_seq"))
    except (TypeError, ValueError):
        event_stage_seq = None

    if not event_run or event_attempt is None:
        if allow_legacy:
            return True, "legacy_context"
        return False, "missing_event_context"
    if current_run and event_run != current_run:
        return False, "run_id_mismatch"
    if current_attempt is not None and event_attempt != current_attempt:
        return False, "attempt_id_mismatch"
    if (
        current_stage_seq is not None
        and event_stage_seq is not None
        and event_stage_seq != current_stage_seq
    ):
        return False, "stage_seq_mismatch"
    return True, ""


def processing_cycle_due(
    last_monotonic: Optional[float],
    now_monotonic: float,
    rate_hz: float,
) -> bool:
    rate = float(rate_hz)
    if rate <= 0.0:
        return False
    if last_monotonic is None:
        return True
    return float(now_monotonic) - float(last_monotonic) >= (1.0 / rate) - 1e-9


class BarrierRegistry:
    """Single-active-request registry for the current sequential executor."""

    def __init__(self, terminal_retention_sec: float = 60.0):
        self.terminal_retention_sec = max(0.0, float(terminal_retention_sec))
        self._active_request: Optional[ValidationRequest] = None
        self._active_started_monotonic: Optional[float] = None
        self._scene_versions: List[int] = []
        self._semantic_signature: Any = None
        self._comparison_passed: Optional[bool] = None
        self._terminal_by_request: Dict[str, Tuple[ValidationResult, float]] = {}

    @property
    def active_request(self) -> Optional[ValidationRequest]:
        return self._active_request

    def _prune(self, monotonic_now: float) -> None:
        expired = [
            request_id
            for request_id, (_, expires_at) in self._terminal_by_request.items()
            if monotonic_now >= expires_at
        ]
        for request_id in expired:
            self._terminal_by_request.pop(request_id, None)

    def _progress_result(self, status: str, message: str = "") -> ValidationResult:
        if self._active_request is None:
            raise RuntimeError("no active validation request")
        return ValidationResult.for_request(
            self._active_request,
            status=status,
            scene_versions=self._scene_versions,
            message=message,
        )

    def register(self, request: ValidationRequest, monotonic_now: float) -> ValidationResult:
        now = float(monotonic_now)
        self._prune(now)

        cached = self._terminal_by_request.get(request.request_id)
        if cached is not None:
            return cached[0]

        if self._active_request is not None:
            if self._active_request.request_id == request.request_id:
                status = (
                    ValidationStatus.WAITING_SCENE
                    if self._scene_versions
                    else ValidationStatus.ACCEPTED
                )
                return self._progress_result(status)
            return ValidationResult.for_request(
                request,
                status=ValidationStatus.BUSY,
                message="another validation request is active",
            )

        self._active_request = request
        self._active_started_monotonic = now
        self._scene_versions = []
        self._semantic_signature = None
        self._comparison_passed = None
        return self._progress_result(ValidationStatus.ACCEPTED)

    def observe(
        self,
        snapshot: Dict[str, Any],
        *,
        semantic_signature: Any,
        comparison_passed: bool = True,
        monotonic_now: float,
    ) -> Optional[ValidationResult]:
        if self._active_request is None:
            return None

        timed_out = self.expire(monotonic_now=float(monotonic_now))
        if timed_out is not None:
            return timed_out

        matched, reason = snapshot_matches_request(self._active_request, snapshot)
        if not matched:
            return self._progress_result(ValidationStatus.WAITING_SCENE, reason)

        scene_version = int(snapshot["scene_version"])
        if scene_version in self._scene_versions:
            return self._progress_result(ValidationStatus.WAITING_SCENE)

        passed = bool(comparison_passed)
        if (
            self._semantic_signature is None
            or (
                semantic_signature == self._semantic_signature
                and passed == self._comparison_passed
            )
        ):
            self._scene_versions.append(scene_version)
        else:
            self._scene_versions = [scene_version]
        self._semantic_signature = semantic_signature
        self._comparison_passed = passed

        if len(self._scene_versions) >= self._active_request.required_distinct_frames:
            return self.complete(
                ValidationResult.for_request(
                    self._active_request,
                    status=(ValidationStatus.PASSED if passed else ValidationStatus.FAILED),
                    scene_versions=self._scene_versions,
                ),
                monotonic_now=float(monotonic_now),
            )
        return self._progress_result(ValidationStatus.WAITING_SCENE)

    def complete(
        self,
        result: ValidationResult,
        *,
        monotonic_now: float,
    ) -> ValidationResult:
        now = float(monotonic_now)
        self._prune(now)
        cached = self._terminal_by_request.get(result.request_id)
        if cached is not None:
            return cached[0]

        if not result.terminal:
            return result
        if self._active_request is None or self._active_request.request_id != result.request_id:
            return result

        expires_at = now + self.terminal_retention_sec
        self._terminal_by_request[result.request_id] = (result, expires_at)
        self._active_request = None
        self._active_started_monotonic = None
        self._scene_versions = []
        self._semantic_signature = None
        self._comparison_passed = None
        return result

    def expire(self, monotonic_now: float) -> Optional[ValidationResult]:
        if self._active_request is None or self._active_started_monotonic is None:
            return None
        now = float(monotonic_now)
        if now - self._active_started_monotonic < self._active_request.timeout_sec:
            return None
        return self.complete(
            ValidationResult.for_request(
                self._active_request,
                status=ValidationStatus.TIMEOUT,
                scene_versions=self._scene_versions,
                message="validation request timed out",
            ),
            monotonic_now=now,
        )

    def cancel(self, request_id: str, monotonic_now: float, message: str = "") -> Optional[ValidationResult]:
        if self._active_request is None or self._active_request.request_id != str(request_id):
            return None
        return self.complete(
            ValidationResult.for_request(
                self._active_request,
                status=ValidationStatus.CANCELLED,
                scene_versions=self._scene_versions,
                message=message,
            ),
            monotonic_now=float(monotonic_now),
        )


def validate_snapshot_event_alignment(
    snapshot: Dict[str, Any],
    *,
    stamp_tolerance_sec: float,
    reference_ts: Optional[float] = None,
    require_event_alignment: bool = True,
    now_sec: Optional[float] = None,
    max_detection_age_sec: Optional[float] = None,
) -> Tuple[bool, str]:
    if not require_event_alignment:
        return True, ""
    if not isinstance(snapshot, dict):
        return False, "missing_scene_graph_snapshot"
    try:
        detection_stamp = float(snapshot.get("detection_stamp"))
    except (TypeError, ValueError):
        return False, "missing_detection_stamp"
    if detection_stamp <= 0.0:
        return False, "missing_detection_stamp"
    if now_sec is not None and max_detection_age_sec is not None:
        detection_age = float(now_sec) - detection_stamp
        tolerance = float(stamp_tolerance_sec)
        if detection_age > float(max_detection_age_sec) + tolerance:
            return False, "stale_detection_age"
        if detection_age < -tolerance:
            return False, "detection_stamp_in_future"
    if reference_ts is not None and detection_stamp < float(reference_ts) - float(stamp_tolerance_sec):
        return False, "detection_before_reference"
    return True, ""


class VisionUnavailableError(RuntimeError):
    def __init__(self, message: str):
        super().__init__(message)
        self.error_code = ERROR_CODE_VISION_UNAVAILABLE


class FreshSceneWaitRegistry:
    def __init__(self, timeout_sec: float):
        self.timeout_sec = max(0.0, float(timeout_sec))
        self._started_at_by_key: Dict[tuple, float] = {}

    def mark_waiting(self, key: tuple, now_sec: float) -> Tuple[float, bool]:
        now = float(now_sec)
        started = self._started_at_by_key.setdefault(tuple(key), now)
        elapsed = max(0.0, now - started)
        return elapsed, elapsed >= self.timeout_sec

    def mark_fresh(self, key: tuple) -> None:
        self._started_at_by_key.pop(tuple(key), None)


class VersionedSnapshotBuffer:
    """Thread-safe bounded FIFO that retains each scene version once."""

    def __init__(self, maxlen: int = 10):
        size = int(maxlen)
        if size <= 0:
            raise ValueError("maxlen must be positive")
        self.maxlen = size
        self._items = deque()
        self._versions = set()
        self._lock = threading.Lock()

    def append(self, snapshot: Dict[str, Any]) -> bool:
        if not isinstance(snapshot, dict):
            return False
        try:
            scene_version = int(snapshot.get("scene_version"))
        except (TypeError, ValueError):
            return False
        if scene_version <= 0:
            return False

        with self._lock:
            if scene_version in self._versions:
                return False
            if len(self._items) >= self.maxlen:
                removed = self._items.popleft()
                try:
                    self._versions.discard(int(removed.get("scene_version")))
                except (TypeError, ValueError):
                    pass
            copied = dict(snapshot)
            self._items.append(copied)
            self._versions.add(scene_version)
        return True

    def drain(self) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._items)
            self._items.clear()
            self._versions.clear()
        return items

    def clear(self) -> None:
        self.drain()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
