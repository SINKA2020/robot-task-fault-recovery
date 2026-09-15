from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, List, Optional, Sequence, Tuple


GRIPPER_PROTOCOL_VERSION = 1
STATE_OPEN = "open"
STATE_CLOSED = "closed"
STATE_HOLDING = "holding"
STATE_UNKNOWN = "unknown"
SUPPORTED_STATES = frozenset({STATE_OPEN, STATE_CLOSED, STATE_HOLDING, STATE_UNKNOWN})


def normalize_supported_states(states: Iterable[str], *, valid: bool = True) -> Tuple[str, ...]:
    """Return a deterministic, minimal state set used by the runtime protocol."""
    if not valid:
        return (STATE_UNKNOWN,)

    normalized = []
    for state in states or []:
        value = str(state or "").strip().lower()
        if value in SUPPORTED_STATES and value not in normalized:
            normalized.append(value)

    if STATE_HOLDING in normalized and STATE_CLOSED not in normalized:
        normalized.insert(0, STATE_CLOSED)
    if STATE_UNKNOWN in normalized and len(normalized) > 1:
        normalized.remove(STATE_UNKNOWN)
    if not normalized:
        normalized.append(STATE_UNKNOWN)
    return tuple(normalized)


def semantic_state(states: Iterable[str], *, valid: bool = True) -> str:
    normalized = set(normalize_supported_states(states, valid=valid))
    if STATE_UNKNOWN in normalized:
        return STATE_UNKNOWN
    if STATE_HOLDING in normalized:
        return STATE_HOLDING
    if STATE_OPEN in normalized:
        return STATE_OPEN
    if STATE_CLOSED in normalized:
        return STATE_CLOSED
    return STATE_UNKNOWN


@dataclass(frozen=True)
class GripperCommandContext:
    command_id: str
    command: str
    command_stamp: float
    source: str = ""
    run_id: str = ""
    attempt_id: int = -1
    step: int = -1
    stage_seq: int = -1
    actuation_seq: int = -1
    stage_name: str = ""

    def __post_init__(self) -> None:
        if not str(self.command_id or "").strip():
            raise ValueError("command_id must be non-empty")
        command = str(self.command or "").strip().lower()
        if command not in {"open", "close"}:
            raise ValueError("command must be open or close")
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "command_stamp", float(self.command_stamp))


@dataclass(frozen=True)
class GripperSample:
    monitor_instance_id: str
    sample_seq: int
    sample_start_stamp: float
    sample_end_stamp: float
    states: Tuple[str, ...]
    valid: bool = True
    stable_count: int = 1
    holding_latched: bool = False
    command_id: str = ""
    command: str = ""

    def __post_init__(self) -> None:
        start = float(self.sample_start_stamp)
        end = float(self.sample_end_stamp)
        if end < start:
            raise ValueError("sample_end_stamp must be >= sample_start_stamp")
        object.__setattr__(self, "sample_start_stamp", start)
        object.__setattr__(self, "sample_end_stamp", end)
        object.__setattr__(self, "sample_seq", int(self.sample_seq))
        object.__setattr__(self, "stable_count", max(0, int(self.stable_count)))
        object.__setattr__(
            self,
            "states",
            normalize_supported_states(self.states, valid=bool(self.valid)),
        )

    @property
    def midpoint(self) -> float:
        return (self.sample_start_stamp + self.sample_end_stamp) / 2.0

    @property
    def semantic_state(self) -> str:
        return semantic_state(self.states, valid=self.valid)


@dataclass
class GripperStateStabilizer:
    required_count: int = 3
    _last_state: str = field(default=STATE_UNKNOWN, init=False)
    _count: int = field(default=0, init=False)
    holding_latched: bool = field(default=False, init=False)

    def reset(self, *, clear_latch: bool = True) -> None:
        self._last_state = STATE_UNKNOWN
        self._count = 0
        if clear_latch:
            self.holding_latched = False

    def update(self, states: Iterable[str], *, valid: bool = True) -> Tuple[str, int, bool]:
        current = semantic_state(states, valid=valid)
        if current == self._last_state:
            self._count += 1
        else:
            self._last_state = current
            self._count = 1

        stable = self._count >= max(1, int(self.required_count))
        if stable and current == STATE_HOLDING:
            self.holding_latched = True
        elif stable and current in {STATE_OPEN, STATE_CLOSED}:
            self.holding_latched = False
        # Unknown never clears a previously confirmed holding state.
        return current, self._count, self.holding_latched


class GripperSampleHistory:
    """Bounded time history with interval-aware camera/sample matching."""

    def __init__(self, retention_sec: float = 10.0, max_samples: int = 500):
        self.retention_sec = max(0.1, float(retention_sec))
        self.max_samples = max(1, int(max_samples))
        self._samples: Deque[GripperSample] = deque()

    def clear(self) -> None:
        self._samples.clear()

    def add(self, sample: GripperSample) -> bool:
        if self._samples:
            last = self._samples[-1]
            if sample.monitor_instance_id == last.monitor_instance_id and sample.sample_seq <= last.sample_seq:
                return False
            if sample.sample_end_stamp < last.sample_end_stamp:
                return False
        self._samples.append(sample)
        cutoff = sample.sample_end_stamp - self.retention_sec
        while self._samples and (
            len(self._samples) > self.max_samples
            or self._samples[0].sample_end_stamp < cutoff
        ):
            self._samples.popleft()
        return True

    def samples(self) -> List[GripperSample]:
        return list(self._samples)

    @staticmethod
    def _interval_gap(sample: GripperSample, lower: float, upper: float) -> float:
        if sample.sample_end_stamp < lower:
            return lower - sample.sample_end_stamp
        if sample.sample_start_stamp > upper:
            return sample.sample_start_stamp - upper
        return 0.0

    def match_capture_interval(
        self,
        capture_lower_bound: float,
        capture_upper_bound: float,
        *,
        max_skew_sec: float,
    ) -> Tuple[Optional[GripperSample], Optional[float]]:
        lower = float(capture_lower_bound)
        upper = float(capture_upper_bound)
        if upper < lower:
            raise ValueError("capture_upper_bound must be >= capture_lower_bound")
        if not self._samples:
            return None, None

        capture_midpoint = (lower + upper) / 2.0
        ranked = []
        for sample in self._samples:
            gap = self._interval_gap(sample, lower, upper)
            midpoint_delta = abs(sample.midpoint - capture_midpoint)
            ranked.append((gap, midpoint_delta, -sample.sample_seq, sample))
        gap, _, _, sample = min(ranked, key=lambda item: item[:3])
        if gap > max(0.0, float(max_skew_sec)):
            return None, gap
        return sample, gap


def command_matches_sample(
    command: GripperCommandContext,
    sample: GripperSample,
    *,
    require_command_id: bool = True,
) -> Tuple[bool, str]:
    if not sample.valid:
        return False, "invalid_sample"
    if sample.sample_start_stamp < command.command_stamp:
        return False, "sample_started_before_command"
    if require_command_id and sample.command_id != command.command_id:
        return False, "command_id_mismatch"
    return True, ""
