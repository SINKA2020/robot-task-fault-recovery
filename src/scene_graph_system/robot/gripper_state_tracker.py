from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_GRIPPER_NODE_ID = "gripper"
REALMAN_OPEN_POSITION_THRESHOLD = 900.0
REALMAN_FULLY_CLOSED_POSITION_THRESHOLD = 50.0
REALMAN_HOLDING_FORCE_THRESHOLD = 1.0


def _normalize_text(value: Any) -> str:
    return str(value).strip().lower()


def _first_present(payload: Dict[str, Any], keys: Iterable[str]) -> Tuple[Optional[str], Any]:
    for key in keys:
        if key in payload:
            return key, payload[key]
    return None, None


def _coerce_boolish(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = _normalize_text(value)
        if normalized in {"1", "true", "yes", "on", "open", "opened", "closed", "close", "moving", "running", "holding", "grasped"}:
            return True
        if normalized in {"0", "false", "no", "off", "idle", "unknown", "none"}:
            return False
    try:
        return bool(float(value))
    except (TypeError, ValueError):
        return None


def _coerce_floatish(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _payload_to_dict(payload: Any) -> Dict[str, Any]:
    if payload is None:
        return {}

    def _store_scalar(out: Dict[str, Any], key: Any, value: Any) -> None:
        if not isinstance(value, (bool, int, float, str)):
            return
        key_str = str(key).strip()
        if not key_str:
            return
        if key_str not in out:
            out[key_str] = value
        lower_key = key_str.lower()
        if lower_key not in out:
            out[lower_key] = value

    if isinstance(payload, dict):
        extracted: Dict[str, Any] = {}
        for key, value in payload.items():
            _store_scalar(extracted, key, value)
        return extracted

    items_fn = getattr(payload, "items", None)
    if callable(items_fn):
        try:
            extracted: Dict[str, Any] = {}
            for key, value in dict(items_fn()).items():
                _store_scalar(extracted, key, value)
            if extracted:
                return extracted
        except Exception:
            pass

    raw_dict = getattr(payload, "__dict__", None)
    if isinstance(raw_dict, dict) and raw_dict:
        extracted: Dict[str, Any] = {}
        for key, value in raw_dict.items():
            _store_scalar(extracted, key, value)
        if extracted:
            return extracted

    extracted: Dict[str, Any] = {}

    candidate_keys = (
        "enable", "enabled", "enable_state", "is_enable", "is_enabled",
        "status", "state", "gripper_status", "gripper_state", "grip_state",
        "mode", "gripper_mode",
        "current_force", "force", "gripper_force",
        "actpos", "actual_position", "position", "gripper_position",
        "open", "opened", "is_open", "is_opened",
        "closed", "close", "is_closed", "clamped",
        "holding", "is_holding", "grasped", "is_grasped", "is_gripping",
        "object_detected", "has_object",
        "moving", "is_moving", "in_motion", "running", "busy", "is_running",
        "error", "error_code", "err_code", "fault", "fault_code", "alarm",
    )

    for key in candidate_keys:
        if hasattr(payload, key):
            try:
                value = getattr(payload, key)
            except Exception:
                continue
            _store_scalar(extracted, key, value)

    # 兜底：扫描所有公共属性
    for key in dir(payload):
        if key.startswith("_"):
            continue
        if key in extracted or key.lower() in extracted:
            continue

        try:
            value = getattr(payload, key)
        except Exception:
            continue

        if callable(value):
            continue

        if isinstance(value, (bool, int, float, str)):
            _store_scalar(extracted, key, value)
            continue

        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                _store_scalar(extracted, sub_key, sub_value)
                _store_scalar(extracted, f"{key}.{sub_key}", sub_value)
            continue

        sub_dict = getattr(value, "__dict__", None)
        if isinstance(sub_dict, dict):
            for sub_key, sub_value in sub_dict.items():
                _store_scalar(extracted, sub_key, sub_value)
                _store_scalar(extracted, f"{key}.{sub_key}", sub_value)

    return extracted


def _infer_open_closed_from_text(payload: Dict[str, Any]) -> Tuple[Optional[bool], Optional[bool]]:
    _, raw_value = _first_present(
        payload,
        ("state", "status", "gripper_state", "gripper_status", "grip_state"),
    )
    if raw_value is None:
        return None, None

    text = _normalize_text(raw_value)
    is_open = text in {"open", "opened", "release", "released"}
    is_closed = text in {"closed", "close", "closing", "grasped", "holding", "clamped"}
    return is_open, is_closed


def _infer_open_closed_from_realman_position(payload: Dict[str, Any]) -> Tuple[Optional[bool], Optional[bool]]:
    _, raw_value = _first_present(payload, ("actpos", "actual_position", "position", "gripper_position"))
    position_value = _coerce_floatish(raw_value)
    if position_value is None:
        return None, None

    if position_value >= REALMAN_OPEN_POSITION_THRESHOLD:
        return True, False

    if position_value <= REALMAN_FULLY_CLOSED_POSITION_THRESHOLD:
        return False, True

    # RealMan open uses position=1000; an intermediate actpos usually means
    # the jaws stopped early on an object rather than reaching full close.
    return False, True


def infer_realman_gripper_state_from_position(position, force=None):
    """
    从夹爪位置推断夹爪状态（公开 helper，供 repair 等模块复用）。

    返回:
      {
        "position": float,
        "states": ["open"] | ["closed"] | ["closed", "holding"],
        "open": bool,
        "closed": bool,
        "holding": bool,
      }
    """
    payload = {"position": float(position)}
    is_open, is_closed = _infer_open_closed_from_realman_position(payload)

    states = []
    if is_open:
        states.append("open")
    if is_closed:
        states.append("closed")

    pos = float(position)
    force_value = _coerce_floatish(force)
    if (
        REALMAN_FULLY_CLOSED_POSITION_THRESHOLD < pos < REALMAN_OPEN_POSITION_THRESHOLD
        and force_value is not None
        and abs(force_value) >= REALMAN_HOLDING_FORCE_THRESHOLD
    ):
        states.append("holding")

    return {
        "position": pos,
        "states": states,
        "open": "open" in states,
        "closed": "closed" in states,
        "holding": "holding" in states,
    }


def _infer_holding_from_realman_payload(payload: Dict[str, Any]) -> Optional[bool]:
    _, raw_position = _first_present(payload, ("actpos", "actual_position", "position", "gripper_position"))
    position_value = _coerce_floatish(raw_position)
    _, raw_force = _first_present(payload, ("current_force", "force", "gripper_force"))
    force_value = _coerce_floatish(raw_force)

    if position_value is None:
        # RealMan ``status`` only means online/offline.  Force without an
        # opening measurement is not sufficient to prove that an object is
        # between the jaws.
        return None

    if position_value >= REALMAN_OPEN_POSITION_THRESHOLD:
        return False

    if position_value <= REALMAN_FULLY_CLOSED_POSITION_THRESHOLD:
        return False

    return bool(
        force_value is not None
        and abs(force_value) >= REALMAN_HOLDING_FORCE_THRESHOLD
    )

def _infer_realman_states_fastpath(payload: Dict[str, Any]) -> Optional[List[str]]:
    """
    Deterministic RealMan-specific normalization.

    Expected behavior for current SDK payload:
    - actpos >= 900  -> open
    - actpos <= 50   -> closed
    - 50 < actpos < 900 -> closed; holding additionally requires contact force
    """
    _, raw_position = _first_present(payload, ("actpos", "actual_position", "position", "gripper_position"))
    position_value = _coerce_floatish(raw_position)
    if position_value is None:
        return None

    _, raw_error = _first_present(payload, ("error", "error_code", "err_code", "fault", "fault_code", "alarm"))
    error_value = _coerce_floatish(raw_error)

    _, raw_enable = _first_present(payload, ("enable", "enabled", "enable_state", "is_enable", "is_enabled"))
    enable_value = _coerce_floatish(raw_enable)

    _, raw_status = _first_present(payload, ("status", "gripper_status", "state"))
    status_value = _coerce_floatish(raw_status)

    _, raw_force = _first_present(payload, ("current_force", "force", "gripper_force"))
    force_value = _coerce_floatish(raw_force)

    states: List[str] = []

    if enable_value is not None and enable_value <= 0:
        states.append("offline")

    # Per the RealMan API, status is 0=offline and 1=online.  It is not a
    # motion or grasp-result code and must never be used as holding evidence.
    if status_value is not None and status_value <= 0 and "offline" not in states:
        states.append("offline")

    if error_value is not None and error_value != 0:
        states.append("fault")

    # RealMan open position is typically around 1000
    if position_value >= REALMAN_OPEN_POSITION_THRESHOLD:
        if "open" not in states:
            states.append("open")
        return states

    # Fully closed without object
    if position_value <= REALMAN_FULLY_CLOSED_POSITION_THRESHOLD:
        if "closed" not in states:
            states.append("closed")
        return states

    # Intermediate position: usually stopped on an object
    if "closed" not in states:
        states.append("closed")

    holding = False
    if force_value is not None and abs(force_value) >= REALMAN_HOLDING_FORCE_THRESHOLD:
        holding = True

    if holding and "holding" not in states:
        states.append("holding")

    return states

def normalize_gripper_state(
    api_status_code: int,
    payload: Optional[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Convert rm_get_gripper_state() output into stable scene-graph states.

    The RealMan SDK may expose slightly different field names across versions,
    so this function uses tolerant key matching and preserves the raw payload.
    """
    raw_payload = _payload_to_dict(payload)
    attributes: Dict[str, Any] = dict(raw_payload)
    attributes["api_status_code"] = api_status_code
    attributes["payload_keys"] = sorted(raw_payload.keys())
    if payload is not None and not isinstance(payload, dict):
        attributes["raw_payload_repr"] = repr(payload)

    # ----- Deterministic RealMan fastpath -----
    fastpath_states = _infer_realman_states_fastpath(raw_payload)
    if api_status_code == 0 and fastpath_states is not None:
        deduped_fastpath: List[str] = []
        for state in fastpath_states:
            if state not in deduped_fastpath:
                deduped_fastpath.append(state)
        attributes["normalized_states"] = list(deduped_fastpath)
        attributes["normalization_source"] = "realman_fastpath"
        return deduped_fastpath, attributes

    states: List[str] = []

    if api_status_code != 0:
        states.append("fault")

    error_key, error_value = _first_present(
        raw_payload,
        ("error", "error_code", "err_code", "fault", "fault_code", "alarm"),
    )
    if error_key is not None:
        error_flag = _coerce_boolish(error_value)
        if error_flag is True:
            states.append("fault")
        elif isinstance(error_value, (int, float)) and error_value != 0:
            states.append("fault")

    _, enabled_value = _first_present(
        raw_payload,
        ("enable", "enabled", "is_enable", "is_enabled", "online", "is_online", "enable_state"),
    )
    enabled_flag = _coerce_boolish(enabled_value)
    if enabled_flag is False:
        states.append("offline")

    _, moving_value = _first_present(
        raw_payload,
        ("moving", "is_moving", "in_motion", "running", "busy", "is_running"),
    )
    moving_flag = _coerce_boolish(moving_value)
    if moving_flag is True:
        states.append("moving")

    _, holding_value = _first_present(
        raw_payload,
        (
            "holding",
            "is_holding",
            "grasped",
            "is_grasped",
            "is_gripping",
            "object_detected",
            "has_object",
        ),
    )
    holding_flag = _coerce_boolish(holding_value)
    if holding_flag is None:
        holding_flag = _infer_holding_from_realman_payload(raw_payload)
    if holding_flag is True:
        states.append("holding")

    _, open_value = _first_present(raw_payload, ("open", "opened", "is_open", "is_opened"))
    _, closed_value = _first_present(raw_payload, ("closed", "is_closed", "close", "clamped"))
    open_flag = _coerce_boolish(open_value)
    closed_flag = _coerce_boolish(closed_value)

    text_open, text_closed = _infer_open_closed_from_text(raw_payload)
    realman_open, realman_closed = _infer_open_closed_from_realman_position(raw_payload)
    if open_flag is None:
        open_flag = text_open
    if open_flag is None:
        open_flag = realman_open
    if closed_flag is None:
        closed_flag = text_closed
    if closed_flag is None:
        closed_flag = realman_closed

    if open_flag is True and "open" not in states:
        states.append("open")
    if closed_flag is True and "closed" not in states:
        states.append("closed")

    if "holding" in states and "closed" not in states:
        states.append("closed")

    if not states:
        states.append("unknown")

    deduped_states: List[str] = []
    for state in states:
        if state not in deduped_states:
            deduped_states.append(state)

    attributes["normalized_states"] = list(deduped_states)
    return deduped_states, attributes


class RealmanGripperStateProvider:
    def __init__(self, arm: Any):
        self.arm = arm

    def fetch(self) -> Tuple[int, Dict[str, Any]]:
        return self.arm.rm_get_gripper_state()

    def __call__(self) -> Tuple[int, Dict[str, Any]]:
        return self.fetch()


def create_realman_gripper_state_provider(
    robot_ip: str,
    port: int = 8080,
    thread_mode_name: str = "RM_TRIPLE_MODE_E",
) -> Tuple[RealmanGripperStateProvider, Any]:
    """
    Create a gripper-state provider backed by the RealMan Python SDK.

    Returns:
        (provider, arm)

    The caller owns the returned arm instance and should invoke
    arm.rm_delete_robot_arm() during teardown.
    """
    from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

    if not robot_ip or not str(robot_ip).strip():
        raise ValueError("robot_ip is required")

    try:
        thread_mode = getattr(rm_thread_mode_e, thread_mode_name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported thread mode: {thread_mode_name}") from exc

    arm = RoboticArm(thread_mode)
    handle = arm.rm_create_robot_arm(robot_ip, int(port))
    handle_id = getattr(handle, "id", None)
    if handle_id in (None, -1):
        try:
            arm.rm_delete_robot_arm()
        except Exception:
            pass
        raise RuntimeError(
            f"Failed to create RealMan robot arm connection to {robot_ip}:{port}"
        )

    return RealmanGripperStateProvider(arm), arm
