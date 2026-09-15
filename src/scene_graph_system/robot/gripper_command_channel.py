#!/usr/bin/env python3
from __future__ import annotations

import math
import threading
import time
import uuid
from typing import Any, Dict, Optional, Tuple

import rospy

from scene_graph_system.msg import GripperCommand, GripperState


TOPIC_GRIPPER_COMMAND = "/task_execution/gripper_command"
TOPIC_GRIPPER_STATE = "/gripper/state"
GRIPPER_PROTOCOL_VERSION = 1

_publisher = None
_publisher_topic = None
_publisher_lock = threading.Lock()


def _int_or(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _publisher_for(topic: str):
    global _publisher, _publisher_topic
    topic = str(topic or TOPIC_GRIPPER_COMMAND)
    with _publisher_lock:
        if _publisher is None or _publisher_topic != topic:
            _publisher = rospy.Publisher(topic, GripperCommand, queue_size=20, latch=False)
            _publisher_topic = topic
        return _publisher


def publish_gripper_command(
    command: str,
    *,
    source: str,
    context: Optional[Dict[str, Any]] = None,
    requested_position: int = -1,
    requested_speed: int = -1,
    requested_force: int = -1,
    topic: str = TOPIC_GRIPPER_COMMAND,
) -> Tuple[str, float]:
    command = str(command or "").strip().lower()
    if command not in {"open", "close"}:
        raise ValueError("command must be open or close")

    ctx = dict(context or {})
    stamp = rospy.Time.now()
    command_id = "gripper_%s_%s" % (command, uuid.uuid4().hex[:12])
    msg = GripperCommand()
    msg.header.stamp = stamp
    msg.schema_version = GRIPPER_PROTOCOL_VERSION
    msg.command_id = command_id
    msg.source = str(source or "unknown")
    msg.run_id = str(ctx.get("run_id", "") or "")
    msg.attempt_id = _int_or(ctx.get("attempt_id"))
    msg.step = _int_or(ctx.get("step"))
    msg.stage_seq = _int_or(ctx.get("stage_seq"))
    msg.actuation_seq = _int_or(ctx.get("actuation_seq"))
    msg.stage_name = str(ctx.get("stage_name", "") or "")
    msg.command = command
    msg.requested_position = _int_or(requested_position)
    msg.requested_speed = _int_or(requested_speed)
    msg.requested_force = _int_or(requested_force)
    publisher = _publisher_for(topic)
    # The state monitor is a required consumer.  Give the newly-created ROS
    # publisher a short bounded window to establish its first connection so
    # the command context is not lost before the actuator command is sent.
    deadline = time.monotonic() + 0.5
    while not rospy.is_shutdown() and publisher.get_num_connections() <= 0:
        if time.monotonic() >= deadline:
            raise RuntimeError("gripper command topic has no subscriber")
        rospy.sleep(0.01)
    publisher.publish(msg)
    return command_id, float(stamp.to_sec())


def gripper_state_is_fresh(msg: GripperState, *, max_age_sec: float) -> Tuple[bool, str]:
    if int(getattr(msg, "schema_version", 0)) != GRIPPER_PROTOCOL_VERSION:
        return False, "schema_version_mismatch"
    if not bool(getattr(msg, "valid", False)):
        return False, "state_invalid"
    sample_end = float(msg.sample_end_stamp.to_sec())
    age = float(rospy.Time.now().to_sec()) - sample_end
    if age < -0.5:
        return False, "state_from_future"
    if age > max(0.0, float(max_age_sec)):
        return False, "state_stale"
    return True, ""


def gripper_monitor_sample_is_current(
    msg: GripperState,
    *,
    max_age_sec: float,
) -> Tuple[bool, str]:
    """Check monitor liveness without requiring a valid physical state."""
    if int(getattr(msg, "schema_version", 0)) != GRIPPER_PROTOCOL_VERSION:
        return False, "schema_version_mismatch"
    sample_end = float(msg.sample_end_stamp.to_sec())
    if sample_end <= 0.0:
        return False, "sample_stamp_missing"
    age = float(rospy.Time.now().to_sec()) - sample_end
    if age < -0.5:
        return False, "state_from_future"
    if age > max(0.0, float(max_age_sec)):
        return False, "state_stale"
    return True, ""


def wait_for_gripper_monitor_sample(
    *,
    timeout_sec: float,
    max_age_sec: float,
    topic: str = TOPIC_GRIPPER_STATE,
) -> Tuple[Optional[GripperState], str]:
    """Wait for a current monitor sample, allowing ``valid=False``.

    This proves that the monitor process and its ROS publisher are alive before
    startup initialization.  It deliberately does not claim that the SDK state
    is usable yet; a cold RealMan gripper may report an all-zero invalid state
    until its first control command.
    """
    deadline = time.monotonic() + max(0.1, float(timeout_sec))
    last_reason = "no_state"
    received_sample = False
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        remaining = max(0.05, deadline - time.monotonic())
        try:
            msg = rospy.wait_for_message(topic, GripperState, timeout=min(0.5, remaining))
        except rospy.ROSException:
            if not received_sample:
                last_reason = "state_timeout"
            continue
        received_sample = True
        ok, reason = gripper_monitor_sample_is_current(msg, max_age_sec=max_age_sec)
        if ok:
            return msg, ""
        last_reason = reason
    return None, last_reason


def gripper_state_matches_startup_requirement(
    msg: GripperState,
    *,
    max_age_sec: float,
    required_state: str,
    min_stable_count: int,
    command_id: str = "",
    not_before: float = 0.0,
) -> Tuple[bool, str]:
    """Validate a stable startup state and, when supplied, its command context."""
    ok, reason = gripper_state_is_fresh(msg, max_age_sec=max_age_sec)
    if not ok:
        return False, reason
    if int(getattr(msg, "error_code", -1)) != 0:
        return False, "gripper_error"
    if float(not_before) > 0.0:
        if float(msg.sample_start_stamp.to_sec()) < float(not_before):
            return False, "sample_before_command"
    expected_command_id = str(command_id or "").strip()
    if expected_command_id and str(getattr(msg, "command_id", "") or "") != expected_command_id:
        return False, "command_id_mismatch"
    required_state = str(required_state or "").strip().lower()
    states = {str(value).strip().lower() for value in list(getattr(msg, "states", []) or [])}
    if required_state and required_state not in states:
        return False, "required_state_missing"
    if int(getattr(msg, "stable_count", 0)) < max(1, int(min_stable_count)):
        return False, "state_not_stable"
    return True, ""


def wait_for_gripper_startup_state(
    *,
    timeout_sec: float,
    max_age_sec: float,
    required_state: str,
    min_stable_count: int,
    command_id: str = "",
    not_before: float = 0.0,
    topic: str = TOPIC_GRIPPER_STATE,
) -> Tuple[Optional[GripperState], str]:
    """Wait for a fresh, stable startup state, optionally tied to one command."""
    deadline = time.monotonic() + max(0.1, float(timeout_sec))
    last_reason = "no_state"
    received_sample = False
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        remaining = max(0.05, deadline - time.monotonic())
        try:
            msg = rospy.wait_for_message(topic, GripperState, timeout=min(0.5, remaining))
        except rospy.ROSException:
            if not received_sample:
                last_reason = "state_timeout"
            continue
        received_sample = True
        ok, reason = gripper_state_matches_startup_requirement(
            msg,
            max_age_sec=max_age_sec,
            required_state=required_state,
            min_stable_count=min_stable_count,
            command_id=command_id,
            not_before=not_before,
        )
        if ok:
            return msg, ""
        last_reason = reason
    return None, last_reason


def wait_for_fresh_gripper_state(
    *,
    timeout_sec: float,
    max_age_sec: float,
    topic: str = TOPIC_GRIPPER_STATE,
) -> Tuple[Optional[GripperState], str]:
    deadline = time.monotonic() + max(0.1, float(timeout_sec))
    last_reason = "no_state"
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        remaining = max(0.05, deadline - time.monotonic())
        try:
            msg = rospy.wait_for_message(topic, GripperState, timeout=min(0.5, remaining))
        except rospy.ROSException:
            last_reason = "state_timeout"
            continue
        ok, reason = gripper_state_is_fresh(msg, max_age_sec=max_age_sec)
        if ok:
            return msg, ""
        last_reason = reason
    return None, last_reason
