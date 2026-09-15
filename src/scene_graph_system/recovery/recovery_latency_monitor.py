#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from scene_graph_system.resources import runtime_data_path

import json
import os
import select
import sys
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

import rospy
from std_msgs.msg import String

from scene_graph_system.diagnosis.fault_event import TOPIC_DEVIATION_ALERT, TOPIC_RECOVERY_PLAN

try:
    import termios
    import tty
except ImportError:  # pragma: no cover - only used outside Linux terminals
    termios = None
    tty = None


_BEIJING_TIMEZONE = timezone(timedelta(hours=8))
_RECORD_PATH = runtime_data_path("recovery_latency.txt")


def _as_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_time(value) -> str:
    stamp = _as_float(value)
    if stamp is None:
        return "unavailable"
    return datetime.fromtimestamp(
        stamp,
        tz=_BEIJING_TIMEZONE,
    ).isoformat(sep=" ", timespec="microseconds")


def format_result(
    injection_stamp: float,
    detection_stamp: float,
    recovery_stamp: float,
) -> str:
    return (
        "[TIMING][RESULT]\n"
        "故障注入时间：%s\n"
        "首次告警时间：%s\n"
        "恢复触发时间：%s\n"
        "故障检测延迟：%.6f 秒\n"
        "故障至恢复触发延迟：%.6f 秒\n"
        "检测至恢复触发延迟：%.6f 秒"
        % (
            _format_time(injection_stamp),
            _format_time(detection_stamp),
            _format_time(recovery_stamp),
            detection_stamp - injection_stamp,
            recovery_stamp - injection_stamp,
            recovery_stamp - detection_stamp,
        )
    )


class TimingSession:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._injection_stamp = None
        self._detection_stamp = None

    def record_injection(self, stamp) -> None:
        injection_stamp = _as_float(stamp)
        if injection_stamp is None:
            raise ValueError("injection timestamp must be numeric")
        with self._lock:
            self._injection_stamp = injection_stamp
            self._detection_stamp = None

    def record_alert(self, payload: dict, receipt_stamp=None) -> bool:
        alert_stamp = _as_float(payload.get("timestamp", receipt_stamp))
        if alert_stamp is None:
            return False
        with self._lock:
            if self._injection_stamp is None:
                return False
            if self._detection_stamp is not None:
                return False
            if alert_stamp < self._injection_stamp:
                return False
            self._detection_stamp = alert_stamp
            return True

    def record_recovery(self, payload: dict, receipt_stamp=None) -> Optional[str]:
        recovery_stamp = _as_float(payload.get("timestamp", receipt_stamp))
        if recovery_stamp is None:
            return None
        with self._lock:
            if self._injection_stamp is None or self._detection_stamp is None:
                return None
            if recovery_stamp < self._detection_stamp:
                return None
            injection_stamp = self._injection_stamp
            detection_stamp = self._detection_stamp
            self._injection_stamp = None
            self._detection_stamp = None
        return format_result(
            injection_stamp,
            detection_stamp,
            recovery_stamp,
        )


def _record_separator(record_path: str) -> str:
    try:
        size = os.path.getsize(record_path)
    except OSError:
        return ""
    if size <= 0:
        return ""
    with open(record_path, "rb") as record_file:
        record_file.seek(-min(size, 2), os.SEEK_END)
        tail = record_file.read()
    if tail.endswith(b"\n\n"):
        return ""
    if tail.endswith(b"\n"):
        return "\n"
    return "\n\n"


def append_record(record_path: str, record: str) -> None:
    record_dir = os.path.dirname(record_path)
    if record_dir:
        os.makedirs(record_dir, exist_ok=True)
    separator = _record_separator(record_path)
    with open(record_path, "a", encoding="utf-8") as record_file:
        record_file.write(separator)
        record_file.write(record.rstrip("\n"))
        record_file.write("\n")


def record_key(session: TimingSession, key: str, stamp) -> bool:
    if key != " ":
        return False
    session.record_injection(stamp)
    return True


def _message_payload(msg: String):
    try:
        payload = json.loads(str(msg.data or ""))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


_SESSION = TimingSession()


def _alert_callback(msg: String) -> None:
    payload = _message_payload(msg)
    if payload is None:
        return
    receipt_stamp = float(rospy.Time.now().to_sec())
    _SESSION.record_alert(payload, receipt_stamp=receipt_stamp)


def _recovery_callback(msg: String) -> None:
    payload = _message_payload(msg)
    if payload is None:
        return
    receipt_stamp = float(rospy.Time.now().to_sec())
    record = _SESSION.record_recovery(payload, receipt_stamp=receipt_stamp)
    if record is None:
        return
    print(record + "\n", flush=True)
    try:
        append_record(_RECORD_PATH, record)
    except OSError as exc:
        rospy.logerr("[TIMING] failed to append record to %s: %s", _RECORD_PATH, exc)


def _keyboard_loop() -> None:
    if termios is None or tty is None or not sys.stdin.isatty():
        rospy.logerr("[TIMING] stdin is not a Linux terminal; space-key capture unavailable")
        return

    input_fd = sys.stdin.fileno()
    previous_settings = termios.tcgetattr(input_fd)
    try:
        tty.setcbreak(input_fd)
        while not rospy.is_shutdown():
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not readable:
                continue
            key = sys.stdin.read(1)
            injection_stamp = float(rospy.Time.now().to_sec())
            if not record_key(_SESSION, key, injection_stamp):
                continue
            print(
                "[TIMING] 已记录故障注入，等待首次告警和恢复触发。",
                flush=True,
            )
    except Exception as exc:
        if not rospy.is_shutdown():
            rospy.logerr("[TIMING] keyboard capture failed: %s", exc)
    finally:
        termios.tcsetattr(input_fd, termios.TCSADRAIN, previous_settings)


def main() -> None:
    rospy.init_node("recovery_latency_monitor", anonymous=True)
    rospy.Subscriber(
        TOPIC_DEVIATION_ALERT,
        String,
        _alert_callback,
        queue_size=50,
    )
    rospy.Subscriber(
        TOPIC_RECOVERY_PLAN,
        String,
        _recovery_callback,
        queue_size=20,
    )
    keyboard_thread = threading.Thread(
        target=_keyboard_loop,
        name="recovery_latency_keyboard",
        daemon=True,
    )
    keyboard_thread.start()
    print("[TIMING] 按空格记录故障注入时间。", flush=True)
    try:
        rospy.spin()
    finally:
        keyboard_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
