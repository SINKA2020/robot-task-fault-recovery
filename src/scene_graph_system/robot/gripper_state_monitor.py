#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
import uuid
from typing import Any, Dict, Optional

import rospy

from scene_graph_system.msg import GripperCommand, GripperState

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))



from scene_graph_system.robot.gripper_state_protocol import (
    GRIPPER_PROTOCOL_VERSION,
    GripperCommandContext,
    GripperStateStabilizer,
    normalize_supported_states,
)
from scene_graph_system.robot.gripper_state_tracker import create_realman_gripper_state_provider, normalize_gripper_state
from scene_graph_system.robot.gripper_command_channel import TOPIC_GRIPPER_COMMAND, TOPIC_GRIPPER_STATE


DEFAULT_REALMAN_IP = "192.168.0.18"
DEFAULT_REALMAN_PORT = 8080
DEFAULT_REALMAN_THREAD_MODE = "RM_TRIPLE_MODE_E"


def _float_value(payload: Dict[str, Any], names, default=math.nan) -> float:
    for name in names:
        if name in payload and payload[name] is not None:
            try:
                return float(payload[name])
            except (TypeError, ValueError):
                continue
    return float(default)


def _int_value(payload: Dict[str, Any], names, default=-1) -> int:
    value = _float_value(payload, names, default=float(default))
    if math.isnan(value):
        return int(default)
    return int(value)


class GripperStateMonitor:
    def __init__(self):
        self.robot_ip = str(rospy.get_param("~realman_ip", DEFAULT_REALMAN_IP)).strip()
        self.robot_port = int(rospy.get_param("~realman_port", DEFAULT_REALMAN_PORT))
        self.thread_mode = str(
            rospy.get_param("~realman_thread_mode", DEFAULT_REALMAN_THREAD_MODE)
        ).strip()
        self.poll_hz = max(0.5, float(rospy.get_param("~poll_hz", 10.0)))
        self.stable_count_required = max(1, int(rospy.get_param("~stable_count", 3)))
        self.reconnect_interval_sec = max(
            0.5, float(rospy.get_param("~reconnect_interval_sec", 2.0))
        )
        self.command_topic = str(
            rospy.get_param("~command_topic", TOPIC_GRIPPER_COMMAND)
        ).strip()
        self.state_topic = str(rospy.get_param("~state_topic", TOPIC_GRIPPER_STATE)).strip()

        self.monitor_instance_id = "gripper_monitor_%s" % uuid.uuid4().hex[:12]
        self.sample_seq = 0
        self.provider = None
        self.arm = None
        self.last_connect_attempt_monotonic = 0.0
        self.command_lock = threading.Lock()
        self.latest_command: Optional[GripperCommandContext] = None
        self.latest_command_source = ""
        self.stabilizer = GripperStateStabilizer(self.stable_count_required)

        self.state_pub = rospy.Publisher(
            self.state_topic, GripperState, queue_size=50, latch=False
        )
        self.command_sub = rospy.Subscriber(
            self.command_topic, GripperCommand, self._command_callback, queue_size=50
        )
        rospy.on_shutdown(self.shutdown)

    def _command_callback(self, msg: GripperCommand) -> None:
        if int(getattr(msg, "schema_version", 0)) != GRIPPER_PROTOCOL_VERSION:
            rospy.logwarn_throttle(2.0, "[GRIPPER] rejected command with unsupported schema")
            return
        try:
            command = GripperCommandContext(
                command_id=msg.command_id,
                command=msg.command,
                command_stamp=float(msg.header.stamp.to_sec()),
                source=msg.source,
                run_id=msg.run_id,
                attempt_id=int(msg.attempt_id),
                step=int(msg.step),
                stage_seq=int(msg.stage_seq),
                actuation_seq=int(msg.actuation_seq),
                stage_name=msg.stage_name,
            )
        except Exception as exc:
            rospy.logwarn("[GRIPPER] rejected invalid command: %s", str(exc))
            return
        with self.command_lock:
            current = self.latest_command
            if current is not None and command.command_stamp < current.command_stamp:
                return
            self.latest_command = command
            self.latest_command_source = str(msg.source or "")
            self.stabilizer.reset(clear_latch=False)

    def _disconnect(self) -> None:
        arm = self.arm
        self.provider = None
        self.arm = None
        if arm is not None:
            try:
                arm.rm_delete_robot_arm()
            except Exception:
                pass

    def _connect_if_due(self) -> bool:
        if self.provider is not None:
            return True
        now = time.monotonic()
        if now - self.last_connect_attempt_monotonic < self.reconnect_interval_sec:
            return False
        self.last_connect_attempt_monotonic = now
        try:
            self.provider, self.arm = create_realman_gripper_state_provider(
                self.robot_ip, self.robot_port, self.thread_mode
            )
            rospy.loginfo(
                "[GRIPPER] connected to RealMan state provider %s:%d",
                self.robot_ip,
                self.robot_port,
            )
            return True
        except Exception as exc:
            self._disconnect()
            rospy.logwarn_throttle(2.0, "[GRIPPER] provider connection failed: %s", str(exc))
            return False

    def _command_snapshot(self) -> Optional[GripperCommandContext]:
        with self.command_lock:
            return self.latest_command

    def _publish_sample(
        self,
        *,
        start_stamp,
        end_stamp,
        api_status_code: int,
        payload: Optional[Dict[str, Any]],
        states,
        valid: bool,
        reason: str,
    ) -> None:
        self.sample_seq += 1
        raw_payload = dict(payload or {}) if isinstance(payload, dict) else {}
        supported_states = normalize_supported_states(states, valid=valid)
        with self.command_lock:
            _, stable_count, holding_latched = self.stabilizer.update(
                supported_states, valid=valid
            )
            command = self.latest_command

        msg = GripperState()
        msg.header.seq = int(self.sample_seq) & 0xFFFFFFFF
        msg.header.stamp = end_stamp
        msg.schema_version = GRIPPER_PROTOCOL_VERSION
        msg.monitor_instance_id = self.monitor_instance_id
        msg.sample_seq = int(self.sample_seq)
        msg.sample_start_stamp = start_stamp
        msg.sample_end_stamp = end_stamp
        msg.valid = bool(valid)
        msg.api_status_code = int(api_status_code)
        msg.position = _float_value(
            raw_payload, ("actpos", "actual_position", "position", "gripper_position")
        )
        msg.force = _float_value(
            raw_payload, ("current_force", "force", "gripper_force")
        )
        msg.status_code = _int_value(raw_payload, ("status", "gripper_status", "state"))
        msg.mode_code = _int_value(raw_payload, ("mode", "gripper_mode"))
        msg.error_code = _int_value(
            raw_payload, ("error", "error_code", "err_code", "fault", "fault_code", "alarm")
        )
        msg.states = list(supported_states)
        msg.stable_count = int(stable_count)
        msg.holding_latched = bool(holding_latched)
        if command is not None and float(start_stamp.to_sec()) >= command.command_stamp:
            msg.command_id = command.command_id
            msg.command = command.command
            msg.command_source = command.source
            msg.run_id = command.run_id
            msg.attempt_id = command.attempt_id
            msg.step = command.step
            msg.stage_seq = command.stage_seq
            msg.actuation_seq = command.actuation_seq
            msg.stage_name = command.stage_name
            msg.command_stamp = rospy.Time.from_sec(command.command_stamp)
        else:
            msg.attempt_id = -1
            msg.step = -1
            msg.stage_seq = -1
            msg.actuation_seq = -1
            msg.command_stamp = rospy.Time(0)
        msg.reason = str(reason or "")
        try:
            msg.raw_json = json.dumps(raw_payload, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            msg.raw_json = repr(payload)
        self.state_pub.publish(msg)

    def poll_once(self) -> None:
        start_stamp = rospy.Time.now()
        if not self._connect_if_due():
            self._publish_sample(
                start_stamp=start_stamp,
                end_stamp=rospy.Time.now(),
                api_status_code=-1,
                payload={},
                states=["unknown"],
                valid=False,
                reason="provider_unavailable",
            )
            return
        try:
            status_code, payload = self.provider.fetch()
            end_stamp = rospy.Time.now()
            states, attributes = normalize_gripper_state(status_code, payload)
            invalid_states = {"unknown", "fault", "offline"}
            valid = int(status_code) == 0 and not bool(set(states) & invalid_states)
            reason = "" if valid else "provider_state_invalid"
            self._publish_sample(
                start_stamp=start_stamp,
                end_stamp=end_stamp,
                api_status_code=int(status_code),
                payload=attributes,
                states=states,
                valid=valid,
                reason=reason,
            )
        except Exception as exc:
            end_stamp = rospy.Time.now()
            self._disconnect()
            self._publish_sample(
                start_stamp=start_stamp,
                end_stamp=end_stamp,
                api_status_code=-1,
                payload={},
                states=["unknown"],
                valid=False,
                reason="provider_exception:%s" % str(exc),
            )

    def run(self) -> None:
        rospy.loginfo(
            "[GRIPPER] monitor started: topic=%s poll_hz=%.2f stable_count=%d instance=%s",
            self.state_topic,
            self.poll_hz,
            self.stable_count_required,
            self.monitor_instance_id,
        )
        rate = rospy.Rate(self.poll_hz)
        while not rospy.is_shutdown():
            self.poll_once()
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break

    def shutdown(self) -> None:
        self._disconnect()


def main() -> None:
    rospy.init_node("gripper_state_monitor", anonymous=False)
    GripperStateMonitor().run()


if __name__ == "__main__":
    main()
