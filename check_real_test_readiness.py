#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

import importlib
import json
import socket
import time
from typing import Dict, List, Tuple

import rospy
from std_msgs.msg import String
from scene_graph_system.msg import GripperState

from scene_graph_system.resources import find_package_resource



def import_check(module_name: str) -> Tuple[bool, str]:
    try:
        importlib.import_module(module_name)
        return True, 'ok'
    except Exception as exc:
        return False, str(exc)


def file_check(path_value: str) -> Tuple[bool, str]:
    if path_value and os.path.exists(path_value):
        return True, path_value
    return False, f'missing: {path_value}'


def tcp_check(host: str, port: int, timeout: float = 2.0) -> Tuple[bool, str]:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True, f'{host}:{port}'
    except Exception as exc:
        return False, str(exc)


def topic_presence_check(required_topics: List[str]) -> Dict[str, bool]:
    published_topics = dict(rospy.get_published_topics())
    return {topic: topic in published_topics for topic in required_topics}


def json_topic_fields_check(topic: str, required_fields: List[str], timeout: float) -> Tuple[bool, str]:
    try:
        msg = rospy.wait_for_message(topic, String, timeout=float(timeout))
        payload = json.loads(str(msg.data or "{}"))
    except Exception as exc:
        return False, "%s: %s" % (topic, str(exc))
    missing = [field for field in required_fields if payload.get(field) is None]
    if missing:
        return False, "%s missing fields: %s" % (topic, ",".join(missing))
    return True, "%s schema ok" % topic


def gripper_state_check(
    topic: str,
    timeout: float,
    max_age_sec: float,
    min_stable_count: int,
) -> Tuple[bool, str]:
    try:
        msg = rospy.wait_for_message(topic, GripperState, timeout=float(timeout))
    except Exception as exc:
        return False, "%s: %s" % (topic, str(exc))
    if int(msg.schema_version) != 1:
        return False, "%s schema_version=%s" % (topic, str(msg.schema_version))
    if not bool(msg.valid):
        return False, "%s invalid: %s" % (topic, str(msg.reason))
    age_sec = float(rospy.Time.now().to_sec()) - float(msg.sample_end_stamp.to_sec())
    if age_sec < -0.5 or age_sec > float(max_age_sec):
        return False, "%s stale age=%.3f" % (topic, age_sec)
    if int(msg.stable_count) < int(min_stable_count):
        return False, "%s unstable count=%d/%d" % (
            topic, int(msg.stable_count), int(min_stable_count)
        )
    states = set(msg.states or [])
    if not states or "unknown" in states:
        return False, "%s unknown states=%s" % (topic, sorted(states))
    unsupported = states - {"open", "closed", "holding"}
    if unsupported:
        return False, "%s unsupported states=%s" % (topic, sorted(unsupported))
    return True, "%s fresh stable states=%s age=%.3f" % (
        topic, sorted(states), age_sec
    )


def scene_graph_gripper_alignment_check(
    topic: str,
    timeout: float,
    max_skew_sec: float,
) -> Tuple[bool, str]:
    deadline = time.monotonic() + float(timeout)
    last_reason = "no_scene_graph"
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        try:
            msg = rospy.wait_for_message(
                topic,
                String,
                timeout=min(1.0, max(0.1, deadline - time.monotonic())),
            )
            payload = json.loads(str(msg.data or "{}"))
            alignment = payload.get("gripper_alignment")
        except Exception as exc:
            last_reason = str(exc)
            continue
        if not isinstance(alignment, dict):
            last_reason = "missing gripper_alignment object"
            continue
        missing = [key for key in ("valid", "reason", "skew_sec") if key not in alignment]
        if missing:
            last_reason = "gripper_alignment missing: %s" % ",".join(missing)
            continue
        if not bool(alignment.get("valid", False)):
            last_reason = str(alignment.get("reason", "alignment_invalid"))
            continue
        try:
            skew_sec = float(alignment.get("skew_sec"))
        except (TypeError, ValueError):
            last_reason = "invalid alignment skew"
            continue
        if skew_sec > float(max_skew_sec):
            last_reason = "alignment skew %.3f > %.3f" % (skew_sec, max_skew_sec)
            continue
        return True, "%s gripper alignment valid skew=%.3f" % (topic, skew_sec)
    return False, "%s: %s" % (topic, last_reason)


def summarize_checks(checks: Dict[str, object]) -> bool:
    all_passed = True
    for value in checks.values():
        if isinstance(value, dict):
            if not all(bool(item) for item in value.values()):
                all_passed = False
        elif isinstance(value, tuple):
            if not value[0]:
                all_passed = False
    return all_passed


def main():
    rospy.init_node('check_real_test_readiness', anonymous=True)

    plan_file = str(rospy.get_param('~plan_file', find_package_resource('action.py') or '')).strip()
    model_path = str(rospy.get_param('~model_path', find_package_resource('best.pt') or '')).strip()
    robot_ip = str(rospy.get_param('~realman_ip', '192.168.0.18')).strip()
    robot_port = int(rospy.get_param('~realman_port', 8080))
    check_topics = bool(rospy.get_param('~check_topics', True))
    check_robot_socket = bool(rospy.get_param('~check_robot_socket', True))
    strict_temporal_checks = bool(rospy.get_param('~strict_temporal_checks', True))
    temporal_topic_timeout = max(0.1, float(rospy.get_param('~temporal_topic_timeout', 5.0)))
    gripper_state_topic = str(rospy.get_param('~gripper_state_topic', '/gripper/state')).strip()
    gripper_state_max_age_sec = max(
        0.1, float(rospy.get_param('~gripper_state_max_age_sec', 1.0))
    )
    gripper_min_stable_count = max(
        1, int(rospy.get_param('~gripper_min_stable_count', 3))
    )
    gripper_camera_max_skew_sec = max(
        0.0, float(rospy.get_param('~gripper_camera_max_skew_sec', 0.2))
    )

    checks = {
        'files': {
            'plan_file': file_check(plan_file)[0],
            'model_path': file_check(model_path)[0],
        },
        'imports': {
            'ultralytics': import_check('ultralytics')[0],
            'pyrealsense2': import_check('pyrealsense2')[0],
            'cv_bridge': import_check('cv_bridge')[0],
            'scipy': import_check('scipy')[0],
            'Robotic_Arm.rm_robot_interface': import_check('Robotic_Arm.rm_robot_interface')[0],
        },
        'details': {
            'plan_file': file_check(plan_file)[1],
            'model_path': file_check(model_path)[1],
            'ultralytics': import_check('ultralytics')[1],
            'pyrealsense2': import_check('pyrealsense2')[1],
            'cv_bridge': import_check('cv_bridge')[1],
            'scipy': import_check('scipy')[1],
            'Robotic_Arm.rm_robot_interface': import_check('Robotic_Arm.rm_robot_interface')[1],
        },
    }

    if check_robot_socket:
        checks['robot_socket'] = tcp_check(robot_ip, robot_port)

    if check_topics:
        checks['topics'] = topic_presence_check(
            [
                '/rm_driver/Arm_Current_State',
                '/rm_driver/ArmCurrentState',
                '/wrist/detected_objects',
                '/global_camera/seg_detections',
                '/scene_graph/current',
                '/task_execution/validation_request',
                '/task_execution/validation_result',
                '/task_execution/gripper_command',
                gripper_state_topic,
            ]
        )

    checks['clock'] = {
        'use_sim_time_disabled': not bool(rospy.get_param('/use_sim_time', False)),
    }

    if strict_temporal_checks:
        checks['global_detection_schema'] = json_topic_fields_check(
            '/global_camera/seg_detections',
            [
                'capture_stamp',
                'capture_lower_bound',
                'capture_upper_bound',
                'source_frame_seq',
                'publish_stamp',
            ],
            temporal_topic_timeout,
        )
        checks['scene_graph_schema'] = json_topic_fields_check(
            '/scene_graph/current',
            [
                'schema_version',
                'scene_version',
                'capture_stamp',
                'source_frame_seq',
                'graph_stamp_sec',
                'gripper_alignment',
            ],
            temporal_topic_timeout,
        )
        checks['gripper_state_schema'] = gripper_state_check(
            gripper_state_topic,
            temporal_topic_timeout,
            gripper_state_max_age_sec,
            gripper_min_stable_count,
        )
        checks['scene_graph_gripper_alignment'] = scene_graph_gripper_alignment_check(
            '/scene_graph/current',
            temporal_topic_timeout,
            gripper_camera_max_skew_sec,
        )

    result = {
        'passed': summarize_checks(checks),
        'checks': checks,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
