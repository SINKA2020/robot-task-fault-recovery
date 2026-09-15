#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
robot_motion_primitives.py

底层动作原语：纯 ROS Publisher 封装与姿态工具函数。
所有函数无 catch.py 状态机依赖，可供 catch.py、cleanup_executor.py、
pick_place_executor.py、robot_recovery_adapter.py 共用。

提取自 catch.py。
"""

from __future__ import annotations

from scene_graph_system.resources import package_resource_path

import os
import time

import numpy as np
import rospy
from rm_msgs.msg import Arm_Current_State, ArmState, Gripper_Pick, Gripper_Set, MoveJ, MoveJ_P, MoveL
from scipy.spatial.transform import Rotation as R

from scene_graph_system.robot.gripper_command_channel import (
    TOPIC_GRIPPER_STATE,
    publish_gripper_command,
    wait_for_fresh_gripper_state,
)

# ============================================================
# 常量
# ============================================================

ARM_READY_JOINTS = [-0.047124, -0.359014, 1.487719, 0.032463, 1.704491, -0.017977]
ARM_READY_TOLERANCE = 0.02

CLEANUP_OBSERVE_JOINTS = [
    -1.5666086673736572,
    -0.5103427171707153,
    1.342707633972168,
    0.031584497541189194,
    2.07389760017395,
    -0.01760704815387726,
]
DEFAULT_CLEANUP_CONFIG_FILE = package_resource_path('config/cleanup_config.yaml')

GRIP_JOINTS = [0.01094, 0.50146, 1.17590, -0.02993, 1.47406, -0.00206]
GRIP_JOINT_TOLERANCE = 0.02

PLACE_QUAT = [
    0.9996736367553412,
    -0.019567201537117926,
    0.004190573821055128,
    0.01588029254788618,
]
PLACE_YAW_OFFSET_RAD = np.deg2rad(90.0)


# ============================================================
# 姿态工具
# ============================================================

def normalize_half_turn_rad(angle_rad):
    return (angle_rad + np.pi / 2.0) % np.pi - np.pi / 2.0


def rotate_tool_quaternion_by_local_z(base_quaternion, delta_yaw_rad):
    base_rotation = R.from_quat(base_quaternion)
    yaw_rotation = R.from_euler("z", delta_yaw_rad, degrees=False)
    rotated = base_rotation * yaw_rotation
    return rotated.as_quat()


def get_current_tool_quaternion(timeout=2.0):
    """
    读取机械臂当前末端真实四元数，并做归一化。
    返回格式：[x, y, z, w]
    """
    msg = rospy.wait_for_message("/rm_driver/ArmCurrentState", ArmState, timeout=timeout)

    q = np.array([
        msg.Pose.orientation.x,
        msg.Pose.orientation.y,
        msg.Pose.orientation.z,
        msg.Pose.orientation.w,
    ], dtype=float)

    norm = np.linalg.norm(q)
    if norm < 1e-6:
        raise RuntimeError("Invalid quaternion from /rm_driver/ArmCurrentState")

    q = q / norm

    if q[3] < 0:
        q = -q

    return q.tolist()


def get_current_tcp_pose_list(timeout=1.0, topic=None, msg_type=None):
    """
    读取当前 TCP 位姿，返回 [x, y, z, rx, ry, rz] 或至少前三轴 xyz。

    topic 默认 /rm_driver/Arm_Current_State（带下划线，实测可用）。
    msg_type 默认尝试 Arm_Current_State 回退 ArmState，也可显式指定。

    兼容 Pose 为 list 或 geometry_msgs/Pose 两种格式。
    """
    topic = topic or rospy.get_param(
        "~arm_current_state_topic", "/rm_driver/Arm_Current_State"
    )

    if msg_type is None:
        try:
            MsgType = Arm_Current_State
        except NameError:
            MsgType = ArmState
    else:
        MsgType = msg_type

    try:
        msg = rospy.wait_for_message(topic, MsgType, timeout=float(timeout))
    except Exception:
        # 回退：尝试 ArmState
        msg = rospy.wait_for_message(topic, ArmState, timeout=float(timeout))

    pose = getattr(msg, "Pose", None)
    if pose is None:
        raise RuntimeError("Arm state message has no Pose field")

    # Pose 是 list，例如 [x, y, z, rx, ry, rz]
    if isinstance(pose, (list, tuple)):
        if len(pose) < 3:
            raise RuntimeError("Pose list length < 3: %s" % str(pose))
        return [float(v) for v in pose]

    # Pose 是 geometry_msgs/Pose
    if hasattr(pose, "position"):
        return [
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
        ]

    raise RuntimeError("Unsupported Pose field type: %s" % str(type(pose)))


def get_gripper_position(timeout=0.5):
    """
    从独立夹爪监控话题读取位置；保留该接口供旧调用兼容。
    """
    topic = str(rospy.get_param("~gripper_state_topic", TOPIC_GRIPPER_STATE)).strip()
    max_age_sec = max(
        0.1, float(rospy.get_param("~gripper_state_max_age_sec", 1.0))
    )
    msg, _reason = wait_for_fresh_gripper_state(
        timeout_sec=float(timeout),
        max_age_sec=max_age_sec,
        topic=topic,
    )
    return None if msg is None else float(msg.position)


def wait_for_tcp_xyz(
    target_xyz,
    tolerance=0.015,
    timeout=20.0,
    check_continue_fn=None,
    topic=None,
):
    """
    等待 TCP 到达目标 xyz 附近。tolerance 默认 1.5cm。

    返回 True 表示已到达，False 表示超时或被中断。
    """
    target = [float(v) for v in target_xyz[:3]]
    deadline = time.time() + float(timeout)

    rate = rospy.Rate(20)

    while not rospy.is_shutdown() and time.time() < deadline:
        if check_continue_fn is not None and not check_continue_fn():
            return False

        try:
            pose = get_current_tcp_pose_list(timeout=1.0, topic=topic)
            current = [float(pose[0]), float(pose[1]), float(pose[2])]

            err = max(abs(current[i] - target[i]) for i in range(3))
            if err <= float(tolerance):
                rospy.logwarn(
                    "[MOTION] TCP reached target xyz=%s current=%s err=%.4f",
                    [round(v, 4) for v in target],
                    [round(v, 4) for v in current],
                    err,
                )
                return True

        except Exception as exc:
            rospy.logwarn("[MOTION] wait_for_tcp_xyz read failed: %s", str(exc))

        rate.sleep()

    rospy.logerr(
        "[MOTION] timeout waiting TCP target=%s tolerance=%.3f",
        [round(v, 4) for v in target],
        float(tolerance),
    )
    return False


def get_default_grasp_quat():
    """
    返回默认竖直向下抓取姿态四元数（不移动机械臂）。

    用于 cleanup_executor 等不需要动态读取抓取姿态的场景。
    经验四元数对应工具 z 轴竖直向下。
    """
    return [0.707, -0.707, 0.0, 0.0]


def get_place_quaternion():
    """返回放置姿态四元数。"""
    q = np.array(PLACE_QUAT, dtype=float)
    norm = np.linalg.norm(q)

    if norm < 1e-6:
        raise RuntimeError("Invalid PLACE_QUAT: norm is too small")

    q = q / norm

    if q[3] < 0:
        q = -q

    base_rotation = R.from_quat(q)
    yaw_rotation = R.from_euler("z", PLACE_YAW_OFFSET_RAD, degrees=False)

    rotated = base_rotation * yaw_rotation
    q_rotated = rotated.as_quat()

    if q_rotated[3] < 0:
        q_rotated = -q_rotated

    return q_rotated.tolist()


# ============================================================
# 关节运动
# ============================================================

def movej_type(joint, speed):
    try:
        moveJ_pub = rospy.Publisher("/rm_driver/MoveJ_Cmd", MoveJ, queue_size=1)
        rospy.sleep(0.5)

        move_joint = MoveJ()
        move_joint.joint = joint
        move_joint.speed = speed

        moveJ_pub.publish(move_joint)
        return True

    except Exception as e:
        rospy.logerr("Error publishing MoveJ command: %s", e)
        return False


def movejp_type(pose, speed):
    try:
        moveJ_P_pub = rospy.Publisher("/rm_driver/MoveJ_P_Cmd", MoveJ_P, queue_size=1)
        rospy.sleep(0.5)

        move_joint_pose = MoveJ_P()
        move_joint_pose.Pose.position.x = pose[0]
        move_joint_pose.Pose.position.y = pose[1]
        move_joint_pose.Pose.position.z = pose[2]
        move_joint_pose.Pose.orientation.x = pose[3]
        move_joint_pose.Pose.orientation.y = pose[4]
        move_joint_pose.Pose.orientation.z = pose[5]
        move_joint_pose.Pose.orientation.w = pose[6]
        move_joint_pose.speed = speed

        moveJ_P_pub.publish(move_joint_pose)
        return True

    except Exception as e:
        rospy.logerr("Error publishing MoveJ_P command: %s", e)
        return False


def movel_type(pose, speed):
    try:
        moveL_pub = rospy.Publisher("/rm_driver/MoveL_Cmd", MoveL, queue_size=1)
        rospy.sleep(0.5)

        move_line_pose = MoveL()
        move_line_pose.Pose.position.x = pose[0]
        move_line_pose.Pose.position.y = pose[1]
        move_line_pose.Pose.position.z = pose[2]
        move_line_pose.Pose.orientation.x = pose[3]
        move_line_pose.Pose.orientation.y = pose[4]
        move_line_pose.Pose.orientation.z = pose[5]
        move_line_pose.Pose.orientation.w = pose[6]
        move_line_pose.speed = speed

        moveL_pub.publish(move_line_pose)
        return True

    except Exception as e:
        rospy.logerr("Error publishing MoveL command: %s", e)
        return False


def wait_for_joint_pose(target_joints, tolerance=0.02, timeout=10.0, check_continue_fn=None):
    """
    阻塞等待机械臂实际到达指定关节角度。

    check_continue_fn: 可选回调，返回 False 时提前退出。
        catch.py 传入 wait_if_paused_or_aborted；
        cleanup_executor 不传，使用普通循环等待。
    """
    start_time = rospy.Time.now()
    rate = rospy.Rate(50)
    rospy.loginfo("Waiting for arm to stabilize at target joint pose...")

    while not rospy.is_shutdown():
        if check_continue_fn is not None and not check_continue_fn():
            return False

        try:
            msg = rospy.wait_for_message("/rm_driver/ArmCurrentState", ArmState, timeout=1.0)
            current_joints = getattr(msg, "joint", getattr(msg, "q", None))

            if current_joints and len(current_joints) == len(target_joints):
                if all(abs(c - t) < tolerance for c, t in zip(current_joints, target_joints)):
                    rospy.loginfo("Arm has reached target joint pose.")
                    return True

        except rospy.ROSException:
            pass

        if (rospy.Time.now() - start_time).to_sec() > timeout:
            rospy.logwarn("Timeout waiting for target joint pose.")
            return False

        rate.sleep()


def grip_pose():
    try:
        moveJ_pub = rospy.Publisher("/rm_driver/MoveJ_Cmd", MoveJ, queue_size=1)
        rospy.sleep(1)

        pic_joint = MoveJ()
        pic_joint.joint = GRIP_JOINTS
        pic_joint.speed = 0.1

        moveJ_pub.publish(pic_joint)
        return True

    except Exception as e:
        rospy.logerr("Error publishing grip_pose command: %s", e)
        return False


def get_reference_grasp_quaternion(sleep_fn=None, check_continue_fn=None):
    """
    运动到 grip_pose，读取该姿态下的真实四元数。

    sleep_fn: 可选的可中断 sleep 函数，签名 fn(seconds) -> bool。
        catch.py 传入 interruptible_sleep；
        cleanup_executor 不传，使用 rospy.sleep。

    check_continue_fn: 可选回调，返回 False 时提前退出。
        catch.py 传入 wait_if_paused_or_aborted。
    """
    if not grip_pose():
        raise RuntimeError("Failed to send grip_pose command")

    if not wait_for_joint_pose(
        GRIP_JOINTS,
        tolerance=GRIP_JOINT_TOLERANCE,
        timeout=10.0,
        check_continue_fn=check_continue_fn,
    ):
        raise RuntimeError("Robot did not reach grip_pose joints in time")

    if check_continue_fn is not None and not check_continue_fn():
        raise RuntimeError("Interrupted before reading grasp quaternion")

    if sleep_fn is not None:
        if not sleep_fn(0.3):
            raise RuntimeError("Interrupted while waiting after grip_pose")
    else:
        rospy.sleep(0.3)

    return get_current_tool_quaternion(timeout=2.0)


# ============================================================
# 预定义姿态
# ============================================================

def arm_ready_pose():
    try:
        moveJ_pub = rospy.Publisher("/rm_driver/MoveJ_Cmd", MoveJ, queue_size=1)
        rospy.sleep(1)

        pic_joint = MoveJ()
        pic_joint.joint = ARM_READY_JOINTS
        pic_joint.speed = 0.2

        moveJ_pub.publish(pic_joint)
        return True

    except Exception as e:
        rospy.logerr("Error publishing arm_ready_pose command: %s", e)
        return False


def _load_cleanup_observe_joints_from_config(config_file=None):
    config_file = config_file or DEFAULT_CLEANUP_CONFIG_FILE
    try:
        if not os.path.exists(config_file):
            return list(CLEANUP_OBSERVE_JOINTS)

        import yaml

        with open(config_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        joints = cfg.get("cleanup_observe_joints", None)
        if joints is None or len(joints) != 6:
            return list(CLEANUP_OBSERVE_JOINTS)

        return [float(v) for v in joints]
    except Exception as e:
        rospy.logwarn("Failed to load cleanup_observe_joints from config: %s", str(e))
        return list(CLEANUP_OBSERVE_JOINTS)


def cleanup_observe_pose(config_file=None):
    """
    移动到清理观测位姿。

    优先从 cleanup_config.yaml 读取 cleanup_observe_joints；
    读取失败时使用 CLEANUP_OBSERVE_JOINTS。
    """
    try:
        joints = _load_cleanup_observe_joints_from_config(config_file=config_file)

        moveJ_pub = rospy.Publisher("/rm_driver/MoveJ_Cmd", MoveJ, queue_size=1)
        rospy.sleep(1)

        cmd = MoveJ()
        cmd.joint = joints
        cmd.speed = 0.15

        moveJ_pub.publish(cmd)
        print("Moved to cleanup observe pose: %s" % joints)
        return True

    except Exception as e:
        rospy.logerr("Error publishing cleanup_observe_pose command: %s", e)
        return False


# ============================================================
# 夹爪
# ============================================================

def gripper_open():
    try:
        set_pub = rospy.Publisher("/rm_driver/Gripper_Set", Gripper_Set, queue_size=1)
        rospy.sleep(1)

        set_cmd = Gripper_Set()
        set_cmd.position = 1000

        publish_gripper_command(
            "open",
            source="robot_motion_primitives",
            requested_position=set_cmd.position,
        )
        set_pub.publish(set_cmd)
        print("Gripper opened.")
        return True

    except Exception as e:
        rospy.logerr("Error opening gripper: %s", e)
        return False


def gripper_close():
    try:
        pick_pub = rospy.Publisher("/rm_driver/Gripper_Pick_On", Gripper_Pick, queue_size=1)
        rospy.sleep(1)

        pick_cmd = Gripper_Pick()
        pick_cmd.speed = 200
        pick_cmd.force = 1000

        publish_gripper_command(
            "close",
            source="robot_motion_primitives",
            requested_speed=pick_cmd.speed,
            requested_force=pick_cmd.force,
        )
        pick_pub.publish(pick_cmd)
        print("Gripper closed.")
        return True

    except Exception as e:
        rospy.logerr("Error closing gripper: %s", e)
        return False
