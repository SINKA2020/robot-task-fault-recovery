#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import time
from typing import List, Optional

import rospy
from std_msgs.msg import String
from scipy.spatial.transform import Rotation as R

from scene_graph_system.diagnosis.fault_event import TOPIC_EXECUTION_CONTROL
try:
    from scene_graph_system.diagnosis.fault_event import TOPIC_EXECUTION_RECOVERY_READY
except Exception:
    TOPIC_EXECUTION_RECOVERY_READY = "/task_execution/recovery_ready"

# 复用 catch.py 中已经调试好的底层动作接口。
# movel_type 用于从当前位姿竖直上抬；
# movejp_type 用于回到 source_pose 上方；
# arm_ready_pose 用于回观测 / ready 位。
from scene_graph_system.robot.robot_motion_primitives import (
    arm_ready_pose,
    cleanup_observe_pose,
    gripper_close,
    gripper_open,
    movel_type,
    movejp_type,
    wait_for_tcp_xyz,
)
from scene_graph_system.robot.gripper_command_channel import (
    TOPIC_GRIPPER_STATE,
    wait_for_fresh_gripper_state,
)

try:
    from rm_msgs.msg import ArmState
except Exception:
    ArmState = None


class RobotRecoveryAdapter:
    def __init__(self, control_topic: str = TOPIC_EXECUTION_CONTROL):
        self.control_pub = rospy.Publisher(control_topic, String, queue_size=10)

        self.arm_state_topic = str(
            rospy.get_param("~arm_state_topic", "/rm_driver/ArmCurrentState")
        )

        fallback = rospy.get_param(
            "~arm_state_fallback_topics",
            ["/rm_driver/Arm_Current_State", "/rm_driver/ArmCurrentState"],
        )

        if isinstance(fallback, str):
            try:
                fallback = json.loads(fallback)
            except Exception:
                fallback = [v.strip() for v in fallback.split(",") if v.strip()]

        self.arm_state_topics = []
        for topic in [self.arm_state_topic] + list(fallback):
            topic = str(topic).strip()
            if topic and topic not in self.arm_state_topics:
                self.arm_state_topics.append(topic)

        rospy.logwarn("[RECOVERY] arm_state_topics=%s", str(self.arm_state_topics))

        self.recovery_ready_topic = str(
            rospy.get_param("~recovery_ready_topic", TOPIC_EXECUTION_RECOVERY_READY)
        )

        self._recovery_request_counter = 0
        self._pending_recovery_request_id = ""
        self.last_repair_result = None
        self.gripper_state_topic = str(
            rospy.get_param("~gripper_state_topic", TOPIC_GRIPPER_STATE)
        ).strip()
        self.gripper_state_max_age_sec = max(
            0.1, float(rospy.get_param("~gripper_state_max_age_sec", 1.0))
        )
        self.gripper_min_stable_count = max(
            1, int(rospy.get_param("~gripper_min_stable_count", 3))
        )

    # ============================================================
    # 执行端控制
    # ============================================================

    def _send_control_payload(self, payload: dict):
        self.control_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        rospy.logwarn("[RECOVERY] sent control payload: %s", payload)

    def _send_control(self, cmd: str):
        self._send_control_payload({"cmd": str(cmd)})

    def pause_execution(self):
        self._send_control("pause")
        return True

    def pause_for_recovery(self):
        """
        请求执行端让出控制权。

        生成唯一的 request_id，让 catch.py 在 recovery_ready 中回传，
        配合两阶段握手（ready=false ack → ready=true handoff）防止
        latch 残留消息和跨恢复周期误匹配。
        """
        self._recovery_request_counter += 1
        request_id = "recovery_{}_{:.0f}".format(
            self._recovery_request_counter, time.time() * 1000
        )
        self._pending_recovery_request_id = request_id

        self._send_control_payload({
            "cmd": "pause_for_recovery",
            "request_id": request_id,
        })
        return True

    def wait_recovery_ready(self, timeout: float = 15.0) -> bool:
        """
        等待执行端确认 recovery_ready（两阶段握手）。

        忽略 ready=false（ack/pending）和 request_id 不匹配的消息，
        仅当 ready=true 且 request_id 匹配时返回 True。

        由于 catch.py 的 recovery_ready_pub 启用了 latch=True，
        rospy.wait_for_message 创建临时订阅者时会自动收到 latch 缓存
        的最新消息，消除了竞态窗口。
        """
        expected_request_id = str(self._pending_recovery_request_id or "")

        deadline = time.monotonic() + float(timeout)

        while not rospy.is_shutdown() and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            try:
                msg = rospy.wait_for_message(
                    self.recovery_ready_topic,
                    String,
                    timeout=min(remaining, 1.0),
                )
            except Exception:
                # 1 秒内没有消息，继续循环等待
                continue

            try:
                payload = json.loads(str(msg.data or "{}"))
            except Exception:
                payload = {"raw": str(msg.data)}

            msg_request_id = str(payload.get("request_id", ""))
            is_ready = bool(payload.get("ready", False))

            rospy.logwarn(
                "[RECOVERY] recovery_ready received: ready=%s request_id=%s expected=%s payload=%s",
                str(is_ready),
                msg_request_id,
                expected_request_id,
                payload,
            )

            # request_id 不匹配 → 这是 latch 残留的上一轮消息，忽略
            if expected_request_id and msg_request_id and msg_request_id != expected_request_id:
                rospy.logwarn(
                    "[RECOVERY] recovery_ready request_id mismatch (got=%s expected=%s); ignoring stale latch message",
                    msg_request_id,
                    expected_request_id,
                )
                continue

            # ready=false → 执行端已收到请求但还在等运动退出，继续等 ready=true
            if not is_ready:
                rospy.logwarn(
                    "[RECOVERY] recovery_ready pending (ready=false); waiting for handoff...",
                )
                continue

            return True

        rospy.logerr(
            "[RECOVERY] timeout waiting recovery_ready on %s (request_id=%s)",
            self.recovery_ready_topic,
            expected_request_id,
        )
        return False

    def resume_execution(self):
        self._send_control("resume")
        return True

    def abort_execution(self):
        self._send_control("abort")
        return True

    def resume_from_step(self, resume_step: int):
        """
        通知执行系统从指定 step 继续执行。
        """
        resume_step = int(resume_step)

        self._send_control_payload({
            "cmd": "resume_from_step",
            "resume_step": resume_step,
        })

        return True

    # ============================================================
    # 基础机器人动作
    # ============================================================

    def move_ready(self):
        return bool(arm_ready_pose())

    def open_gripper(self):
        gripper_open()
        return True

    def verify_gripper_open(self, timeout=1.5, threshold=900.0) -> bool:
        """
        使用独立夹爪监控链确认夹爪达到稳定 open 状态。

        threshold 参数仅为兼容旧调用保留，不再用于重复推断状态。
        """
        deadline = time.monotonic() + float(timeout)

        while time.monotonic() < deadline:
            msg, _reason = wait_for_fresh_gripper_state(
                timeout_sec=min(0.4, max(0.1, deadline - time.monotonic())),
                max_age_sec=self.gripper_state_max_age_sec,
                topic=self.gripper_state_topic,
            )
            if (
                msg is not None
                and int(msg.stable_count) >= self.gripper_min_stable_count
                and "open" in set(msg.states or [])
            ):
                return True

        rospy.logerr("[RECOVERY] verify_gripper_open: failed within %.1fs", timeout)
        return False

    def close_gripper(self):
        gripper_close()
        return True

    def wait_stable_scene(self, seconds: float = 1.0):
        time.sleep(float(seconds))
        return True

    # ============================================================
    # 安全撤离
    # ============================================================

    def safe_retreat(
        self,
        lift_z: float = 0.06,
        lift_speed: float = 0.08,
        move_ready: bool = True,
        release_object: bool = False,
        lift_wait: float = 1.0,
        wait_after: float = 1.0,
    ) -> bool:
        """
        安全撤离：

        1. pause 当前执行系统；
        2. 从当前 TCP 位姿沿 base z 方向竖直上抬；
        3. 可选回 ready pose；
        4. 默认不打开夹爪；
        5. 等待场景稳定。

        注意：
        如果夹爪中有物体，恢复计划应设置 move_ready=False，
        然后调用 return_held_object_to_source() 把物体放回 source_pose。
        不建议在 ready 位直接 release_object=True。
        """
        rospy.logwarn(
            "[RECOVERY] safe_retreat started: lift_z=%.3f lift_speed=%.3f "
            "move_ready=%s release_object=%s",
            float(lift_z),
            float(lift_speed),
            str(move_ready),
            str(release_object),
        )

        ok = self.pause_for_recovery()
        if not ok:
            rospy.logerr("[RECOVERY] pause_for_recovery failed")
            return False

        if not self.wait_recovery_ready(timeout=15.0):
            rospy.logerr("[RECOVERY] execution side did not report recovery_ready; abort safe_retreat")
            return False

        # 给底层运动一点稳定时间，避免上一条 MoveL/MoveJ 指令尚未完全结束。
        time.sleep(0.5)

        lifted = self._lift_from_current_pose(
            lift_z=float(lift_z),
            speed=float(lift_speed),
            wait_after=float(lift_wait),
        )

        if not lifted:
            rospy.logwarn(
                "[RECOVERY] lift_from_current_pose failed. Fallback to next recovery action."
            )

        if move_ready:
            ok = self.move_ready()
            if not ok:
                rospy.logerr("[RECOVERY] move_ready failed during safe_retreat")
                return False

            time.sleep(1.0)

        if release_object:
            rospy.logwarn("[RECOVERY] release_object=True, opening gripper after retreat")
            ok = self.open_gripper()
            if not ok:
                rospy.logerr("[RECOVERY] open_gripper failed during safe_retreat")
                return False

        if wait_after > 0:
            self.wait_stable_scene(float(wait_after))

        rospy.logwarn("[RECOVERY] safe_retreat finished")
        return True

    def return_held_object_to_source(
        self,
        source_pose=None,
        source_return_approach_pose=None,
        source_return_release_pose=None,
        source_return_lift_pose=None,
        move_speed: float = 0.15,
        wait_after_release: float = 1.0,
        **unused,
    ) -> bool:
        """
        将当前夹爪中物体放回原抓取位置。

        安全要求：
        - 必须使用 catch.py 在抓取时记录的 7D 安全返回位姿：
            source_return_approach_pose / source_return_release_pose / source_return_lift_pose
        - 禁止使用“当前 TCP 姿态 + source_pose”临时拼接，因为故障可能发生在放置姿态，
          直接带该姿态回源位可能导致腕部扭曲。
        """
        poses = {
            "approach": source_return_approach_pose,
            "release": source_return_release_pose,
            "lift": source_return_lift_pose,
        }

        for name, pose in poses.items():
            if pose is None:
                rospy.logerr(
                    "[RECOVERY] cannot return held object: missing safe %s pose. "
                    "source_pose=%s. Refuse unsafe return.",
                    name,
                    str(source_pose),
                )
                return False
            try:
                pose_list = list(pose)
                if len(pose_list) < 7:
                    rospy.logerr("[RECOVERY] invalid %s pose: %s", name, str(pose))
                    return False
                poses[name] = [float(v) for v in pose_list[:7]]
            except Exception as exc:
                rospy.logerr("[RECOVERY] failed to parse %s pose=%s: %s", name, str(pose), str(exc))
                return False

        approach_pose = poses["approach"]
        release_pose = poses["release"]
        lift_pose = poses["lift"]

        rospy.logwarn(
            "[RECOVERY] returning held object using recorded safe poses: approach_xyz=%s release_xyz=%s lift_xyz=%s",
            str(approach_pose[:3]),
            str(release_pose[:3]),
            str(lift_pose[:3]),
        )

        try:
            if not bool(movejp_type(approach_pose, float(move_speed))):
                rospy.logerr("[RECOVERY] failed to move above source")
                return False
            if not bool(wait_for_tcp_xyz(approach_pose[:3], timeout=20.0)):
                rospy.logerr("[RECOVERY] timed out waiting for approach pose")
                return False

            if not bool(movel_type(release_pose, float(move_speed))):
                rospy.logerr("[RECOVERY] failed to descend to source")
                return False
            if not bool(wait_for_tcp_xyz(release_pose[:3], tolerance=0.01, timeout=15.0)):
                rospy.logerr("[RECOVERY] timed out waiting for release pose")
                return False

            rospy.logwarn("[RECOVERY] opening gripper to return object")
            self.open_gripper()
            time.sleep(float(wait_after_release))

            if not bool(movel_type(lift_pose, float(move_speed))):
                rospy.logerr("[RECOVERY] failed to lift after release")
                return False
            if not bool(wait_for_tcp_xyz(lift_pose[:3], timeout=10.0)):
                rospy.logerr("[RECOVERY] timed out waiting for lift pose")
                return False

        except Exception as exc:
            rospy.logerr("[RECOVERY] return_held_object_to_source failed: %s", str(exc))
            return False

        rospy.logwarn("[RECOVERY] held object returned to source")
        return True

    def _lift_from_current_pose(
        self,
        lift_z: float = 0.06,
        speed: float = 0.08,
        wait_after: float = 1.0,
    ) -> bool:
        """
        从当前 TCP 位姿沿 base z 方向上抬。
        """
        pose = self._get_current_tcp_pose(timeout=1.5)

        if pose is None:
            rospy.logwarn("[RECOVERY] Cannot get current TCP pose for lift")
            return False

        target_pose = list(pose)
        target_pose[2] = float(target_pose[2]) + float(lift_z)

        rospy.logwarn(
            "[RECOVERY] lifting TCP: current_xyz=(%.4f, %.4f, %.4f) target_z=%.4f",
            float(pose[0]),
            float(pose[1]),
            float(pose[2]),
            float(target_pose[2]),
        )

        try:
            ok = bool(movel_type(target_pose, float(speed)))
        except Exception as exc:
            rospy.logerr("[RECOVERY] movel_type failed during lift: %s", str(exc))
            return False

        if not ok:
            rospy.logerr("[RECOVERY] movel_type returned False during lift")
            return False

        if wait_after > 0:
            time.sleep(float(wait_after))

        return True

    def _wait_for_arm_state_msg(self, timeout: float = 1.5):
        """
        从 arm_state_topics 列表中依次尝试读取 ArmState 消息。

        优先主 topic，失败后自动 fallback 到备用 topic。
        """
        if ArmState is None:
            rospy.logerr("[RECOVERY] rm_msgs.msg.ArmState is unavailable")
            return None

        topics = list(getattr(self, "arm_state_topics", []) or [self.arm_state_topic])

        for topic in topics:
            try:
                msg = rospy.wait_for_message(
                    topic,
                    ArmState,
                    timeout=float(timeout),
                )
                rospy.logwarn("[RECOVERY] got arm state from %s", topic)
                return msg
            except Exception as exc:
                rospy.logwarn(
                    "[RECOVERY] wait_for_message(%s) failed: %s",
                    topic,
                    str(exc),
                )

        return None

    def _get_current_tcp_pose(self, timeout: float = 1.5) -> Optional[List[float]]:
        """
        读取当前 TCP 位姿，统一返回 7 元素四元数格式：

          [x, y, z, qx, qy, qz, qw]

        兼容两种 ArmState.Pose 格式：
          1. Pose = [x, y, z, rx, ry, rz]
          2. Pose.position + Pose.orientation
        """
        if ArmState is None:
            rospy.logerr("[RECOVERY] rm_msgs.msg.ArmState is unavailable")
            return None

        msg = self._wait_for_arm_state_msg(timeout=float(timeout))
        if msg is None:
            return None

        pose = getattr(msg, "Pose", None)
        if pose is None:
            rospy.logwarn("[RECOVERY] ArmState has no Pose field")
            return None

        # 格式 1：Pose 是 [x, y, z, rx, ry, rz]
        try:
            values = list(pose)
            if len(values) >= 6:
                x, y, z, rx, ry, rz = [float(v) for v in values[:6]]
                qx, qy, qz, qw = R.from_euler("xyz", [rx, ry, rz], degrees=False).as_quat()
                return [
                    float(x),
                    float(y),
                    float(z),
                    float(qx),
                    float(qy),
                    float(qz),
                    float(qw),
                ]
        except Exception:
            pass

        # 格式 2：Pose 是 geometry_msgs/Pose
        if hasattr(pose, "position") and hasattr(pose, "orientation"):
            try:
                return [
                    float(pose.position.x),
                    float(pose.position.y),
                    float(pose.position.z),
                    float(pose.orientation.x),
                    float(pose.orientation.y),
                    float(pose.orientation.z),
                    float(pose.orientation.w),
                ]
            except Exception as exc:
                rospy.logwarn("[RECOVERY] failed to parse Pose position/orientation: %s", str(exc))
                return None

        rospy.logwarn("[RECOVERY] Unsupported ArmState.Pose format: %s", repr(pose))
        return None

    def _get_current_tcp_pose_rpy(self, timeout: float = 1.5) -> Optional[List[float]]:
        """
        读取当前 TCP 位姿，统一返回：

          [x, y, z, rx, ry, rz]

        仅用于 cleanup_observe_pose 到位判断。
        """
        if ArmState is None:
            rospy.logerr("[RECOVERY] rm_msgs.msg.ArmState is unavailable")
            return None

        msg = self._wait_for_arm_state_msg(timeout=float(timeout))
        if msg is None:
            return None

        pose = getattr(msg, "Pose", None)
        if pose is None:
            rospy.logwarn("[RECOVERY] ArmState has no Pose field")
            return None

        # 格式 1：Pose 是 [x, y, z, rx, ry, rz]
        try:
            values = list(pose)
            if len(values) >= 6:
                return [float(v) for v in values[:6]]
        except Exception:
            pass

        # 格式 2：Pose 是 geometry_msgs/Pose
        if hasattr(pose, "position") and hasattr(pose, "orientation"):
            try:
                q = [
                    float(pose.orientation.x),
                    float(pose.orientation.y),
                    float(pose.orientation.z),
                    float(pose.orientation.w),
                ]
                rx, ry, rz = R.from_quat(q).as_euler("xyz", degrees=False)
                return [
                    float(pose.position.x),
                    float(pose.position.y),
                    float(pose.position.z),
                    float(rx),
                    float(ry),
                    float(rz),
                ]
            except Exception as exc:
                rospy.logwarn("[RECOVERY] failed to parse Pose position/orientation as rpy: %s", str(exc))
                return None

        rospy.logwarn("[RECOVERY] Unsupported ArmState.Pose format: %s", repr(pose))
        return None

    def _angle_diff(self, a: float, b: float) -> float:
        """计算角度差，范围 [-pi, pi]。"""
        return (float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi

    def _parse_cleanup_observe_expected_pose(self) -> List[float]:
        """
        读取清理观测位姿期望 TCP 位姿。

        默认值来自实测正确的 cleanup observe pose：
          [-0.004, 0.063, 0.465, 3.121, 0.235, -1.532]

        可通过 ROS 参数覆盖：
          _cleanup_observe_expected_pose:='[-0.004,0.063,0.465,3.121,0.235,-1.532]'
        """
        default_pose = [-0.004, 0.063, 0.465, 3.121, 0.235, -1.532]
        raw = rospy.get_param("~cleanup_observe_expected_pose", default_pose)

        if isinstance(raw, (list, tuple)):
            values = [float(v) for v in list(raw)]
        else:
            text = str(raw).strip()
            try:
                values = [float(v) for v in json.loads(text)]
            except Exception:
                values = [
                    float(v.strip())
                    for v in text.strip("[]").split(",")
                    if v.strip()
                ]

        if len(values) < 3:
            rospy.logwarn(
                "[RECOVERY] invalid cleanup_observe_expected_pose=%s, fallback to default",
                str(raw),
            )
            return default_pose

        if len(values) < 6:
            values = values[:3] + default_pose[3:6]

        return values[:6]

    def _wait_for_cleanup_observe_pose(
        self,
        timeout: Optional[float] = None,
        xyz_tolerance: Optional[float] = None,
        rpy_tolerance: Optional[float] = None,
        stable_count: Optional[int] = None,
        settle_after: Optional[float] = None,
    ) -> bool:
        """
        阻塞等待机械臂实际到达清理观测位姿。

        必须在 cleanup_build_region 扫描前调用。
        如果未到位，禁止开始清理扫描，避免 object_pose 使用错误机械臂姿态转换坐标。
        """
        expected = self._parse_cleanup_observe_expected_pose()

        timeout = float(
            rospy.get_param(
                "~cleanup_observe_wait_timeout",
                25.0 if timeout is None else timeout,
            )
        )
        xyz_tolerance = float(
            rospy.get_param(
                "~cleanup_observe_xyz_tolerance",
                0.025 if xyz_tolerance is None else xyz_tolerance,
            )
        )
        rpy_tolerance = float(
            rospy.get_param(
                "~cleanup_observe_rpy_tolerance",
                0.080 if rpy_tolerance is None else rpy_tolerance,
            )
        )
        stable_count = int(
            rospy.get_param(
                "~cleanup_observe_stable_count",
                3 if stable_count is None else stable_count,
            )
        )
        settle_after = float(
            rospy.get_param(
                "~cleanup_observe_settle_after",
                1.0 if settle_after is None else settle_after,
            )
        )

        stable = 0
        start = time.monotonic()

        rospy.logwarn(
            "[RECOVERY] waiting cleanup_observe_pose: expected_xyz=[%.3f, %.3f, %.3f] "
            "expected_rpy=[%.3f, %.3f, %.3f] timeout=%.1f xyz_tol=%.3f rpy_tol=%.3f stable_count=%d",
            expected[0], expected[1], expected[2],
            expected[3], expected[4], expected[5],
            timeout,
            xyz_tolerance,
            rpy_tolerance,
            stable_count,
        )

        while not rospy.is_shutdown() and (time.monotonic() - start) <= timeout:
            pose = self._get_current_tcp_pose_rpy(timeout=1.0)
            if pose is None or len(pose) < 3:
                stable = 0
                time.sleep(0.2)
                continue

            dx = float(pose[0]) - expected[0]
            dy = float(pose[1]) - expected[1]
            dz = float(pose[2]) - expected[2]
            xyz_err = math.sqrt(dx * dx + dy * dy + dz * dz)

            rpy_ok = True
            rpy_err = 0.0

            if len(pose) == 6:
                dr = self._angle_diff(float(pose[3]), expected[3])
                dp = self._angle_diff(float(pose[4]), expected[4])
                dyaw = self._angle_diff(float(pose[5]), expected[5])
                rpy_err = max(abs(dr), abs(dp), abs(dyaw))
                rpy_ok = rpy_err <= rpy_tolerance

            xyz_ok = xyz_err <= xyz_tolerance

            rospy.logwarn(
                "[RECOVERY] cleanup_observe wait: current_xyz=[%.3f, %.3f, %.3f] "
                "xyz_err=%.4f rpy_err=%.4f stable=%d/%d",
                float(pose[0]),
                float(pose[1]),
                float(pose[2]),
                xyz_err,
                rpy_err,
                stable,
                stable_count,
            )

            if xyz_ok and rpy_ok:
                stable += 1
                if stable >= stable_count:
                    rospy.logwarn(
                        "[RECOVERY] cleanup_observe_pose reached and stable. wait %.2fs before scan",
                        settle_after,
                    )
                    if settle_after > 0:
                        time.sleep(settle_after)
                    return True
            else:
                stable = 0

            time.sleep(0.2)

        rospy.logerr(
            "[RECOVERY] timeout waiting cleanup_observe_pose. "
            "Refuse cleanup scan to avoid wrong coordinate transform."
        )
        return False

    # ============================================================
    # 清理相关动作
    # ============================================================

    def move_to_cleanup_observe_pose(self, config_file: str = "") -> bool:
        """
        移动到清理观测位姿，并等待机械臂实际到位。

        1. 调用 cleanup_observe_pose() 下发运动指令；
        2. 阻塞等待 TCP 到达期望的 cleanup observe pose；
        3. 两阶段任意失败则返回 False。

        期望位姿由 ROS param _cleanup_observe_expected_pose 指定，
        默认值为 [-0.004, 0.063, 0.465, 3.121, 0.235, -1.532]。
        """
        try:
            rospy.logwarn("[RECOVERY] moving to cleanup_observe_pose")
            ok = bool(cleanup_observe_pose())
            if not ok:
                rospy.logerr("[RECOVERY] cleanup_observe_pose command failed")
                return False

            rospy.logwarn("[RECOVERY] waiting for arm to reach cleanup_observe_pose")
            ok = self._wait_for_cleanup_observe_pose()
            if not ok:
                rospy.logerr(
                    "[RECOVERY] arm did not reach cleanup_observe_pose within tolerance/timeout"
                )
                return False

            rospy.logwarn("[RECOVERY] cleanup_observe_pose confirmed reached")
            return True
        except Exception as exc:
            rospy.logerr("[RECOVERY] move_to_cleanup_observe_pose exception: %s", str(exc))
            return False

    def move_to_task_start_pose(self) -> bool:
        """
        移动到任务起始位姿（ready pose）。

        清理完成后、断点恢复前，机械臂需要回到安全起始位姿。
        """
        try:
            ok = bool(arm_ready_pose())
            if ok:
                rospy.logwarn("[RECOVERY] moved to task_start_pose (ready)")
                time.sleep(1.5)
            else:
                rospy.logerr("[RECOVERY] move_to_task_start_pose failed")
            return ok
        except Exception as exc:
            rospy.logerr("[RECOVERY] move_to_task_start_pose exception: %s", str(exc))
            return False

    def cleanup_build_region(
        self,
        plan_file: str = "",
        config_file: str = "",
        resume_step: int = 1,
        failed_steps: Optional[List[int]] = None,
        execute_cleanup: bool = False,
        **unused,
    ) -> bool:
        """
        执行搭建区域清理。

        当 execute_cleanup=False（scan-only 模式）：
          - 调用 cleanup_planner.cleanup_scan_only()；
          - 只打印分类结果，不执行真实清理；
          - 必须返回 True，不阻塞后续 resume_from_step。

        当 execute_cleanup=True（真实清理模式）：
          - 调用 cleanup_executor.CleanupExecutorNode.execute_cleanup_cycle()；
          - 迭代执行 扫描→清理一个→重新扫描→再清理 循环；
          - 返回清理是否成功。
        """
        if not self._wait_for_cleanup_observe_pose(
            timeout=float(rospy.get_param("~cleanup_observe_precheck_timeout", 3.0)),
            stable_count=1,
            settle_after=0.2,
        ):
            rospy.logerr(
                "[RECOVERY] cleanup_build_region: arm not at cleanup_observe_pose. "
                "Refuse scan/cleanup to avoid wrong coordinate transform."
            )
            return False

        if not execute_cleanup:
            try:
                from scene_graph_system.recovery.cleanup_planner import cleanup_scan_only

                _ = cleanup_scan_only(
                    plan_file=plan_file,
                    config_file=config_file,
                    resume_step=int(resume_step),
                )
                rospy.logwarn("[RECOVERY] cleanup_build_region: scan-only completed (no real cleanup)")
                return True
            except Exception as exc:
                rospy.logwarn(
                    "[RECOVERY] cleanup_build_region scan-only failed: %s. "
                    "Continuing recovery anyway (scan-only is non-blocking).",
                    str(exc),
                )
                return True

        # 真实清理模式
        try:
            from scene_graph_system.recovery.cleanup_executor import CleanupExecutorNode

            executor = CleanupExecutorNode()
            result = executor.execute_cleanup_cycle(
                plan_file=plan_file,
                config_file=config_file,
                resume_step=int(resume_step),
                failed_steps=failed_steps,
            )

            success = bool(result.get("success", False))
            cleaned = int(result.get("cleaned_count", 0))
            failed = int(result.get("failed_count", 0))
            remaining = int(result.get("remaining_candidates", 0))

            rospy.logwarn(
                "[RECOVERY] cleanup_build_region: real cleanup done. "
                "cleaned=%d failed=%d remaining=%d success=%s",
                cleaned, failed, remaining, str(success),
            )

            if not success:
                rospy.logerr(
                    "[RECOVERY] cleanup failed/incomplete: cleaned=%d failed=%d remaining=%d. "
                    "Block resume_from_step.",
                    cleaned, failed, remaining,
                )
                return False

            if cleaned <= 0:
                rospy.logerr(
                    "[RECOVERY] cleanup executed but cleaned_count=0. Block resume_from_step."
                )
                return False

            return True

        except Exception as exc:
            rospy.logerr("[RECOVERY] cleanup_build_region exception: %s", str(exc))
            traceback_info = __import__("traceback").format_exc()
            rospy.logerr("[RECOVERY] traceback: %s", traceback_info)
            return False

    def cleanup_sorting_region(
        self,
        config_file: str = "",
        allowed_class: str = "cube",
        target_place: Optional[List[float]] = None,
        max_cleanup_count: int = 1,
        **unused,
    ) -> bool:
        """Remove a non-target-class object from the sorting/build region."""
        if not self._wait_for_cleanup_observe_pose(
            timeout=float(rospy.get_param("~cleanup_observe_precheck_timeout", 3.0)),
            stable_count=1,
            settle_after=0.2,
        ):
            rospy.logerr(
                "[RECOVERY] cleanup_sorting_region: arm not at cleanup_observe_pose. "
                "Refuse cleanup to avoid wrong coordinate transform."
            )
            return False

        try:
            from scene_graph_system.recovery.repair_executor import RepairExecutorNode

            executor = RepairExecutorNode()
            result = executor.execute_sorting_cleanup_cycle(
                config_file=config_file or DEFAULT_CLEANUP_CONFIG_FILE,
                allowed_class=str(allowed_class or "cube"),
                target_place=list(target_place or []),
                max_cleanup_count=int(max_cleanup_count),
            )
            success = bool(result.get("success", False))
            cleaned = int(result.get("cleaned_count", 0))
            failed = int(result.get("failed_count", 0))
            rospy.logwarn(
                "[RECOVERY] cleanup_sorting_region: cleaned=%d failed=%d success=%s",
                cleaned,
                failed,
                str(success),
            )
            return success and cleaned > 0 and failed == 0
        except Exception as exc:
            rospy.logerr("[RECOVERY] cleanup_sorting_region exception: %s", str(exc))
            return False

    def repair_build_region(
        self,
        plan_file: str = "",
        config_file: str = "",
        resume_step: int = 1,
        failed_steps: Optional[List[int]] = None,
        max_repair_count: Optional[int] = None,
        **unused,
    ) -> bool:
        """
        执行搭建区域修复：扫描 → 匹配候选 → 放置到 target_place。

        与 cleanup_build_region 的区别：
          - cleanup: 物体 → return_slot（回槽）
          - repair:  物体 → target_place（直接修复到位）

        优势：
          - target_place 通常在 z≈0.19，远低于 safe_z=0.34；
          - 不需要 return_slot 几何与去重；
          - 修复后场景即为正确状态。
        """
        if not self._wait_for_cleanup_observe_pose(
            timeout=float(rospy.get_param("~cleanup_observe_precheck_timeout", 3.0)),
            stable_count=1,
            settle_after=0.2,
        ):
            self.last_repair_result = {
                "success": False,
                "repaired_count": 0,
                "failed_count": 0,
                "repaired_steps": [],
                "failed_steps": list(failed_steps or []),
                "partial": False,
                "details": ["arm not at cleanup_observe_pose"],
            }
            rospy.logerr(
                "[RECOVERY] repair_build_region: arm not at cleanup_observe_pose. "
                "Refuse repair to avoid wrong coordinate transform."
            )
            return False

        try:
            from scene_graph_system.recovery.repair_executor import RepairExecutorNode

            executor = RepairExecutorNode()
            result = executor.execute_repair_cycle(
                plan_file=plan_file,
                config_file=config_file,
                resume_step=int(resume_step),
                failed_steps=failed_steps,
                max_repair_count=max_repair_count,
            )
            result = dict(result or {})
            self.last_repair_result = result

            success = bool(result.get("success", False))
            repaired = int(result.get("repaired_count", 0))
            failed = int(result.get("failed_count", 0))
            partial = bool(result.get("partial", False))
            repaired_steps = list(result.get("repaired_steps", []))
            failed_steps = list(result.get("failed_steps", []))

            rospy.logwarn(
                "[RECOVERY] repair_build_region: done. "
                "repaired=%d failed=%d partial=%s "
                "repaired_steps=%s failed_steps=%s success=%s",
                repaired, failed, str(partial),
                str(repaired_steps), str(failed_steps), str(success),
            )

            if not success:
                rospy.logerr(
                    "[RECOVERY] repair failed/incomplete: repaired=%d failed=%d "
                    "repaired_steps=%s failed_steps=%s partial=%s. "
                    "Block resume_from_step.",
                    repaired, failed,
                    str(repaired_steps), str(failed_steps), str(partial),
                )
                return False

            if repaired <= 0:
                rospy.logerr(
                    "[RECOVERY] repair executed but repaired_count=0. Block resume_from_step."
                )
                return False

            return True

        except Exception as exc:
            self.last_repair_result = {
                "success": False,
                "repaired_count": 0,
                "failed_count": 0,
                "repaired_steps": [],
                "failed_steps": list(failed_steps or []),
                "partial": False,
                "details": [str(exc)],
            }
            rospy.logerr("[RECOVERY] repair_build_region exception: %s", str(exc))
            traceback_info = __import__("traceback").format_exc()
            rospy.logerr("[RECOVERY] traceback: %s", traceback_info)
            return False

    def request_breakpoint_search(self, **context) -> dict:
        """
        请求验证节点搜索正确的断点。

        生成唯一 request_id，向 /recovery/request_breakpoint_search 发布请求，
        等待 /recovery/breakpoint_result 返回匹配 request_id 的结果。

        返回:
          {"success": bool, "resume_step": int | None}
        """
        self._recovery_request_counter += 1
        request_id = "bp_{}_{:.0f}".format(
            self._recovery_request_counter, time.time() * 1000
        )
        request_ts = float(rospy.Time.now().to_sec())
        run_id = str(context.get("run_id", "") or "")
        attempt_id = context.get("attempt_id")
        stage_seq = context.get("stage_seq")

        try:
            rospy.logwarn(
                "[RECOVERY] requesting breakpoint search: request_id=%s ts=%.3f",
                request_id, request_ts,
            )

            pub = rospy.Publisher(
                "/recovery/request_breakpoint_search",
                String,
                queue_size=1,
                latch=True,
            )
            rospy.sleep(0.3)
            payload = json.dumps({
                "request_id": request_id,
                "timestamp": request_ts,
                "run_id": run_id,
                "attempt_id": attempt_id,
                "stage_seq": stage_seq,
            }, ensure_ascii=False)
            pub.publish(String(data=payload))
            rospy.sleep(0.5)

            breakpoint_result_timeout = max(
                0.1,
                float(rospy.get_param("~breakpoint_result_timeout", 75.0)),
            )
            deadline = time.monotonic() + breakpoint_result_timeout
            while not rospy.is_shutdown() and time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    result_msg = rospy.wait_for_message(
                        "/recovery/breakpoint_result",
                        String,
                        timeout=min(remaining, 2.0),
                    )
                except Exception:
                    continue

                try:
                    result = json.loads(str(result_msg.data or "{}"))
                except Exception:
                    result = {}

                # 严格校验 request_id 和 timestamp
                result_request_id = str(result.get("request_id", ""))
                if result_request_id and result_request_id != request_id:
                    rospy.logwarn(
                        "[RECOVERY] breakpoint_result request_id mismatch "
                        "(got=%s expected=%s); ignoring stale latch message",
                        result_request_id, request_id,
                    )
                    continue

                if run_id and str(result.get("run_id", "") or "") != run_id:
                    rospy.logwarn("[RECOVERY] breakpoint_result run_id mismatch; ignoring")
                    continue
                if attempt_id is not None:
                    try:
                        if int(result.get("attempt_id")) != int(attempt_id):
                            rospy.logwarn("[RECOVERY] breakpoint_result attempt_id mismatch; ignoring")
                            continue
                    except (TypeError, ValueError):
                        continue

                result_ts = float(result.get("timestamp", 0))
                if result_ts < request_ts:
                    rospy.logwarn(
                        "[RECOVERY] breakpoint_result timestamp too old "
                        "(result_ts=%.3f request_ts=%.3f); ignoring",
                        result_ts, request_ts,
                    )
                    continue

                if not bool(result.get("success", False)):
                    rospy.logerr(
                        "[RECOVERY] breakpoint search failed: %s error_code=%s",
                        result.get("error", ""),
                        result.get("error_code", ""),
                    )
                    return {
                        "success": False,
                        "resume_step": None,
                        "error": result.get("error", ""),
                        "error_code": result.get("error_code", ""),
                    }

                found_step = result.get("resume_step")
                if found_step is not None:
                    rospy.logwarn(
                        "[RECOVERY] breakpoint found: resume_step=%s request_id=%s",
                        str(found_step), request_id,
                    )
                    return {"success": True, "resume_step": int(found_step)}
                else:
                    rospy.logerr("[RECOVERY] breakpoint result missing resume_step")
                    return {"success": False, "resume_step": None}

            rospy.logerr(
                "[RECOVERY] timeout waiting breakpoint_result: request_id=%s timeout=%.1fs",
                request_id,
                breakpoint_result_timeout,
            )
            return {
                "success": False,
                "resume_step": None,
                "error": "timeout waiting breakpoint_result",
                "error_code": "breakpoint_result_timeout",
            }

        except Exception as exc:
            rospy.logerr("[RECOVERY] request_breakpoint_search exception: %s", str(exc))
            return {
                "success": False,
                "resume_step": None,
                "error": str(exc),
                "error_code": "request_breakpoint_search_exception",
            }

    def resume_from_dynamic_step(self, resume_step: int) -> bool:
        """
        使用动态断点恢复执行。

        直接调用 resume_from_step()，不再等待 topic。
        resume_step 由 recovery_manager 从 _dynamic_resume_step 传入。
        """
        resume_step = int(resume_step)
        rospy.logwarn(
            "[RECOVERY] resume_from_dynamic_step: step=%d", resume_step,
        )
        return self.resume_from_step(resume_step)
