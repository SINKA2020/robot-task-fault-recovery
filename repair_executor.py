#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
repair_executor.py

修复执行器：将搭建区域中的故障物体直接放置到其目标位置（target_place）。

与 cleanup_executor 的区别：
  - cleanup: 物体 → return_slot（回槽）
  - repair:  物体 → target_place（直接修复到位）

优势：
  - target_place 通常在 z≈0.19，远低于 safe_z=0.34，彻底解决高 Z 超时问题；
  - 不需要维护 return_slot 几何与去重逻辑；
  - 修复后场景即为正确状态，断点搜索只需找到下一个未完成步骤。

流程：
  1. move_to_cleanup_observe_pose
  2. scan build_region
  3. 对每个 failed_step 按 class_name 匹配候选物体
  4. 执行 pick-place：抓取物体 → 放置到 target_place
  5. 返回结果
"""

from __future__ import annotations

from scene_graph_system.resources import package_resource_path

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

import rospy
from std_msgs.msg import String

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))



from scene_graph_system.recovery.cleanup_planner import (
    CleanupScanner,
    build_protected_slots,
    classify_objects,
    load_cleanup_config,
    load_action_steps,
    inside_build_region,
    distance,
)

# 底层动作原语
from scene_graph_system.robot.robot_motion_primitives import (
    arm_ready_pose,
    cleanup_observe_pose,
    get_default_grasp_quat,
    get_place_quaternion,
    gripper_open,
)

from scene_graph_system.robot.gripper_command_channel import (
    TOPIC_GRIPPER_STATE,
    wait_for_fresh_gripper_state,
)

# 公共夹爪阶段预期规则表
from scene_graph_system.robot.gripper_stage_expectations import check_gripper_states_for_stage, get_expected_gripper_states

# 公共 pick-place 流程
from scene_graph_system.robot.pick_place_executor import execute_pick_place_by_base_pose


DEFAULT_PLAN_FILE = package_resource_path('scripts/action.py')
DEFAULT_CONFIG_FILE = package_resource_path('config/cleanup_config.yaml')

TOPIC_REPAIR_FEEDBACK = "/recovery/repair_feedback"
TOPIC_REPAIR_RESULT = "/recovery/repair_result"


def select_foreign_sorting_candidate(
    candidates: List[dict],
    allowed_class: str,
    build_region: dict,
) -> Optional[dict]:
    """Select a stable non-target-class object inside the sorting region."""
    ignored_classes = {str(allowed_class), "gripper", "global_node"}
    matching = []

    for obj in list(candidates or []):
        class_name = str(obj.get("class_name", "") or "")
        position = obj.get("base_position")
        if not class_name or class_name in ignored_classes or position is None:
            continue
        if not inside_build_region(position, build_region):
            continue
        matching.append(obj)

    if not matching:
        return None

    matching.sort(
        key=lambda item: (
            -int(item.get("count", 0) or 0),
            -float(item.get("base_position", [0.0, 0.0, 0.0])[2]),
            str(item.get("class_name", "")),
        )
    )
    return matching[0]


class RepairExecutorNode:
    """
    修复执行器 ROS 节点。

    接收失败步骤列表，扫描搭建区域，将匹配的物体直接放置到其 target_place。
    """

    def __init__(self):
        self.feedback_pub = rospy.Publisher(TOPIC_REPAIR_FEEDBACK, String, queue_size=20)
        self.result_pub = rospy.Publisher(TOPIC_REPAIR_RESULT, String, queue_size=20)

        self.grasp_speed = float(rospy.get_param("~grasp_speed", 0.12))
        self.place_speed = float(rospy.get_param("~place_speed", 0.12))
        self.traverse_speed = float(rospy.get_param("~traverse_speed", 0.18))
        self.approach_offset = float(rospy.get_param("~approach_offset", 0.06))
        self.lift_offset = float(rospy.get_param("~lift_offset", 0.08))
        self.gripper_wait = float(rospy.get_param("~gripper_wait", 1.0))
        self.post_action_wait = float(rospy.get_param("~post_action_wait", 0.8))
        self.gripper_state_topic = str(
            rospy.get_param("~gripper_state_topic", TOPIC_GRIPPER_STATE)
        ).strip()
        self.gripper_state_max_age_sec = max(
            0.1, float(rospy.get_param("~gripper_state_max_age_sec", 1.0))
        )
        self.gripper_min_stable_count = max(
            1, int(rospy.get_param("~gripper_min_stable_count", 3))
        )

        self.settle_before_close = float(rospy.get_param("~settle_before_close", 1.0))
        self.gripper_close_wait = float(rospy.get_param("~gripper_close_wait", 2.5))
        self.settle_after_lift = float(rospy.get_param("~settle_after_lift", 0.3))

        self.repair_grasp_policy = {}
        self.repair_place_policy = {}
        self.motion_tolerance = float(rospy.get_param("~motion_tolerance", 0.020))
        self.motion_timeout = float(rospy.get_param("~motion_timeout", 30.0))

        self.repair_action_retry_enabled = bool(
            rospy.get_param("~repair_action_retry_enabled", True)
        )
        self.repair_action_retry_count = int(
            rospy.get_param("~repair_action_retry_count", 1)
        )

        # SDK 夹爪状态读取（优先），失败降级 ROS topic
        self.rm_ip = str(rospy.get_param("~realman_ip", "192.168.0.18")).strip()
        self.rm_port = int(rospy.get_param("~realman_port", 8080))
        self.rm_thread_mode = str(rospy.get_param("~realman_thread_mode", "RM_TRIPLE_MODE_E"))

        rospy.logwarn(
            "[REPAIR_EXEC] RepairExecutorNode initialized "
            "retry_enabled=%s retry_count=%d",
            str(self.repair_action_retry_enabled),
            self.repair_action_retry_count,
        )

    # ============================================================
    # 发布反馈
    # ============================================================

    def _publish_feedback(self, status: str, message: str, **details):
        payload = {
            "status": status,
            "message": message,
            "timestamp": time.time(),
            "details": details,
        }
        self.feedback_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        rospy.logwarn("[REPAIR_EXEC] %s: %s", status, message)

    def _publish_result(self, success: bool, summary: dict):
        payload = {
            "success": success,
            "summary": summary,
            "timestamp": time.time(),
        }
        self.result_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    # ============================================================
    # 策略读取
    # ============================================================

    def _get_repair_grasp_policy(self, class_name: str) -> dict:
        policies = getattr(self, "repair_grasp_policy", {}) or {}
        default_policy = policies.get("default", {}) or {}
        class_policy = policies.get(str(class_name), {}) or {}
        out = dict(default_policy)
        out.update(class_policy)
        return out

    def _get_repair_place_policy(self, class_name: str) -> dict:
        policies = getattr(self, "repair_place_policy", {}) or {}
        default_policy = policies.get("default", {}) or {}
        class_policy = policies.get(str(class_name), {}) or {}
        out = dict(default_policy)
        out.update(class_policy)
        return out

    # ============================================================
    # 候选选择
    # ============================================================

    def _select_candidate_for_target(
        self,
        candidates: List[dict],
        target_class: str,
        target_place: List[float],
        protected_slots: List[dict],
        build_region: dict,
        candidate_match_cfg: dict,
    ) -> Optional[dict]:
        """
        从候选物体中选择最匹配目标位置的物体。

        规则：
          1. class_name 必须匹配（allow_same_class_only=true）；
          2. 必须在 build_region 内；
          3. 不能在 protected_slot 内；
          4. 距离 target_place 在 [min_target_distance, max_target_distance] 内；
          5. 优先选择距离 target_place 最近的。
        """
        allow_same_class_only = bool(candidate_match_cfg.get("allow_same_class_only", True))
        min_dist = float(candidate_match_cfg.get("min_target_distance", 0.025))
        max_dist = float(candidate_match_cfg.get("max_target_distance", 0.45))

        matching = []
        for obj in candidates:
            if allow_same_class_only and str(obj.get("class_name", "")) != str(target_class):
                continue

            pos = obj.get("base_position")
            if pos is None:
                continue

            if not inside_build_region(pos, build_region):
                continue

            # 排除已落入保护槽的物体（属于已完成结构）
            from scene_graph_system.recovery.cleanup_planner import inside_slot
            in_protected = False
            for slot in protected_slots:
                if inside_slot(pos, slot):
                    in_protected = True
                    break
            if in_protected:
                continue

            dist = distance(pos, target_place)
            if dist < min_dist or dist > max_dist:
                continue

            matching.append((dist, obj))

        if not matching:
            return None

        matching.sort(key=lambda x: x[0])
        best = matching[0][1]
        rospy.logwarn(
            "[REPAIR_EXEC] selected candidate for class=%s at dist=%.3f pos=%s",
            target_class,
            matching[0][0],
            str([round(float(v), 4) for v in best["base_position"][:3]]),
        )
        return best

    # ============================================================
    # 夹爪监督（复用验证端阈值与逻辑）
    # ============================================================

    def _make_gripper_check_fn(self):
        """
        返回 (check_continue_fn, fail_reason) 用于 repair pick-place 夹爪监督。

        check_continue_fn 签名为 (stage_name, phase)，由 pick_place_executor 在
        每个阶段的 pre/run/post 检查点调用。

        规则（消费公共 GRIPPER_EXPECTATIONS 表）：
          pre:
            - close_gripper / open_gripper 强制检查，其余阶段跳过
          run:
            - 连续 2 次不满足预期 → 失败（防瞬时抖动）
          post:
            - 1.5s 短窗口确认：窗口内任意一次满足预期即通过

        夹爪状态读取：只消费独立 gripper_state_monitor 的稳定新鲜样本。
        状态不可用时不伪造成功；post 窗口会明确失败并停止修复动作。
        """
        fail_reason = [""]
        run_fail_count = [0]
        CONSECUTIVE = 2

        POST_WINDOW = 1.5
        POST_POLL_INTERVAL = 0.15
        CACHE_SEC = 0.10

        _cached_states = None
        _cached_ts = 0.0

        # --------------------------------------------------------
        def _read_gripper_states():
            """读取独立监控话题中的新鲜稳定状态。返回 set 或 None。"""
            nonlocal _cached_states, _cached_ts

            now = time.time()
            if _cached_states is not None and (now - _cached_ts) < CACHE_SEC:
                return _cached_states

            msg, _reason = wait_for_fresh_gripper_state(
                timeout_sec=0.25,
                max_age_sec=self.gripper_state_max_age_sec,
                topic=self.gripper_state_topic,
            )
            states = None
            if (
                msg is not None
                and int(msg.stable_count) >= self.gripper_min_stable_count
                and "unknown" not in set(msg.states or [])
            ):
                states = set(msg.states or [])

            if states is not None:
                _cached_states = states
                _cached_ts = now

            return states

        # --------------------------------------------------------
        def check(stage_name, phase=""):
            # ---- pre: 只检查 close_gripper 和 open_gripper ----
            if phase == "pre":
                if stage_name not in ("close_gripper", "open_gripper"):
                    return True
                return _check_single(stage_name, "pre", fail_reason)

            # ---- post: 短窗口确认 ----
            if phase == "post":
                return _check_post_window(stage_name, fail_reason)

            # ---- run: 连续确认 ----
            if phase == "run":
                return _check_run(stage_name, fail_reason)

            # 未知 phase，默认通过
            return True

        # --------------------------------------------------------
        def _check_single(stage_name, phase, fail_ref):
            observed = _read_gripper_states()
            if observed is None:
                return True  # 读不到状态 → 跳过，不误判

            passed, reason = check_gripper_states_for_stage(
                stage_name, phase, observed,
            )
            if not passed:
                fail_ref[0] = reason
                rospy.logerr("[REPAIR_EXEC] %s", reason)
                return False
            return True

        # --------------------------------------------------------
        def _check_post_window(stage_name, fail_ref):
            required = get_expected_gripper_states(stage_name, "post")
            if required is None:
                return True

            deadline = time.time() + POST_WINDOW
            last_reason = ""
            any_read = False

            while time.time() < deadline:
                observed = _read_gripper_states()
                if observed is None:
                    time.sleep(POST_POLL_INTERVAL)
                    continue

                any_read = True
                passed, reason = check_gripper_states_for_stage(
                    stage_name, "post", observed,
                )
                if passed:
                    return True
                last_reason = reason
                time.sleep(POST_POLL_INTERVAL)

            # 窗口内完全没有可靠状态：显式失败，禁止盲目继续修复。
            if not any_read:
                fail_ref[0] = "gripper_state_unavailable_in_post_window"
                rospy.logerr(
                    "[REPAIR_EXEC] post window stage=%s: "
                    "no stable gripper state in %.1fs",
                    stage_name, POST_WINDOW,
                )
                return False

            fail_ref[0] = last_reason or (
                f"post window expired stage={stage_name}"
            )
            rospy.logerr("[REPAIR_EXEC] %s", fail_ref[0])
            return False

        # --------------------------------------------------------
        def _check_run(stage_name, fail_ref):
            observed = _read_gripper_states()
            if observed is None:
                return True  # 读不到状态 → 跳过，不误判

            passed, reason = check_gripper_states_for_stage(
                stage_name, "run", observed,
            )

            if not passed:
                run_fail_count[0] += 1
                if run_fail_count[0] >= CONSECUTIVE:
                    fail_ref[0] = reason
                    rospy.logerr("[REPAIR_EXEC] run consecutive fail: %s", reason)
                    return False
            else:
                run_fail_count[0] = 0

            return True

        # --------------------------------------------------------
        def _clear_cache():
            nonlocal _cached_states, _cached_ts
            _cached_states = None
            _cached_ts = 0.0

        return check, fail_reason, _clear_cache

    def _reset_gripper_before_repair_retry(self):
        """retry 前打开夹爪，确保不会带着闭合夹爪移动。"""
        try:
            rospy.logwarn("[REPAIR_EXEC] reset gripper before retry")
            gripper_open()
            rospy.sleep(1.0)
            return True
        except Exception as exc:
            rospy.logerr("[REPAIR_EXEC] gripper open before retry failed: %s", str(exc))
            return False

    # ============================================================
    # 单个修复动作
    # ============================================================

    def _execute_single_repair(
        self,
        candidate: dict,
        target_place: List[float],
        grasp_quat: List[float],
        place_quat: List[float],
        safe_z: float,
        failure_reason_out: Optional[dict] = None,
    ) -> bool:
        """
        修复单个物体：从当前位置抓取，直接放置到 target_place。
        """
        grasp_position = candidate.get("base_position")
        if grasp_position is None:
            self._publish_feedback("failed", "candidate missing base_position")
            return False

        class_name = str(candidate.get("class_name", "unknown"))

        src_x, src_y, src_z = [float(v) for v in grasp_position[:3]]
        tgt_x, tgt_y, tgt_z = [float(v) for v in target_place[:3]]

        grasp_policy = self._get_repair_grasp_policy(class_name)
        place_policy = self._get_repair_place_policy(class_name)

        approach_x_offset = float(grasp_policy.get("approach_x_offset", 0.0))
        approach_y_offset = float(grasp_policy.get("approach_y_offset", 0.0))
        grasp_z_offset = float(grasp_policy.get("grasp_z_offset", 0.0))

        approach_z = max(float(safe_z), src_z + float(self.approach_offset))
        grasp_z = src_z + grasp_z_offset
        lift_z = max(float(safe_z), src_z + float(self.lift_offset))

        # 放置高度：由 repair_place_policy 控制
        place_z_offset = float(place_policy.get("place_z_offset", 0.018))
        target_above_z = float(place_policy.get("target_above_z", float(safe_z)))
        place_z = tgt_z + place_z_offset
        leave_z = float(place_policy.get("leave_z", target_above_z))

        if target_above_z < place_z:
            rospy.logerr(
                "[REPAIR_EXEC] invalid repair_place_policy: "
                "target_above_z %.4f < place_z %.4f",
                target_above_z,
                place_z,
            )
            return False

        if leave_z < place_z:
            rospy.logerr(
                "[REPAIR_EXEC] invalid repair_place_policy: "
                "leave_z %.4f < place_z %.4f",
                leave_z,
                place_z,
            )
            return False

        rospy.logwarn(
            "[REPAIR_EXEC] height plan: approach_z=%.4f grasp_z=%.4f lift_z=%.4f "
            "target_above_z=%.4f place_z=%.4f leave_z=%.4f",
            approach_z, grasp_z, lift_z,
            target_above_z, place_z, leave_z,
        )

        rospy.logwarn(
            "[REPAIR_EXEC] repairing class=%s src=(%.4f,%.4f,%.4f) tgt=(%.4f,%.4f,%.4f)",
            class_name, src_x, src_y, src_z, tgt_x, tgt_y, tgt_z,
        )

        self._publish_feedback(
            "running",
            "repair pick-place for %s" % class_name,
            phase="pick_place",
            class_name=class_name,
            source=[src_x, src_y, src_z],
            target=[tgt_x, tgt_y, tgt_z],
        )

        check_fn, fail_reason, clear_cache = self._make_gripper_check_fn()

        # 每次 pick-place 前确保夹爪张开，清缓存避免读到旧闭合状态
        try:
            gripper_open()
            clear_cache()
            rospy.sleep(1.0)
        except Exception as exc:
            rospy.logerr("[REPAIR_EXEC] pre-repair gripper open failed: %s", str(exc))
            if failure_reason_out is not None:
                failure_reason_out["reason"] = "gripper_open_before_repair_failed"
            return False

        ok = execute_pick_place_by_base_pose(
            source_base_position=grasp_position,
            target_base_position=target_place,
            grasp_quat=grasp_quat,
            place_quat=place_quat,
            object_class=class_name,
            obj_angle=0.0,
            mode="repair",
            stage_prefix="repair_%s" % class_name,

            check_continue_fn=check_fn,

            approach_z=approach_z,
            grasp_z=grasp_z,
            lift_z=lift_z,
            target_above_z=target_above_z,
            place_z=place_z,
            leave_z=leave_z,

            approach_x_offset=approach_x_offset,
            approach_y_offset=approach_y_offset,

            move_speed=self.traverse_speed,
            descend_speed=self.grasp_speed,
            post_wait=self.post_action_wait,

            settle_before_close=self.settle_before_close,
            gripper_close_wait=self.gripper_close_wait,
            settle_after_lift=self.settle_after_lift,

            return_to_ready=False,

            wait_motion_done=True,
            motion_tolerance=self.motion_tolerance,
            motion_timeout=self.motion_timeout,

            test_phase="full",

            after_close_gripper_fn=None,
            after_lift_fn=None,
            after_open_gripper_fn=None,

            failure_reason_out=failure_reason_out,
        )

        if not ok:
            if failure_reason_out is not None and fail_reason[0]:
                failure_reason_out["reason"] = fail_reason[0]
            self._publish_feedback(
                "failed",
                "repair pick-place failed for %s: %s" % (class_name, fail_reason[0] or "unknown"),
                class_name=class_name,
            )
            return False

        self._publish_feedback(
            "running",
            "repaired %s → target_place" % class_name,
            phase="place_done",
            class_name=class_name,
        )
        return True

    # ============================================================
    # 扫描
    # ============================================================

    def _scan_build_region(
        self,
        cfg: dict,
        build_region: dict,
    ) -> List[dict]:
        """
        扫描搭建区域，返回候选物体列表。
        """
        scan_seconds = float(cfg.get("scan_seconds", 2.5))
        scanner = CleanupScanner(cfg)
        objects = scanner.scan(scan_seconds)

        rospy.logwarn(
            "[REPAIR_EXEC] scan: %d objects detected",
            len(objects),
        )
        return objects

    def execute_sorting_cleanup_cycle(
        self,
        config_file: str,
        allowed_class: str,
        target_place: List[float],
        max_cleanup_count: int = 1,
    ) -> Dict[str, Any]:
        """Move non-``allowed_class`` objects out of the existing build ROI."""
        cfg = load_cleanup_config(config_file)
        build_region = dict(cfg.get("build_region", {}) or {})
        if not build_region:
            return {
                "success": False,
                "cleaned_count": 0,
                "failed_count": 0,
                "details": ["build_region is not configured"],
            }

        if not isinstance(target_place, (list, tuple)) or len(target_place) < 3:
            return {
                "success": False,
                "cleaned_count": 0,
                "failed_count": 0,
                "details": ["target_place must contain x, y, z"],
            }

        self.repair_grasp_policy = cfg.get("repair_grasp_policy", {}) or {}
        self.repair_place_policy = cfg.get("repair_place_policy", {}) or {}
        grasp_quat = get_default_grasp_quat()
        place_quat = get_place_quaternion()
        safe_z = self._compute_safe_z(cfg, [])
        cleanup_limit = max(1, int(max_cleanup_count))
        cleaned_count = 0
        failed_count = 0
        details = []

        for iteration in range(1, cleanup_limit + 1):
            candidates = self._scan_build_region(cfg, build_region)
            candidate = select_foreign_sorting_candidate(
                candidates,
                allowed_class=allowed_class,
                build_region=build_region,
            )
            if candidate is None:
                break

            class_name = str(candidate.get("class_name", "unknown"))
            source_position = list(candidate.get("base_position", []) or [])
            failure_reason = {}
            ok = self._execute_single_repair(
                candidate,
                list(target_place[:3]),
                grasp_quat,
                place_quat,
                safe_z,
                failure_reason_out=failure_reason,
            )
            if not ok:
                failed_count += 1
                details.append({
                    "iteration": iteration,
                    "class_name": class_name,
                    "source_position": source_position,
                    "status": "failed",
                    "reason": str(failure_reason.get("reason", "") or "pick_place_failed"),
                })
                break

            cleaned_count += 1
            details.append({
                "iteration": iteration,
                "class_name": class_name,
                "source_position": source_position,
                "target_place": list(target_place[:3]),
                "status": "cleaned",
            })

            if iteration < cleanup_limit:
                cleanup_observe_pose()
                self._wait_observe_pose_settled()

        try:
            arm_ready_pose()
            time.sleep(1.5)
        except Exception as exc:
            rospy.logwarn("[SORT_CLEANUP] failed to move to ready pose: %s", str(exc))

        success = cleaned_count > 0 and failed_count == 0
        summary = {
            "success": success,
            "cleaned_count": cleaned_count,
            "failed_count": failed_count,
            "allowed_class": str(allowed_class),
            "target_place": list(target_place[:3]),
            "details": details,
        }
        self._publish_result(success, summary)
        return summary

    # ============================================================
    # 主修复循环
    # ============================================================

    def execute_repair_cycle(
        self,
        plan_file: str,
        config_file: str,
        resume_step: int,
        failed_steps: Optional[List[int]] = None,
        max_repair_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        修复主循环：按 step 升序逐个修复，每步重新扫描。

        对每个 failed_step：
          1. 计算当前保护区（已完成步骤 + 已修复步骤）；
          2. 重新扫描搭建区域；
          3. 匹配候选物体；
          4. 执行 pick-place 将物体放置到 target_place；
          5. 成功则推进保护区、回到观测位姿，继续下一步；
          6. 失败则停止（后续步骤依赖前序结构）。

        返回:
          {
            "success": bool,
            "repaired_count": int,
            "failed_count": int,
            "details": [...],
          }
        """
        cfg = load_cleanup_config(config_file)
        self.repair_grasp_policy = cfg.get("repair_grasp_policy", {}) or {}
        self.repair_place_policy = cfg.get("repair_place_policy", {}) or {}

        repair_cfg = cfg.get("repair", {}) or {}
        build_region = repair_cfg.get("region", None)
        if build_region is None:
            build_region = cfg.get("build_region", {})
        candidate_match_cfg = repair_cfg.get("candidate_match", {}) or {}
        if max_repair_count is None:
            max_repair_count = int(repair_cfg.get("max_repair_count", 1))
        else:
            max_repair_count = int(max_repair_count)

        grasp_quat = get_default_grasp_quat()
        place_quat = get_place_quaternion()

        # 加载 action.py 步骤，获取 target_place
        all_steps = load_action_steps(plan_file)
        step_map = {int(s["step"]): s for s in all_steps}

        if failed_steps is None:
            failed_steps = []

        if not failed_steps:
            self._publish_feedback("succeeded", "no failed steps to repair")
            self._publish_result(True, {"repaired": 0, "failed": 0})
            return {
                "success": True,
                "repaired_count": 0,
                "failed_count": 0,
                "repaired_steps": [],
                "failed_steps": [],
                "partial": False,
                "details": ["no failed steps"],
            }

        # _protected_until：当前已确认完成的步骤边界。
        # build_protected_slots(plan_file, _protected_until, cfg) 保护 step < _protected_until。
        # 初始 = min(failed_steps)，即原始 resume_step（前序步骤都已正确）。
        # 每修完一步，推进到 step_num + 1，防止刚修复的物体被后续扫描选中。
        _protected_until = min(failed_steps)
        repaired_steps: List[int] = []
        repaired_count = 0
        failed_count = 0
        details = []

        self._publish_feedback(
            "started",
            f"repair cycle: {len(failed_steps)} failed steps, protected_until={_protected_until}",
            failed_steps=sorted(failed_steps),
        )

        for step_num in sorted(failed_steps):
            if repaired_count >= max_repair_count:
                rospy.logwarn(
                    "[REPAIR_EXEC] reached max_repair_count=%d, stop; "
                    "repaired=%d remaining_failed=%s",
                    max_repair_count, repaired_count,
                    str([s for s in sorted(failed_steps) if s not in repaired_steps]),
                )
                break

            step_info = step_map.get(int(step_num))
            if step_info is None:
                rospy.logwarn("[REPAIR_EXEC] step %d not found in action plan, skip", step_num)
                failed_count += 1
                details.append({
                    "step": step_num,
                    "status": "skipped",
                    "reason": "step not in action plan",
                })
                break

            target_class = str(step_info.get("class_name", ""))
            target_place = step_info.get("target_place")
            if target_place is None or len(target_place) < 3:
                rospy.logwarn("[REPAIR_EXEC] step %d has no target_place, skip", step_num)
                failed_count += 1
                details.append({
                    "step": step_num,
                    "status": "skipped",
                    "reason": "no target_place",
                })
                break

            # --- 每步重建保护区并重扫 ---
            protected_slots = build_protected_slots(plan_file, _protected_until, cfg)
            safe_z = self._compute_safe_z(cfg, protected_slots)

            rospy.logwarn(
                "[REPAIR_EXEC] step=%d: protected_until=%d safe_z=%.3f scanning...",
                step_num, _protected_until, safe_z,
            )

            candidates = self._scan_build_region(cfg, build_region)

            candidate = self._select_candidate_for_target(
                candidates, target_class, target_place, protected_slots,
                build_region, candidate_match_cfg,
            )

            if candidate is None:
                rospy.logwarn(
                    "[REPAIR_EXEC] no candidate found for step=%d class=%s",
                    step_num, target_class,
                )
                failed_count += 1
                details.append({
                    "step": step_num,
                    "class_name": target_class,
                    "status": "no_candidate",
                })
                break

            rospy.logwarn(
                "[REPAIR_EXEC] repairing step=%d class=%s → target=%s",
                step_num, target_class,
                str([round(float(v), 4) for v in target_place[:3]]),
            )

            ok = False
            failure_reason = ""
            max_retry = (
                self.repair_action_retry_count
                if self.repair_action_retry_enabled
                else 0
            )

            for attempt in range(max_retry + 1):
                reason_out: dict = {}
                ok = self._execute_single_repair(
                    candidate, target_place, grasp_quat, place_quat, safe_z,
                    failure_reason_out=reason_out,
                )
                failure_reason = str(reason_out.get("reason", "") or "")
                if ok:
                    break

                rospy.logwarn(
                    "[REPAIR_EXEC] step=%d attempt %d/%d failed: %s",
                    step_num, attempt + 1, max_retry + 1, failure_reason,
                )

                if attempt < max_retry:
                    # 打开夹爪 → 回观测位姿 → 重扫 → 重选候选
                    if not self._reset_gripper_before_repair_retry():
                        failure_reason = "gripper_open_before_retry_failed"
                        break

                    try:
                        cleanup_observe_pose()
                        self._wait_observe_pose_settled()
                    except Exception as exc:
                        rospy.logwarn(
                            "[REPAIR_EXEC] retry return to observe pose failed: %s",
                            str(exc),
                        )

                    # 重扫前更新保护区（已修复的步骤已纳入）
                    protected_slots = build_protected_slots(
                        plan_file, _protected_until, cfg,
                    )
                    candidates = self._scan_build_region(cfg, build_region)
                    candidate = self._select_candidate_for_target(
                        candidates, target_class, target_place,
                        protected_slots, build_region, candidate_match_cfg,
                    )

                    if candidate is None:
                        failure_reason = "no_candidate_after_retry"
                        rospy.logwarn(
                            "[REPAIR_EXEC] step=%d no candidate after retry, stop",
                            step_num,
                        )
                        break

            if not ok:
                failed_count += 1
                details.append({
                    "step": step_num,
                    "class_name": target_class,
                    "status": "failed",
                    "reason": failure_reason,
                })
                rospy.logerr(
                    "[REPAIR_EXEC] step %d repair FAILED after %d attempt(s): %s, stop",
                    step_num, max_retry + 1, failure_reason,
                )
                break

            repaired_count += 1
            repaired_steps.append(step_num)
            details.append({
                "step": step_num,
                "class_name": target_class,
                "target_place": [round(float(v), 4) for v in target_place[:3]],
                "status": "repaired",
            })

            # 推进保护区：刚修好的步骤也纳入保护
            _protected_until = max(_protected_until, step_num + 1)
            rospy.logwarn(
                "[REPAIR_EXEC] step %d repaired, protected_until → %d",
                step_num, _protected_until,
            )

            # 回到观测位姿，等待到位后为下一步重扫做准备
            try:
                cleanup_observe_pose()
                self._wait_observe_pose_settled()
            except Exception as exc:
                rospy.logwarn("[REPAIR_EXEC] return to observe pose failed: %s", str(exc))

        # 回到任务起始位姿
        try:
            arm_ready_pose()
            time.sleep(1.5)
        except Exception as exc:
            rospy.logwarn("[REPAIR_EXEC] failed to move to ready pose: %s", str(exc))

        # 收集未修复的步骤（原 failed_steps 中未出现在 repaired_steps 的）
        failed_step_list = [
            int(s) for s in sorted(failed_steps)
            if int(s) not in repaired_steps
        ]
        partial = repaired_count > 0 and failed_count > 0

        # Only report success when every requested failed step was repaired.
        success = failed_count == 0 and repaired_count > 0 and len(failed_step_list) == 0

        summary = {
            "repaired": repaired_count,
            "failed": failed_count,
            "repaired_steps": repaired_steps,
            "failed_steps": failed_step_list,
            "partial": partial,
        }

        self._publish_result(success, summary)

        rospy.logwarn(
            "[REPAIR_EXEC] cycle done: repaired=%d failed=%d "
            "repaired_steps=%s failed_steps=%s partial=%s success=%s",
            repaired_count, failed_count,
            str(repaired_steps), str(failed_step_list),
            str(partial), str(success),
        )

        return {
            "success": success,
            "repaired_count": repaired_count,
            "failed_count": failed_count,
            "repaired_steps": repaired_steps,
            "failed_steps": failed_step_list,
            "partial": partial,
            "details": details,
        }

    def _compute_safe_z(self, cfg: dict, protected_slots: list) -> float:
        config_safe_z = float(cfg.get("safe_z", 0.34))
        safety_margin = float(cfg.get("cleanup_safe_z_margin", 0.08))

        max_structure_z = 0.0
        for slot in protected_slots:
            center = slot.get("center", [0, 0, 0])
            margin = slot.get("margin", [0.05, 0.05, 0.05])
            top = float(center[2]) + float(margin[2])
            if top > max_structure_z:
                max_structure_z = top

        safe = max(config_safe_z, max_structure_z + safety_margin)
        rospy.logwarn("[REPAIR_EXEC] safe_z=%.3f (config=%.3f max_structure=%.3f margin=%.3f)",
                      safe, config_safe_z, max_structure_z, safety_margin)
        return safe

    def _wait_observe_pose_settled(self, timeout: float = 8.0) -> bool:
        """
        等待机械臂实际到达 cleanup observe pose（XYZ 检查）。

        多步 repair 中，每修完一步需要回到观测位姿再扫描下一步。
        仅调用 cleanup_observe_pose() 下发运动指令不够，必须等 TCP 到位，
        否则腕上相机坐标变换仍使用中间位姿，导致候选坐标错误。
        """
        expected_xyz = [-0.004, 0.063, 0.465]
        xyz_tolerance = float(rospy.get_param("~observe_xyz_tolerance", 0.030))

        # 尝试导入 ArmState 做真实位姿检查；不可用时退化为固定等待
        ArmState = None
        try:
            from rm_msgs.msg import ArmState  # noqa: F811
        except Exception:
            pass

        if ArmState is None:
            rospy.logwarn(
                "[REPAIR_EXEC] rm_msgs not available, settle wait %.1fs",
                timeout * 0.4,
            )
            rospy.sleep(timeout * 0.4)
            return True

        arm_state_topic = str(
            rospy.get_param("~arm_state_topic", "/rm_driver/ArmCurrentState")
        )

        deadline = time.time() + timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            try:
                msg = rospy.wait_for_message(
                    arm_state_topic, ArmState, timeout=1.0,
                )
            except Exception:
                continue

            pose = getattr(msg, "Pose", None)
            if pose is None:
                continue

            x = y = z = None
            try:
                values = list(pose)
                if len(values) >= 3:
                    x, y, z = float(values[0]), float(values[1]), float(values[2])
            except Exception:
                if hasattr(pose, "position"):
                    x = float(pose.position.x)
                    y = float(pose.position.y)
                    z = float(pose.position.z)

            if x is None:
                continue

            err = math.sqrt(
                (x - expected_xyz[0]) ** 2
                + (y - expected_xyz[1]) ** 2
                + (z - expected_xyz[2]) ** 2
            )

            if err <= xyz_tolerance:
                rospy.logwarn(
                    "[REPAIR_EXEC] observe pose reached: xyz=(%.4f,%.4f,%.4f) err=%.4f",
                    x, y, z, err,
                )
                return True

        rospy.logwarn(
            "[REPAIR_EXEC] timeout waiting observe pose; proceeding anyway"
        )
        return False


# ============================================================
# 独立运行入口（调试用）
# ============================================================

def main():
    rospy.init_node("repair_executor", anonymous=True)

    plan_file = str(rospy.get_param("~plan_file", DEFAULT_PLAN_FILE))
    config_file = str(rospy.get_param("~config_file", DEFAULT_CONFIG_FILE))
    resume_step = int(rospy.get_param("~resume_step", 1))
    failed_steps_json = str(rospy.get_param("~failed_steps", "[]")).strip()

    failed_steps = None
    try:
        parsed = json.loads(failed_steps_json)
        if isinstance(parsed, list) and parsed:
            failed_steps = [int(v) for v in parsed]
    except (json.JSONDecodeError, ValueError, TypeError):
        failed_steps = None

    executor = RepairExecutorNode()
    result = executor.execute_repair_cycle(
        plan_file=plan_file,
        config_file=config_file,
        resume_step=resume_step,
        failed_steps=failed_steps,
    )
    rospy.logwarn("[REPAIR_EXEC] final result: %s", json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
