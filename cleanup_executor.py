#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cleanup_executor.py

清理执行器：在恢复链路中执行真实清理动作。

设计准则（参见 spec 12 条）：
  1. 臂上相机定位，不依赖外部相机坐标；
  2. 清理前必须确定保护区；
  3. 7 条规则严格筛选候选；
  4. 同类别互换，return_slot 不重复使用；
  5. source_pose 必须是 base 坐标；
  6. 迭代执行：scan → 清一个 → rescan → 再清下一个；
  7. 优先级：远离保护区 > 高 z > 稳定检测；
  8. 保守路径：高空横移 + 竖直升降；
  9. 清理过程受监督（动作监督 + 结果监督）；
  10. execute_cleanup=false 时不阻塞；
  11. 清理在 resume_from_step 之前完成；
  12. 配置参数可调。

清理流程：
  move_to_cleanup_observe_pose
  → scan
  → 选最优候选
  → 清理一个（高空横移 → 竖直下降抓取 → 竖直上抬 → 高空移动到 return_slot → 竖直下降放置 → 竖直上抬）
  → 回 cleanup_observe_pose
  → rescan
  → 循环直到无候选或无可用的 return_slot
  → move_to_task_start_pose
"""

from __future__ import annotations

from scene_graph_system.resources import package_resource_path

import json
import math
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import rospy
from std_msgs.msg import String

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))



from scene_graph_system.recovery.cleanup_planner import (
    CleanupScanner,
    build_protected_slots,
    classify_objects,
    cleanup_scan_and_plan,
    load_cleanup_config,
    distance,
)
from scene_graph_system.recovery.cleanup_supervisor import CleanupSupervisor

# 底层动作原语
from scene_graph_system.robot.robot_motion_primitives import (
    arm_ready_pose,
    cleanup_observe_pose,
    get_default_grasp_quat,
    get_place_quaternion,
)

# 公共 pick-place 流程
from scene_graph_system.robot.pick_place_executor import execute_pick_place_by_base_pose


DEFAULT_PLAN_FILE = package_resource_path('scripts/action.py')
DEFAULT_CONFIG_FILE = package_resource_path('config/cleanup_config.yaml')

TOPIC_CLEANUP_FEEDBACK = "/recovery/cleanup_feedback"
TOPIC_CLEANUP_RESULT = "/recovery/cleanup_result"


# ============================================================
# 保守清理路径
# ============================================================

def _compute_safe_z(cfg: dict, protected_slots: list) -> float:
    """
    计算安全高度：当前最高保护结构高度 + 安全余量。

    safe_z > max(protected_slot.center.z + margin_z) + safety_margin
    """
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
    rospy.logwarn("[CLEANUP_EXEC] safe_z=%.3f (config=%.3f max_structure=%.3f margin=%.3f)",
                  safe, config_safe_z, max_structure_z, safety_margin)
    return safe


# ============================================================
# 清理执行器节点
# ============================================================


class CleanupExecutorNode:
    """
    清理执行器 ROS 节点。

    接收 /recovery/cleanup_plan 话题上的清理计划，
    迭代执行清理动作，发布 /recovery/cleanup_feedback 和 /recovery/cleanup_result。
    """

    def __init__(self):
        self.supervisor = CleanupSupervisor()
        self.feedback_pub = rospy.Publisher(TOPIC_CLEANUP_FEEDBACK, String, queue_size=20)
        self.result_pub = rospy.Publisher(TOPIC_CLEANUP_RESULT, String, queue_size=20)

        # 可配置参数
        self.grasp_speed = float(rospy.get_param("~grasp_speed", 0.12))
        self.place_speed = float(rospy.get_param("~place_speed", 0.12))
        self.traverse_speed = float(rospy.get_param("~traverse_speed", 0.18))
        self.approach_offset = float(rospy.get_param("~approach_offset", 0.06))
        self.lift_offset = float(rospy.get_param("~lift_offset", 0.08))
        self.release_z_offset = float(rospy.get_param("~release_z_offset", 0.035))
        self.gripper_wait = float(rospy.get_param("~gripper_wait", 1.0))
        self.post_action_wait = float(rospy.get_param("~post_action_wait", 0.8))

        self.settle_before_close = float(rospy.get_param("~settle_before_close", 1.0))
        self.gripper_close_wait = float(rospy.get_param("~gripper_close_wait", 2.5))
        self.settle_after_lift = float(rospy.get_param("~settle_after_lift", 0.3))

        # 清理抓取偏移（独立于正常搭建的 -0.055/+0.03）
        self.cleanup_grasp_policy = {}
        self.cleanup_return_policy = {}
        self.place_above_offset = float(rospy.get_param("~place_above_offset", 0.10))
        self.motion_tolerance = float(rospy.get_param("~motion_tolerance", 0.020))
        self.motion_timeout = float(rospy.get_param("~motion_timeout", 30.0))

        rospy.logwarn("[CLEANUP_EXEC] CleanupExecutorNode initialized")

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
        rospy.logwarn("[CLEANUP_EXEC] %s: %s", status, message)

    def _publish_result(self, success: bool, summary: dict):
        payload = {
            "success": success,
            "summary": summary,
            "timestamp": time.time(),
        }
        self.result_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    # ============================================================
    # 单个清理动作
    # ============================================================

    def _get_cleanup_grasp_policy(self, class_name: str) -> dict:
        """
        按类别读取 cleanup_grasp_policy。

        cleanup_config.yaml 示例：
          cleanup_grasp_policy:
            default:
              approach_x_offset: 0.0
              approach_y_offset: 0.0
              grasp_z_offset: 0.0
            arch:
              approach_x_offset: -0.008
              approach_y_offset: 0.045
              grasp_z_offset: 0.023
        """
        policies = getattr(self, "cleanup_grasp_policy", {}) or {}

        default_policy = policies.get("default", {}) or {}
        class_policy = policies.get(str(class_name), {}) or {}

        out = dict(default_policy)
        out.update(class_policy)

        return out

    def _get_cleanup_return_policy(self, class_name: str) -> dict:
        """
        按类别读取 cleanup_return_policy。

        return_slot 的 x/y 仍来自 action.py source_pose；
        清理放置阶段的 z 高度由 cleanup_return_policy 单独控制。
        """
        policies = getattr(self, "cleanup_return_policy", {}) or {}

        default_policy = policies.get("default", {}) or {}
        class_policy = policies.get(str(class_name), {}) or {}

        out = dict(default_policy)
        out.update(class_policy)

        return out

    def _execute_single_cleanup(
        self,
        candidate: dict,
        grasp_quat: List[float],
        place_quat: List[float],
        safe_z: float,
    ) -> bool:
        """
        清理单个物体：调用公共 pick-place 流程。

        新版要点：
        - 按 class_name 从 cleanup_grasp_policy 读取抓取偏移；
        - source 仍然来自 cleanup_planner 的 candidate["base_position"]；
        - 不固定 source 坐标；
        - grasp/lift/approach 均通过公共 pick_place_executor 执行；
        - close_gripper 前会等待 TCP 到 grasp 位置。
        """
        grasp_position = candidate.get("base_position")
        return_slot = candidate.get("assigned_return_slot")

        if grasp_position is None or return_slot is None:
            self._publish_feedback("failed", "missing grasp_position or return_slot")
            return False

        place_position = return_slot.get("position")
        if place_position is None:
            self._publish_feedback("failed", "return_slot missing position")
            return False

        class_name = str(candidate.get("class_name", "unknown"))

        src_x, src_y, src_z = [float(v) for v in grasp_position[:3]]
        tgt_x, tgt_y, tgt_z = [float(v) for v in place_position[:3]]

        policy = self._get_cleanup_grasp_policy(class_name)

        approach_x_offset = float(policy.get("approach_x_offset", 0.0))
        approach_y_offset = float(policy.get("approach_y_offset", 0.0))
        grasp_z_offset = float(policy.get("grasp_z_offset", 0.0))

        approach_z = max(float(safe_z), src_z + float(self.approach_offset))
        grasp_z = src_z + grasp_z_offset

        # 清理放置高度：由 cleanup_return_policy 独立控制，
        # 不再从 return_slot.position[2]（action.py source_pose.z）推算。
        return_policy = self._get_cleanup_return_policy(class_name)

        default_target_above_z = max(
            float(safe_z),
            tgt_z + float(self.place_above_offset),
        )
        default_place_z = tgt_z + float(self.release_z_offset)
        default_leave_z = default_target_above_z

        target_above_z = float(
            return_policy.get("target_above_z", default_target_above_z)
        )
        place_z = float(
            return_policy.get("place_z", default_place_z)
        )
        leave_z = float(
            return_policy.get("leave_z", default_leave_z)
        )

        # 基本安全校验
        if target_above_z < place_z:
            rospy.logerr(
                "[CLEANUP_EXEC] invalid cleanup_return_policy: "
                "target_above_z %.4f < place_z %.4f",
                target_above_z,
                place_z,
            )
            self._publish_feedback(
                "failed",
                "invalid cleanup_return_policy: target_above_z < place_z",
                class_name=class_name,
                target_above_z=target_above_z,
                place_z=place_z,
            )
            return False

        if leave_z < place_z:
            rospy.logerr(
                "[CLEANUP_EXEC] invalid cleanup_return_policy: "
                "leave_z %.4f < place_z %.4f",
                leave_z,
                place_z,
            )
            self._publish_feedback(
                "failed",
                "invalid cleanup_return_policy: leave_z < place_z",
                class_name=class_name,
                leave_z=leave_z,
                place_z=place_z,
            )
            return False

        # 抓取后只升到安全离场高度，不再强制升到 return_slot 产生的高 z。
        lift_z = max(
            float(safe_z),
            src_z + float(self.lift_offset),
        )

        rospy.logwarn(
            "[CLEANUP_EXEC] height plan: approach_z=%.4f grasp_z=%.4f lift_z=%.4f "
            "target_above_z=%.4f place_z=%.4f leave_z=%.4f return_slot_z=%.4f",
            approach_z,
            grasp_z,
            lift_z,
            target_above_z,
            place_z,
            leave_z,
            tgt_z,
        )

        rospy.logwarn(
            "[CLEANUP_EXEC] cleaning class=%s source=(%.4f,%.4f,%.4f) target=(%.4f,%.4f,%.4f) "
            "policy={x=%.4f,y=%.4f,z=%.4f} safe_z=%.4f grasp=(%.4f,%.4f,%.4f)",
            class_name,
            src_x, src_y, src_z,
            tgt_x, tgt_y, tgt_z,
            approach_x_offset,
            approach_y_offset,
            grasp_z_offset,
            float(safe_z),
            src_x + approach_x_offset,
            src_y + approach_y_offset,
            grasp_z,
        )

        self._publish_feedback(
            "running",
            "cleanup pick-place for %s" % class_name,
            phase="pick_place",
            class_name=class_name,
            source=[src_x, src_y, src_z],
            target=[tgt_x, tgt_y, tgt_z],
            policy={
                "approach_x_offset": approach_x_offset,
                "approach_y_offset": approach_y_offset,
                "grasp_z_offset": grasp_z_offset,
            },
        )

        ok = execute_pick_place_by_base_pose(
            source_base_position=grasp_position,
            target_base_position=place_position,
            grasp_quat=grasp_quat,
            place_quat=place_quat,
            object_class=class_name,
            obj_angle=0.0,
            mode="cleanup",
            stage_prefix="cleanup_%s" % class_name,

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

            # gripper 字段稳定前先不启用监督 hook
            after_close_gripper_fn=None,
            after_lift_fn=None,
            after_open_gripper_fn=None,
        )

        if not ok:
            self._publish_feedback(
                "failed",
                "pick-place failed for %s" % class_name,
                class_name=class_name,
            )
            return False

        self._publish_feedback(
            "running",
            "placed %s at return_slot" % class_name,
            phase="place_done",
            class_name=class_name,
        )

        return True

    # ============================================================
    # 主清理循环
    # ============================================================

    def execute_cleanup_cycle(
        self,
        plan_file: str,
        config_file: str,
        resume_step: int,
        failed_steps: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        清理主循环：迭代 scan → 清一个 → rescan。

        返回:
          {
            "success": bool,
            "cleaned_count": int,
            "failed_count": int,
            "remaining_candidates": int,
            "details": [...],
          }
        """
        cfg = load_cleanup_config(config_file)
        self.cleanup_grasp_policy = cfg.get("cleanup_grasp_policy", {}) or {}
        self.cleanup_return_policy = cfg.get("cleanup_return_policy", {}) or {}
        grasp_quat = get_default_grasp_quat()
        place_quat = get_place_quaternion()
        cleaned_count = 0
        failed_count = 0
        details = []

        # 初始扫描与规划
        plan = cleanup_scan_and_plan(
            plan_file=plan_file,
            config_file=config_file,
            resume_step=resume_step,
            failed_steps=failed_steps,
            execute_cleanup=True,
        )

        if plan.get("summary", "").startswith("SAFETY_ABORT"):
            self._publish_feedback("failed", plan["summary"])
            return {"success": False, "cleaned_count": 0, "failed_count": 0,
                    "remaining_candidates": 0, "details": [plan["summary"]]}

        candidates = list(plan.get("candidates", []))
        protected_slots = list(plan.get("protected_slots", []))
        safe_z = _compute_safe_z(cfg, protected_slots)

        if not candidates:
            self._publish_feedback("succeeded", "no cleanup candidates found")
            self._publish_result(True, {"cleaned": 0, "failed": 0, "skipped": len(plan.get("skipped", []))})
            return {"success": True, "cleaned_count": 0, "failed_count": 0,
                    "remaining_candidates": 0, "details": ["no candidates"]}

        self._publish_feedback(
            "started",
            f"cleanup cycle: {len(candidates)} candidates, safe_z={safe_z:.3f}",
        )

        max_iterations = len(candidates) + 2  # 安全上限
        iteration = 0

        while candidates and iteration < max_iterations:
            iteration += 1

            # 选最优候选（已排序：远离保护区 > 高 z > 稳定）
            candidate = candidates[0]

            class_name = str(candidate.get("class_name", "unknown"))
            grasp_pos = candidate.get("base_position", [0, 0, 0])

            self._publish_feedback(
                "running",
                f"iteration {iteration}: cleaning {class_name} at "
                f"{[round(float(v), 4) for v in grasp_pos[:3]]}",
                iteration=iteration,
                remaining=len(candidates),
            )

            ok = self._execute_single_cleanup(candidate, grasp_quat, place_quat, safe_z)

            if ok:
                cleaned_count += 1
                details.append({
                    "iteration": iteration,
                    "class_name": class_name,
                    "grasp_position": [round(float(v), 4) for v in grasp_pos[:3]],
                    "return_position": [
                        round(float(v), 4)
                        for v in candidate.get("assigned_return_slot", {}).get("position", [0, 0, 0])[:3]
                    ],
                    "status": "cleaned",
                })
                rospy.logwarn("[CLEANUP_EXEC] iteration %d: cleaned %s", iteration, class_name)
            else:
                failed_count += 1
                details.append({
                    "iteration": iteration,
                    "class_name": class_name,
                    "grasp_position": [round(float(v), 4) for v in grasp_pos[:3]],
                    "status": "failed",
                })
                rospy.logerr("[CLEANUP_EXEC] iteration %d: FAILED to clean %s", iteration, class_name)

            # 回到清理观测位姿
            try:
                cleanup_observe_pose()
                time.sleep(1.0)
            except Exception as exc:
                rospy.logwarn("[CLEANUP_EXEC] failed to return to cleanup_observe_pose: %s", str(exc))

            # 等待场景稳定
            time.sleep(float(cfg.get("scan_seconds", 2.5)) * 0.5)

            # 重新扫描与规划
            plan = cleanup_scan_and_plan(
                plan_file=plan_file,
                config_file=config_file,
                resume_step=resume_step,
                failed_steps=failed_steps,
                execute_cleanup=True,
            )

            if plan.get("summary", "").startswith("SAFETY_ABORT"):
                self._publish_feedback("failed", plan["summary"])
                break

            # 结果监督：验证保护结构是否完整
            structure_ok, structure_msg = self.supervisor.verify_protected_structure(
                protected_slots=plan.get("protected_slots", []),
                classified_objects=plan.get("classified_objects", []),
                cfg=cfg,
            )
            if not structure_ok:
                self._publish_feedback("failed", f"protected structure compromised: {structure_msg}")
                details.append({"status": "structure_compromised", "message": structure_msg})
                break

            candidates = list(plan.get("candidates", []))

        # 回到任务起始位姿
        try:
            arm_ready_pose()
            time.sleep(1.5)
        except Exception as exc:
            rospy.logwarn("[CLEANUP_EXEC] failed to move to ready pose: %s", str(exc))

        remaining = len(candidates)
        success = (failed_count == 0) and (remaining == 0)

        summary = {
            "cleaned": cleaned_count,
            "failed": failed_count,
            "remaining": remaining,
            "iterations": iteration,
        }

        self._publish_result(success, summary)

        rospy.logwarn(
            "[CLEANUP_EXEC] cycle done: cleaned=%d failed=%d remaining=%d success=%s",
            cleaned_count, failed_count, remaining, str(success),
        )

        return {
            "success": success,
            "cleaned_count": cleaned_count,
            "failed_count": failed_count,
            "remaining_candidates": remaining,
            "details": details,
        }


# ============================================================
# 独立运行入口（调试用）
# ============================================================

def main():
    rospy.init_node("cleanup_executor", anonymous=True)

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

    executor = CleanupExecutorNode()
    result = executor.execute_cleanup_cycle(
        plan_file=plan_file,
        config_file=config_file,
        resume_step=resume_step,
        failed_steps=failed_steps,
    )
    rospy.logwarn("[CLEANUP_EXEC] final result: %s", json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
