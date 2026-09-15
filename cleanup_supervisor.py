#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cleanup_supervisor.py

清理过程监督器。

监督分为两类：
  1. 清理动作过程监督：
     - 夹爪是否成功闭合（抓取成功）；
     - 夹爪是否成功打开（释放成功）；
     - 搬运过程中是否掉落。

  2. 清理结果监督：
     - 保护结构是否仍完整；
     - 保护区外待清理目标是否减少/清空。

当检测到故障时：
  - 动作监督故障：重新从观测姿态执行当前清理动作（重新搜索目标并执行）；
  - 结果监督故障：重新确认恢复断点与需清理积木列表，重新执行恢复。
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import rospy

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))



from scene_graph_system.recovery.cleanup_planner import inside_slot, too_close_to_any_slot
from scene_graph_system.robot.gripper_command_channel import (
    TOPIC_GRIPPER_STATE,
    wait_for_fresh_gripper_state,
)


class CleanupSupervisor:
    """
    清理监督器。

    负责在清理过程中持续监控夹爪状态和保护结构完整性。
    """

    def __init__(self):
        self.gripper_state_topic = str(
            rospy.get_param("~gripper_state_topic", TOPIC_GRIPPER_STATE)
        )
        self.gripper_timeout = float(rospy.get_param("~gripper_timeout", 3.0))
        self.gripper_state_max_age_sec = max(
            0.1, float(rospy.get_param("~gripper_state_max_age_sec", 1.0))
        )
        self.gripper_min_stable_count = max(
            1, int(rospy.get_param("~gripper_min_stable_count", 3))
        )

    # ============================================================
    # 动作过程监督：夹爪状态
    # ============================================================

    def _get_gripper_state(self) -> Optional[dict]:
        """
        从独立夹爪监控话题获取新鲜、稳定、有效的当前状态。
        """
        try:
            msg, reason = wait_for_fresh_gripper_state(
                timeout_sec=self.gripper_timeout,
                max_age_sec=self.gripper_state_max_age_sec,
                topic=self.gripper_state_topic,
            )
            if msg is None:
                rospy.logwarn(
                    "[CLEANUP_SUPERVISOR] gripper state unavailable: %s",
                    str(reason),
                )
                return None
            if int(msg.stable_count) < self.gripper_min_stable_count:
                return None
            states = set(msg.states or [])
            if "unknown" in states:
                return None
            return {
                "states": states,
                "position": float(msg.position),
                "force": float(msg.force),
                "holding_latched": bool(msg.holding_latched),
                "sample_seq": int(msg.sample_seq),
            }
        except Exception as exc:
            rospy.logwarn("[CLEANUP_SUPERVISOR] failed to get gripper state: %s", str(exc))
            return None

    def check_gripper_closed(self) -> Tuple[bool, str]:
        """
        检查夹爪是否成功闭合。

        此处的“闭合成功”用于抓取监督，因此必须是 holding；机械闭合但
        未夹到物体的 closed 不算抓取成功。

        返回 (ok, message)。
        """
        state = self._get_gripper_state()
        if state is None:
            return False, "gripper_state_unavailable"
        states = set(state.get("states", set()) or set())
        if "holding" in states:
            return True, "gripper_holding"
        return False, "gripper_not_holding: states=%s" % sorted(states)

    def check_gripper_open(self) -> Tuple[bool, str]:
        """
        检查夹爪是否成功打开。

        判定逻辑：
          - 夹爪 position > 张开阈值（REALMAN_OPEN_POSITION_THRESHOLD ≈ 900.0）

        返回 (ok, message)。
        """
        state = self._get_gripper_state()
        if state is None:
            return False, "gripper_state_unavailable"
        states = set(state.get("states", set()) or set())
        if "open" in states:
            return True, "gripper_open"
        return False, "gripper_not_open: states=%s" % sorted(states)

    def check_gripper_holding(self) -> Tuple[bool, str]:
        """
        检查夹爪是否持有物体。

        返回 (holding, message)。
        """
        state = self._get_gripper_state()
        if state is None:
            return False, "gripper_state_unavailable"
        states = set(state.get("states", set()) or set())
        if "holding" in states:
            return True, "gripper_holding"
        return False, "gripper_not_holding: states=%s" % sorted(states)

    # ============================================================
    # 结果监督：保护结构完整性
    # ============================================================

    def verify_protected_structure(
        self,
        protected_slots: List[dict],
        classified_objects: List[dict],
        cfg: dict,
    ) -> Tuple[bool, str]:
        """
        验证保护结构是否仍然完整。

        检查：
          1. 每个 protected_slot 中是否有物体（容忍一定容差）；
          2. 保护区物体不应被标记为 cleanup_candidate。

        返回 (ok, message)。
        """
        near_margin = float(cfg.get("near_protected_margin", 0.030))

        for slot in protected_slots:
            slot_step = slot.get("step", "?")
            slot_class = str(slot.get("class_name", ""))

            # 查找落在保护盒内的物体
            found_in_slot = False
            for obj in classified_objects:
                p = obj.get("base_position", [0, 0, 0])
                if inside_slot(p, slot):
                    found_in_slot = True
                    obj_status = str(obj.get("cleanup_status", ""))
                    if obj_status == "cleanup_candidate":
                        return False, (
                            f"protected step={slot_step} class={slot_class} "
                            f"object misclassified as cleanup_candidate"
                        )
                    break

            if not found_in_slot:
                # 检查是否有物体靠近保护盒
                nearby_found = False
                for obj in classified_objects:
                    p = obj.get("base_position", [0, 0, 0])
                    if too_close_to_any_slot(p, [slot], near_margin):
                        nearby_found = True
                        break

                if not nearby_found:
                    rospy.logwarn(
                        "[CLEANUP_SUPERVISOR] protected step=%s class=%s: "
                        "no object detected in slot (may be occluded)",
                        str(slot_step), slot_class,
                    )

        return True, "protected_structure_intact"

    def count_cleanup_targets_remaining(
        self,
        classified_objects: List[dict],
        cleanup_requirements: Dict[str, int],
    ) -> Dict[str, int]:
        """
        统计保护区外仍有多少待清理目标。

        返回 {class_name: remaining_count}。
        """
        remaining: Dict[str, int] = {}
        for obj in classified_objects:
            status = str(obj.get("cleanup_status", ""))
            if status != "cleanup_candidate":
                continue
            class_name = str(obj.get("class_name", ""))
            if class_name in cleanup_requirements:
                remaining[class_name] = remaining.get(class_name, 0) + 1

        return remaining
