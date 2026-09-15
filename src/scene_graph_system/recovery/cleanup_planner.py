#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cleanup_planner.py

清理规划器：支持 scan-only 与真实清理两种模式。

scan-only 模式（execute_cleanup=False）：
  1. 读取 cleanup_config.yaml；
  2. 读取 action.py 中的任务步骤；
  3. 根据 resume_step 生成 protected_slots；
  4. 在清理观测位姿下扫描 /object_pose；
  5. 将臂上相机坐标转换到机械臂 base 坐标；
  6. 判断每个检测物体是：
     - protected
     - cleanup_candidate
     - near_protected_manual_check
     - outside_build_region
  7. 只打印结果，不执行真实清理，不阻塞恢复链路。

真实清理模式（execute_cleanup=True）：
  1-4. 同上；
  5. 聚合 source_pose 按类别生成 return_slots；
  6. 根据决策失败步骤计算 cleanup_requirements（需清理的类别与数量）；
  7. 多条件筛选清理候选（7 条规则）；
  8. 按优先级排序候选（远离保护区 > 高 z > 稳定检测）；
  9. 返回结构化 CleanupTask 列表给 cleanup_executor.py 消费。

设计准则：
  - 验证端语义判断（外部相机） vs 臂上相机执行定位 严格分离；
  - 保护区优先，宁可少清理不误清理；
  - 同类别互换但 return_slot 不重复使用；
  - source_pose 必须是 base 坐标。
"""

from __future__ import annotations

from scene_graph_system.resources import package_resource_path, runtime_data_path

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

import rospy
import yaml

from blockkit.msg import ObjectInfo
from rm_msgs.msg import Arm_Current_State

from scene_graph_system.robot.catch import convert
from scene_graph_system.planning.task_plan_adapter import build_internal_plan_from_file


DEFAULT_PLAN_FILE = package_resource_path('scripts/action.py')
DEFAULT_CONFIG_FILE = package_resource_path('config/cleanup_config.yaml')


# ============================================================
# 基础工具
# ============================================================

def distance(a, b):
    return math.sqrt(
        sum((float(a[i]) - float(b[i])) ** 2 for i in range(3))
    )


def load_cleanup_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"cleanup config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    return cfg


def inside_build_region(p, region: dict) -> bool:
    x, y, z = [float(v) for v in p[:3]]

    return (
        float(region["x_min"]) <= x <= float(region["x_max"])
        and float(region["y_min"]) <= y <= float(region["y_max"])
        and float(region["z_min"]) <= z <= float(region["z_max"])
    )


def inside_slot(p, slot: dict) -> bool:
    x, y, z = [float(v) for v in p[:3]]
    cx, cy, cz = [float(v) for v in slot["center"][:3]]
    mx, my, mz = [float(v) for v in slot["margin"][:3]]

    return (
        abs(x - cx) <= mx
        and abs(y - cy) <= my
        and abs(z - cz) <= mz
    )


def distance_to_slot_box(p, slot: dict) -> float:
    """
    点到保护盒外表面的距离。
    如果点在盒内，距离为 0。
    """
    x, y, z = [float(v) for v in p[:3]]
    cx, cy, cz = [float(v) for v in slot["center"][:3]]
    mx, my, mz = [float(v) for v in slot["margin"][:3]]

    dx = max(abs(x - cx) - mx, 0.0)
    dy = max(abs(y - cy) - my, 0.0)
    dz = max(abs(z - cz) - mz, 0.0)

    return math.sqrt(dx * dx + dy * dy + dz * dz)


def too_close_to_any_slot(p, protected_slots: list, near_margin: float) -> bool:
    for slot in protected_slots:
        if distance_to_slot_box(p, slot) < float(near_margin):
            return True
    return False


def _candidate_overlaps_protected_slot(obj: dict, protected_slots: list) -> bool:
    """
    检查候选物体是否在空间上与任一保护槽位重叠。

    返回 True 表示候选物体的 base_position 落在某个 protected_slot 的
    inside_slot 范围内，即该位置已被已完成结构占用。
    """
    p = obj.get("base_position")
    if p is None:
        return False
    for slot in protected_slots:
        if inside_slot(p, slot):
            return True
    return False


# ============================================================
# action.py 读取：不依赖 task_plan_adapter
# ============================================================

def load_action_steps(plan_file: str) -> list:
    """
    通过 task_plan_adapter 读取 action.py 中的任务步骤，
    返回 cleanup_planner 所需的归一化格式。
    """
    plan = build_internal_plan_from_file(plan_file)
    steps = []
    for step in plan.get("execution_steps", []):
        target_place = step.get("target_place")
        if target_place is None or len(target_place) < 3:
            continue
        steps.append({
            "step": int(step.get("step", 0)),
            "target_object_id": step.get("target_object_id"),
            "class_name": step.get("target_class") or step.get("source_class"),
            "target_place": [float(target_place[0]), float(target_place[1]), float(target_place[2])],
            "expected_support_object_id": step.get("expected_support_object_id"),
            "source_pose": step.get("source_pose"),
            "source_return_release_pose": step.get("source_return_release_pose"),
            "source_return_approach_pose": step.get("source_return_approach_pose"),
            "source_return_lift_pose": step.get("source_return_lift_pose"),
            "comment": step.get("comment", ""),
        })
    return sorted(steps, key=lambda x: int(x["step"]))


# ============================================================
# protected slots
# ============================================================

def build_protected_slots(plan_file: str, resume_step: int, cfg: dict) -> list:
    steps = load_action_steps(plan_file)

    margin_cfg = cfg.get("protected_slot_margin", {}) or {}
    default_margin = margin_cfg.get("default", [0.055, 0.055, 0.050])

    protected_slots = []

    for step in steps:
        step_num = int(step.get("step", 0))

        if step_num >= int(resume_step):
            continue

        class_name = str(step.get("class_name") or "")
        target_place = step.get("target_place")

        if target_place is None or len(target_place) < 3:
            continue

        margin = margin_cfg.get(class_name, default_margin)

        protected_slots.append({
            "step": step_num,
            "target_object_id": step.get("target_object_id"),
            "class_name": class_name,
            "center": [
                float(target_place[0]),
                float(target_place[1]),
                float(target_place[2]),
            ],
            "margin": [
                float(margin[0]),
                float(margin[1]),
                float(margin[2]),
            ],
        })

    return protected_slots


# ============================================================
# wrist camera scan
# ============================================================

# ============================================================
# wrist camera scan
# ============================================================

class CleanupScanner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.objects: List[dict] = []
        self.merge_dist = float(cfg.get("merge_distance", 0.025))

        # 清理扫描时机械臂应处于 cleanup_observe_pose，理论上静止。
        # 因此只在 scan 开始前读取一次机械臂位姿，避免每个 /object_pose 回调里阻塞等待。
        self.arm_pose = None

    def _load_arm_pose_once(self):
        """
        扫描开始前只读取一次机械臂当前位姿。

        原实现是在每个 /object_pose 回调中 wait_for_message 一次 Arm_Current_State。
        当 /object_pose 一帧连续发布多个物体时，会造成回调阻塞和消息积压，
        导致 cleanup_planner 只能处理少量检测结果。
        """
        arm_pose_msg = rospy.wait_for_message(
            "/rm_driver/Arm_Current_State",
            Arm_Current_State,
            timeout=2.0,
        )

        self.arm_pose = [
            float(arm_pose_msg.Pose[0]),
            float(arm_pose_msg.Pose[1]),
            float(arm_pose_msg.Pose[2]),
            float(arm_pose_msg.Pose[3]),
            float(arm_pose_msg.Pose[4]),
            float(arm_pose_msg.Pose[5]),
        ]

        rospy.logwarn(
            "[CLEANUP] cached arm pose for scan: %s",
            str([round(float(v), 4) for v in self.arm_pose]),
        )

    def _object_pose_to_base(self, msg: ObjectInfo) -> list:
        """
        当前 /object_pose 已确定是臂上相机坐标，因此复用 catch.convert 转 base。

        注意：
        - 不在这里 wait_for_message；
        - 使用 scan() 开始前缓存的 self.arm_pose；
        - 清理扫描期间机械臂必须保持静止。
        """
        if self.arm_pose is None:
            raise RuntimeError(
                "arm_pose cache is empty; call _load_arm_pose_once() before scan"
            )

        result = convert(
            msg.x,
            msg.y,
            msg.z,
            self.arm_pose[0],
            self.arm_pose[1],
            self.arm_pose[2],
            self.arm_pose[3],
            self.arm_pose[4],
            self.arm_pose[5],
        )

        return [float(result[0]), float(result[1]), float(result[2])]

    def add_detection(self, msg: ObjectInfo):
        try:
            p_base = self._object_pose_to_base(msg)
        except Exception as exc:
            rospy.logwarn("[CLEANUP] failed to convert detection: %s", str(exc))
            return

        class_name = str(msg.object_class or "").strip()
        if not class_name:
            return

        for obj in self.objects:
            if obj["class_name"] != class_name:
                continue

            if distance(obj["base_position"], p_base) < self.merge_dist:
                count = int(obj["count"])
                obj["base_position"] = [
                    (obj["base_position"][i] * count + p_base[i]) / float(count + 1)
                    for i in range(3)
                ]
                obj["count"] = count + 1
                return

        self.objects.append({
            "class_name": class_name,
            "base_position": p_base,
            "count": 1,
        })

    def scan(self, seconds: float = 2.0) -> list:
        self.objects = []
        self.arm_pose = None

        # 关键修改：订阅 /object_pose 前先缓存一次机械臂位姿。
        try:
            self._load_arm_pose_once()
        except Exception as exc:
            rospy.logerr("[CLEANUP] failed to cache arm pose before scan: %s", str(exc))
            return []

        rospy.logwarn("[CLEANUP] scanning /object_pose for %.2f seconds", float(seconds))

        # queue_size 加大，避免一帧多个物体时消息堆积/丢失。
        sub = rospy.Subscriber(
            "/object_pose",
            ObjectInfo,
            self.add_detection,
            queue_size=100,
        )

        rospy.sleep(float(seconds))
        sub.unregister()

        # 给已经进入回调队列的最后几条消息一点处理时间。
        rospy.sleep(0.1)

        min_count = int(self.cfg.get("min_detection_count", 2))

        stable_objects = [
            obj for obj in self.objects
            if int(obj.get("count", 0)) >= min_count
        ]

        rospy.logwarn(
            "[CLEANUP] raw clusters=%d stable_clusters=%d min_count=%d",
            len(self.objects),
            len(stable_objects),
            min_count,
        )

        return stable_objects


# ============================================================
# classify
# ============================================================

def classify_objects(objects: list, protected_slots: list, cfg: dict) -> list:
    region = cfg.get("build_region", {})
    near_margin = float(cfg.get("near_protected_margin", 0.030))

    results = []

    for obj in objects:
        p = obj["base_position"]

        if not inside_build_region(p, region):
            status = "outside_build_region"
        else:
            status = "cleanup_candidate"

            for slot in protected_slots:
                if inside_slot(p, slot):
                    status = "protected"
                    break

            if status == "cleanup_candidate":
                if too_close_to_any_slot(p, protected_slots, near_margin):
                    status = "near_protected_manual_check"

        item = dict(obj)
        item["cleanup_status"] = status
        results.append(item)

    return results


def cleanup_scan_only(plan_file: str, config_file: str, resume_step: int) -> dict:
    """
    scan-only 模式：仅扫描分类，不执行真实清理。

    当 execute_cleanup=false 时，此函数必须快速返回 True，
    不阻塞后续 resume_from_step。
    """
    cfg = load_cleanup_config(config_file)

    protected_slots = build_protected_slots(
        plan_file=plan_file,
        resume_step=int(resume_step),
        cfg=cfg,
    )

    rospy.logwarn("[CLEANUP] protected_steps < %s", str(resume_step))

    if not protected_slots:
        rospy.logwarn("[CLEANUP] no protected slots found")

    for slot in protected_slots:
        rospy.logwarn(
            "[CLEANUP] protect step=%s id=%s class=%s center=%s margin=%s",
            str(slot["step"]),
            str(slot.get("target_object_id")),
            str(slot.get("class_name")),
            str([round(float(v), 4) for v in slot["center"]]),
            str([round(float(v), 4) for v in slot["margin"]]),
        )

    scanner = CleanupScanner(cfg)
    objects = scanner.scan(seconds=float(cfg.get("scan_seconds", 2.5)))

    classified = classify_objects(objects, protected_slots, cfg)

    rospy.logwarn("[CLEANUP] scan result:")
    for obj in classified:
        rospy.logwarn(
            "[CLEANUP] class=%s p=%s count=%s status=%s",
            str(obj.get("class_name")),
            str([round(float(v), 4) for v in obj.get("base_position", [0, 0, 0])]),
            str(obj.get("count")),
            str(obj.get("cleanup_status")),
        )

    candidates = [
        obj for obj in classified
        if obj.get("cleanup_status") == "cleanup_candidate"
    ]

    rospy.logwarn("[CLEANUP] cleanup_candidates=%d", len(candidates))

    # 安全检查：空间重叠检测 + 同类警告
    for obj in classified:
        if obj.get("cleanup_status") != "cleanup_candidate":
            continue

        detected_class = str(obj.get("class_name", ""))
        p_str = str([round(float(v), 4) for v in obj.get("base_position", [0, 0, 0])])

        # 同类警告（非阻塞）：候选物体类别与保护区某步骤类别相同
        for slot in protected_slots:
            slot_class = str(slot.get("class_name", ""))
            if detected_class == slot_class:
                rospy.logwarn(
                    "[CLEANUP] same-class warning: detected class=%s at p=%s "
                    "is cleanup_candidate but matches protected class=%s. "
                    "Consider enlarging protected_slot_margin or adjusting build_region. "
                    "Real cleanup is DISABLED in scan-only mode.",
                    detected_class,
                    p_str,
                    slot_class,
                )
                break

        # 空间重叠检测：候选物体位置与保护槽位空间重叠
        if _candidate_overlaps_protected_slot(obj, protected_slots):
            rospy.logerr(
                "[CLEANUP] SPATIAL OVERLAP DETECTED: cleanup_candidate class=%s at p=%s "
                "spatially overlaps a protected slot. "
                "Real cleanup is DISABLED in scan-only mode, but this would ABORT real cleanup.",
                detected_class,
                p_str,
            )

    return {
        "protected_slots": protected_slots,
        "objects": classified,
        "cleanup_candidates": candidates,
    }


# ============================================================
# return_slot 管理：同类别互换，不重复使用
# ============================================================

def _load_runtime_return_slots_cache() -> Dict[str, List[dict]]:
    """
    读取执行端保存的 runtime return slot cache。

    文件路径: ROS_HOME/scene_graph_system/runtime_return_slots.json，支持 SCENE_GRAPH_RUNTIME_DIR 覆盖。

    返回 {class_name: [slot_dict, ...]}，读取失败返回 {}。
    """
    cache_path = runtime_data_path("runtime_return_slots.json")

    if not os.path.isfile(cache_path):
        rospy.logwarn("[CLEANUP] no runtime return_slots cache at %s", cache_path)
        return {}

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        rospy.logwarn("[CLEANUP] failed to read runtime return_slots cache: %s", str(exc))
        return {}

    grouped: Dict[str, List[dict]] = {}
    for step_key, entry in raw.items():
        class_name = str(entry.get("class_name", "")).strip()
        if not class_name:
            continue

        release_pose = entry.get("source_return_release_pose")
        if release_pose is None or len(release_pose) < 2:
            continue

        pos = [
            float(release_pose[0]),
            float(release_pose[1]),
            float(release_pose[2]) if len(release_pose) >= 3 else 0.0,
        ]

        slot = {
            "class_name": class_name,
            "position": pos,
            "source_step": int(entry.get("step", step_key)),
            "occupied": False,
            "pose_source": "runtime_cache",
        }

        if class_name not in grouped:
            grouped[class_name] = []
        grouped[class_name].append(slot)

    if grouped:
        rospy.logwarn(
            "[CLEANUP] loaded %d return_slot classes from runtime cache",
            len(grouped),
        )

    return grouped


def aggregate_source_poses_by_class(plan_file: str) -> Dict[str, List[dict]]:
    """
    从 action.py 提取每个步骤的回收位置，按 class_name 聚合为 return_slots。

    优先使用 source_return_release_pose：
      - 它是执行端真实抓取前记录的安全释放位姿；
      - x/y 比 source_pose 更接近"物体原放置位置"。

    若离线 plan 中没有 source_return_release_pose：
      1. fallback 到 source_pose；
      2. 若 plan 中也没有 source_pose，读取 runtime return_slots cache。

    z 高度不再作为清理放置的决定值——放置阶段的 z 由 cleanup_return_policy 控制。
    """
    steps = load_action_steps(plan_file)

    raw_slots: Dict[str, List[dict]] = {}
    plan_has_any_pose = False

    for step in steps:
        class_name = str(step.get("class_name") or "").strip()
        if not class_name:
            continue

        step_id = int(step.get("step", 0) or 0)

        release_pose = step.get("source_return_release_pose")
        source_pose = step.get("source_pose")

        pose_source = "source_return_release_pose"

        if release_pose is not None and len(release_pose) >= 2:
            plan_has_any_pose = True
            pos = [
                float(release_pose[0]),
                float(release_pose[1]),
                float(release_pose[2]) if len(release_pose) >= 3 else 0.0,
            ]
        elif source_pose is not None and len(source_pose) >= 2:
            plan_has_any_pose = True
            pose_source = "source_pose_fallback"
            pos = [
                float(source_pose[0]),
                float(source_pose[1]),
                float(source_pose[2]) if len(source_pose) >= 3 else 0.0,
            ]
        else:
            continue

        if class_name not in raw_slots:
            raw_slots[class_name] = []
        raw_slots[class_name].append({
            "pos": pos,
            "step": step_id,
            "pose_source": pose_source,
        })

        rospy.logwarn(
            "[CLEANUP] return_slot source: step=%s class=%s source=%s pos=%s",
            str(step_id),
            class_name,
            pose_source,
            str([round(v, 4) for v in pos]),
        )

    # 如果 plan 中没有任何 step 提供有效位姿，尝试 runtime cache
    if not plan_has_any_pose or not raw_slots:
        cache = _load_runtime_return_slots_cache()
        if cache:
            rospy.logwarn(
                "[CLEANUP] return_slots by class (from runtime cache): %s",
                json.dumps(
                    {
                        cls: [
                            {
                                "pos": [round(float(v), 4) for v in s.get("position", [])[:3]],
                                "source": s.get("pose_source", ""),
                                "step": s.get("source_step", 0),
                            }
                            for s in slots
                        ]
                        for cls, slots in cache.items()
                    },
                    ensure_ascii=False,
                ),
            )
            return cache

    # 去重：距离 < 0.02m 视为同一位置
    merge_dist = 0.02
    return_slots: Dict[str, List[dict]] = {}

    for class_name, slots in raw_slots.items():
        merged: List[List[float]] = []
        for slot in slots:
            pos = slot["pos"]
            is_dup = False
            for existing in merged:
                if distance(pos, existing) < merge_dist:
                    is_dup = True
                    break
            if not is_dup:
                merged.append(pos)

        return_slots[class_name] = [
            {
                "class_name": class_name,
                "position": list(p),
                "source_step": -1,
                "occupied": False,
            }
            for p in merged
        ]

    rospy.logwarn(
        "[CLEANUP] return_slots by class: %s",
        json.dumps(
            {
                cls: [
                    {
                        "pos": [round(float(v), 4) for v in s["position"][:3]],
                        "occupied": s["occupied"],
                    }
                    for s in slots
                ]
                for cls, slots in return_slots.items()
            },
            ensure_ascii=False,
        ),
    )

    return return_slots


# ============================================================
# cleanup_requirements：计算需要清理的类别与数量
# ============================================================

def build_cleanup_requirements(
    plan_file: str,
    resume_step: int,
    failed_steps: Optional[List[int]] = None,
) -> Dict[str, int]:
    """
    根据恢复决策计算需要清理的积木类别及数量。

    逻辑：
      - 遍历 action.py 中 step >= resume_step 的步骤；
      - 提取这些步骤的目标物体类别；
      - 如果提供了 failed_steps，则只清理失败步骤涉及的类别；
      - 返回 {class_name: required_count}。

    若 failed_steps 为 None，则清理所有 resume_step 及之后涉及的类别
    （即：把所有待执行步骤的物体从搭建区清理回原位）。
    """
    steps = load_action_steps(plan_file)
    failed_set = set(failed_steps or [])

    requirements: Dict[str, int] = {}

    for step in steps:
        step_num = int(step.get("step", 0))
        if step_num < int(resume_step):
            continue

        if failed_steps is not None and step_num not in failed_set:
            continue

        class_name = str(step.get("class_name") or "").strip()
        if not class_name:
            continue

        requirements[class_name] = requirements.get(class_name, 0) + 1

    rospy.logwarn(
        "[CLEANUP] cleanup_requirements: resume_step=%s failed_steps=%s → %s",
        str(resume_step),
        str(failed_steps),
        json.dumps(requirements, ensure_ascii=False),
    )

    return requirements


# ============================================================
# 清理候选筛选：7 条规则
# ============================================================

def select_cleanup_candidates(
    classified_objects: list,
    cleanup_requirements: Dict[str, int],
    return_slots: Dict[str, List[dict]],
    protected_slots: list,
    cfg: dict,
) -> Dict[str, Any]:
    """
    从分类结果中按 7 条规则筛选清理候选。

    规则：
      1. 在 build_region 内；
      2. 不在 protected_slot 内；
      3. 不靠近 protected_slot（near_protected_margin）；
      4. 类别在 cleanup_requirements 中；
      5. 该类别未超过需要清理的数量；
      6. 有同类别可用（未占用）return_slot；
      7. 检测稳定（count >= min_detection_count）。

    不满足条件的物体标记 skip_reason，不参与清理。

    返回:
      {
        "candidates": [CleanupCandidate-like dict, ...],
        "skipped": [dict, ...],
        "occupied_return_slots": set of return_slot indices,
      }
    """
    near_margin = float(cfg.get("near_protected_margin", 0.030))
    min_detection = int(cfg.get("min_detection_count", 2))
    region = cfg.get("build_region", {})

    # 跟踪已分配数量
    allocated_count: Dict[str, int] = {}
    # 跟踪已占用的 return_slot（用 (class_name, position_index) 标识）
    occupied_slots: set = set()

    candidates = []
    skipped = []

    for obj in classified_objects:
        class_name = str(obj.get("class_name", ""))
        status = str(obj.get("cleanup_status", ""))
        p = obj.get("base_position", [0, 0, 0])
        count = int(obj.get("count", 0))

        # 规则 1：在 build_region 内
        if status == "outside_build_region":
            skipped.append(dict(obj, skip_reason="outside_build_region"))
            continue

        # 规则 2：不在 protected_slot 内
        if status == "protected":
            skipped.append(dict(obj, skip_reason="inside_protected_slot"))
            continue

        # 规则 3：不靠近 protected_slot
        if status == "near_protected_manual_check":
            skipped.append(dict(obj, skip_reason="near_protected_manual_check"))
            continue

        # 规则 4：类别在 cleanup_requirements 中
        required_count = cleanup_requirements.get(class_name, 0)
        if required_count <= 0:
            skipped.append(dict(obj, skip_reason="class_not_required"))
            continue

        # 规则 5：该类别未超过需要清理的数量
        already = allocated_count.get(class_name, 0)
        if already >= required_count:
            skipped.append(dict(obj, skip_reason="required_count_already_satisfied"))
            continue

        # 规则 6：有同类别可用 return_slot
        slots = return_slots.get(class_name, [])
        available_slot = None
        for idx, slot in enumerate(slots):
            slot_key = (class_name, idx)
            if slot_key not in occupied_slots and not slot.get("occupied", False):
                available_slot = slot
                occupied_slots.add(slot_key)
                break

        if available_slot is None:
            skipped.append(dict(obj, skip_reason="no_free_return_slot"))
            continue

        # 规则 7：检测稳定
        if count < min_detection:
            skipped.append(dict(obj, skip_reason=f"unstable_detection_count_{count}_lt_{min_detection}"))
            # 释放刚才占用的 slot
            for idx, slot in enumerate(slots):
                if slot is available_slot:
                    occupied_slots.discard((class_name, idx))
                    break
            continue

        allocated_count[class_name] = already + 1
        candidates.append(dict(
            obj,
            assigned_return_slot=available_slot,
            return_slot_index=next(
                (idx for idx, s in enumerate(slots) if s is available_slot), -1
            ),
        ))

    # 排序：优先级 = 远离保护区 > z 值高 > 检测次数多
    def _priority_key(item: dict) -> tuple:
        p = item.get("base_position", [0, 0, 0])
        min_dist = float("inf")
        for slot in protected_slots:
            d = distance_to_slot_box(p, slot)
            if d < min_dist:
                min_dist = d
        z = float(p[2]) if len(p) > 2 else 0.0
        cnt = int(item.get("count", 0))
        # 远离保护区（距离大优先）、高 z 优先、稳定检测优先
        return (-min_dist, -z, -cnt)

    candidates.sort(key=_priority_key)

    rospy.logwarn("[CLEANUP] selected %d candidates, skipped %d", len(candidates), len(skipped))
    for item in skipped:
        rospy.logwarn(
            "[CLEANUP] skip class=%s reason=%s",
            str(item.get("class_name")),
            str(item.get("skip_reason")),
        )

    return {
        "candidates": candidates,
        "skipped": skipped,
        "occupied_return_slots": occupied_slots,
    }


# ============================================================
# 扫描并规划：scan + classify + select，统一入口
# ============================================================


def cleanup_scan_and_plan(
    plan_file: str,
    config_file: str,
    resume_step: int,
    failed_steps: Optional[List[int]] = None,
    execute_cleanup: bool = False,
) -> Dict[str, Any]:
    """
    清理扫描与规划的统一切入点。

    参数:
      plan_file:        action.py 路径
      config_file:      cleanup_config.yaml 路径
      resume_step:      恢复断点步骤号
      failed_steps:     失败步骤列表（用于精确计算清理需求）
      execute_cleanup:  True=真实清理模式, False=scan-only

    返回:
      {
        "protected_slots": [...],
        "classified_objects": [...],
        "cleanup_requirements": {class_name: count},
        "return_slots": {class_name: [slot_dict, ...]},
        "candidates": [...],         # 通过 7 条规则筛选的清理候选
        "skipped": [...],            # 被跳过的物体及原因
        "execute_cleanup": bool,
        "summary": str,
      }
    """
    cfg = load_cleanup_config(config_file)

    # 1. 保护区
    protected_slots = build_protected_slots(plan_file, resume_step, cfg)
    rospy.logwarn("[CLEANUP] protected_steps < %s, slots=%d", str(resume_step), len(protected_slots))

    for slot in protected_slots:
        rospy.logwarn(
            "[CLEANUP] protect step=%s id=%s class=%s center=%s margin=%s",
            str(slot["step"]),
            str(slot.get("target_object_id")),
            str(slot.get("class_name")),
            str([round(float(v), 4) for v in slot["center"]]),
            str([round(float(v), 4) for v in slot["margin"]]),
        )

    # 2. 臂上相机扫描
    scanner = CleanupScanner(cfg)
    objects = scanner.scan(seconds=float(cfg.get("scan_seconds", 2.5)))

    # 3. 分类
    classified = classify_objects(objects, protected_slots, cfg)

    rospy.logwarn("[CLEANUP] scan result (%d objects):", len(classified))
    for obj in classified:
        rospy.logwarn(
            "[CLEANUP] class=%s p=%s count=%s status=%s",
            str(obj.get("class_name")),
            str([round(float(v), 4) for v in obj.get("base_position", [0, 0, 0])]),
            str(obj.get("count")),
            str(obj.get("cleanup_status")),
        )

    # 4. 安全检查：空间重叠检测 + 同类警告
    if execute_cleanup:
        for obj in classified:
            if obj.get("cleanup_status") != "cleanup_candidate":
                continue

            detected_class = str(obj.get("class_name", ""))
            p_str = str([round(float(v), 4) for v in obj.get("base_position", [0, 0, 0])])

            # 4a. 同类警告（非阻塞）：候选物体类别与保护区某步骤类别相同
            for slot in protected_slots:
                slot_class = str(slot.get("class_name", ""))
                if detected_class == slot_class:
                    rospy.logwarn(
                        "[CLEANUP] same-class warning: detected class=%s at p=%s "
                        "is cleanup_candidate but matches protected class=%s. "
                        "This is NOT a blocking condition; real cleanup will proceed.",
                        detected_class,
                        p_str,
                        slot_class,
                    )
                    break

            # 4b. 空间重叠检测（阻塞）：候选物体位置与保护槽位空间重叠 → 禁止真实清理
            if _candidate_overlaps_protected_slot(obj, protected_slots):
                rospy.logerr(
                    "[CLEANUP] FATAL: cleanup_candidate class=%s at p=%s "
                    "spatially overlaps a protected slot. "
                    "Real cleanup ABORTED. Enlarge protected_slot_margin or adjust build_region.",
                    detected_class,
                    p_str,
                )
                return {
                    "protected_slots": protected_slots,
                    "classified_objects": classified,
                    "cleanup_requirements": {},
                    "return_slots": {},
                    "candidates": [],
                    "skipped": [],
                    "execute_cleanup": False,
                    "summary": "SAFETY_ABORT: cleanup_candidate spatially overlaps protected slot",
                }

    # 5. 聚合 return_slots
    return_slots = aggregate_source_poses_by_class(plan_file)

    # 6. 计算 cleanup_requirements
    cleanup_reqs = build_cleanup_requirements(plan_file, resume_step, failed_steps)

    # 7. 筛选候选（7 条规则）
    selection = select_cleanup_candidates(
        classified_objects=classified,
        cleanup_requirements=cleanup_reqs,
        return_slots=return_slots,
        protected_slots=protected_slots,
        cfg=cfg,
    )

    candidates = selection["candidates"]
    skipped = selection["skipped"]

    summary_parts = [
        f"protected_slots={len(protected_slots)}",
        f"objects_scanned={len(classified)}",
        f"cleanup_requirements={json.dumps(cleanup_reqs, ensure_ascii=False)}",
        f"return_slots_by_class={json.dumps({c: len(s) for c, s in return_slots.items()}, ensure_ascii=False)}",
        f"candidates={len(candidates)}",
        f"skipped={len(skipped)}",
    ]
    summary = "; ".join(summary_parts)
    rospy.logwarn("[CLEANUP] plan summary: %s", summary)

    return {
        "protected_slots": protected_slots,
        "classified_objects": classified,
        "cleanup_requirements": cleanup_reqs,
        "return_slots": return_slots,
        "candidates": candidates,
        "skipped": skipped,
        "execute_cleanup": execute_cleanup,
        "summary": summary,
    }


# ============================================================
# main
# ============================================================

def main():
    rospy.init_node("cleanup_planner", anonymous=True)

    resume_step = int(rospy.get_param("~resume_step", 1))
    plan_file = str(rospy.get_param("~plan_file", DEFAULT_PLAN_FILE))
    config_file = str(rospy.get_param("~config_file", DEFAULT_CONFIG_FILE))
    execute_cleanup = bool(rospy.get_param("~execute_cleanup", False))
    failed_steps_json = str(rospy.get_param("~failed_steps", "[]")).strip()

    failed_steps = None
    try:
        parsed = json.loads(failed_steps_json)
        if isinstance(parsed, list) and parsed:
            failed_steps = [int(v) for v in parsed]
    except (json.JSONDecodeError, ValueError, TypeError):
        failed_steps = None

    if execute_cleanup:
        result = cleanup_scan_and_plan(
            plan_file=plan_file,
            config_file=config_file,
            resume_step=resume_step,
            failed_steps=failed_steps,
            execute_cleanup=True,
        )
    else:
        result = cleanup_scan_only(
            plan_file=plan_file,
            config_file=config_file,
            resume_step=resume_step,
        )

    summary = {
        "resume_step": resume_step,
        "execute_cleanup": execute_cleanup,
        "protected_count": len(result.get("protected_slots", [])),
        "object_count": len(result.get("classified_objects", result.get("objects", []))),
        "cleanup_candidate_count": len(result.get("candidates", result.get("cleanup_candidates", []))),
        "skipped_count": len(result.get("skipped", [])),
    }
    rospy.logwarn("[CLEANUP] summary=%s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
