#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pick_place_executor.py

通用 pick-place 动作流程：8 阶段抓取-放置。

提取自 catch.py 的 catch_and_place()，供 catch.py（普通执行）
和 cleanup_executor.py（清理执行）共用。

设计准则：
  - 所有底层动作通过 robot_motion_primitives 完成；
  - 无 catch.py 状态机依赖（暂停/中断通过回调注入）；
  - 支持 safe_z 覆盖（cleanup_executor 保护结构体）；
  - 保留 source_return_* 安全返回位姿（恢复端需要）。
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import rospy

from scene_graph_system.robot.robot_motion_primitives import (
    arm_ready_pose,
    get_default_grasp_quat,
    get_place_quaternion,
    gripper_close,
    gripper_open,
    movel_type,
    movejp_type,
    normalize_half_turn_rad,
    rotate_tool_quaternion_by_local_z,
    wait_for_tcp_xyz,
)

# 与 catch_and_place 一致
MAX_GRASP_ROT_RAD = np.deg2rad(90.0)
GRASP_YAW_BIAS_RAD = 0.0


def execute_pick_place_by_base_pose(
    source_base_position,
    target_base_position,
    grasp_quat=None,
    place_quat=None,
    object_class="object",
    obj_angle=0.0,
    mode="cleanup",
    stage_prefix="",
    # 中断 / 等待回调
    check_continue_fn=None,
    sleep_fn=None,
    # 阶段回调
    on_stage_start=None,
    on_stage_finish=None,
    # source_return_* 输出
    source_return_pose_out=None,
    # 失败原因输出
    failure_reason_out=None,
    # 安全高度覆盖（None = 使用默认偏移量计算）
    approach_z=None,
    grasp_z=None,
    lift_z=None,
    target_above_z=None,
    place_z=None,
    leave_z=None,
    # 监督 hook
    after_close_gripper_fn=None,
    after_lift_fn=None,
    after_open_gripper_fn=None,
    # 默认动作参数（对齐 catch_and_place）
    approach_x_offset=-0.055,
    approach_y_offset=0.0,
    approach_z_offset=0.07,
    grasp_z_offset=0.03,
    lift_z_offset=0.10,
    place_above_offset=0.10,
    place_release_offset=0.018,
    leave_place_offset=0.04,
    move_speed=0.20,
    descend_speed=0.15,
    post_wait=1.0,

    # 抓取阶段专用等待
    settle_before_close=1.0,
    gripper_close_wait=2.5,
    settle_after_lift=0.3,

    return_to_ready=False,
    # TCP 到位等待
    wait_motion_done=True,
    motion_tolerance=0.015,
    motion_timeout=25.0,
    # 测试模式
    test_phase="full",
):
    """
    通用 pick-place：从 source_base_position 抓取物体，放到 target_base_position。

    阶段顺序（与 catch_and_place 一致）：
      1. approach_source      — MoveJ_P 到抓取点上方
      2. descend_to_grasp     — MoveL 下降到抓取高度
      3. close_gripper        — 闭合夹爪
      4. lift_object          — MoveL 竖直上抬
      5. move_to_target_above — MoveJ_P 到放置点上方
      6. descend_to_place     — MoveL 下降到放置高度
      7. open_gripper         — 打开夹爪
      8. leave_target         — MoveL 竖直上抬离开

    参数：
      grasp_quat: 抓取姿态四元数 [x,y,z,w]。为 None 时自动计算
                  （调用 robot_motion_primitives 默认值 + obj_angle 旋转）。
      place_quat: 放置姿态四元数 [x,y,z,w]。为 None 时使用 get_place_quaternion()。
      check_continue_fn: 可选，返回 False 中断执行。catch.py 传入 wait_if_paused_or_aborted。
      sleep_fn: 可选的可中断 sleep，签名 fn(seconds) -> bool。
      source_return_pose_out: 可选 dict，写入 source_return_approach_pose / _release_pose / _lift_pose。
    """
    # ---- 解析源 / 目标 ----
    src_x = float(source_base_position[0])
    src_y = float(source_base_position[1])
    src_z = float(source_base_position[2])

    tgt_x = float(target_base_position[0])
    tgt_y = float(target_base_position[1])
    tgt_z = float(target_base_position[2])

    # ---- 姿态 ----
    if grasp_quat is None:
        grasp_quat = _compute_grasp_quat(float(obj_angle))
    else:
        grasp_quat = [float(v) for v in grasp_quat[:4]]

    if place_quat is None:
        place_quat = get_place_quaternion()
    else:
        place_quat = [float(v) for v in place_quat[:4]]

    # ---- 计算各阶段 z（覆盖值 > 默认偏移） ----
    _approach_z = float(approach_z) if approach_z is not None else src_z + float(approach_z_offset)
    _grasp_z = float(grasp_z) if grasp_z is not None else src_z + float(grasp_z_offset)
    _lift_z = float(lift_z) if lift_z is not None else src_z + float(lift_z_offset)
    _target_above_z = float(target_above_z) if target_above_z is not None else tgt_z + float(place_above_offset)
    _place_z = float(place_z) if place_z is not None else tgt_z + float(place_release_offset)
    _leave_z = float(leave_z) if leave_z is not None else tgt_z + float(leave_place_offset)

    # ---- 记录安全返回位姿（恢复端 return_held_object_to_source 需要） ----
    _grasp_src_y = src_y + float(approach_y_offset)
    if source_return_pose_out is not None:
        source_return_pose_out["source_return_approach_pose"] = [
            src_x + approach_x_offset, _grasp_src_y, _approach_z,
            grasp_quat[0], grasp_quat[1], grasp_quat[2], grasp_quat[3],
        ]
        source_return_pose_out["source_return_release_pose"] = [
            src_x + approach_x_offset, _grasp_src_y, _grasp_z,
            grasp_quat[0], grasp_quat[1], grasp_quat[2], grasp_quat[3],
        ]
        source_return_pose_out["source_return_lift_pose"] = [
            src_x + approach_x_offset, _grasp_src_y, _lift_z,
            grasp_quat[0], grasp_quat[1], grasp_quat[2], grasp_quat[3],
        ]

    rospy.logwarn(
        "[PICK_PLACE] %s start: class=%s src=(%.4f,%.4f,%.4f) tgt=(%.4f,%.4f,%.4f) "
        "approach_z=%.4f grasp_z=%.4f lift_z=%.4f target_above_z=%.4f place_z=%.4f leave_z=%.4f",
        str(stage_prefix),
        str(object_class),
        src_x, src_y, src_z,
        tgt_x, tgt_y, tgt_z,
        _approach_z, _grasp_z, _lift_z, _target_above_z, _place_z, _leave_z,
    )

    ok = _run_stage_sequence(
        src_x=src_x,
        src_y=src_y,
        tgt_x=tgt_x,
        tgt_y=tgt_y,
        approach_x_offset=float(approach_x_offset),
        approach_y_offset=float(approach_y_offset),
        approach_z=_approach_z,
        grasp_z=_grasp_z,
        lift_z=_lift_z,
        target_above_z=_target_above_z,
        place_z=_place_z,
        leave_z=_leave_z,
        grasp_quat=grasp_quat,
        place_quat=place_quat,
        move_speed=float(move_speed),
        descend_speed=float(descend_speed),
        post_wait=float(post_wait),

        settle_before_close=float(settle_before_close),
        gripper_close_wait=float(gripper_close_wait),
        settle_after_lift=float(settle_after_lift),

        check_continue_fn=check_continue_fn,
        sleep_fn=sleep_fn,
        on_stage_start=on_stage_start,
        on_stage_finish=on_stage_finish,
        after_close_gripper_fn=after_close_gripper_fn,
        after_lift_fn=after_lift_fn,
        after_open_gripper_fn=after_open_gripper_fn,
        stage_prefix=str(stage_prefix),
        return_to_ready=bool(return_to_ready),
        wait_motion_done=bool(wait_motion_done),
        motion_tolerance=float(motion_tolerance),
        motion_timeout=float(motion_timeout),
        test_phase=str(test_phase),
        failure_reason_out=failure_reason_out,
    )

    return ok


def _compute_grasp_quat(obj_angle):
    """计算抓取四元数（不移动机械臂）。"""
    grasp_rotation = normalize_half_turn_rad(float(obj_angle))
    grasp_rotation = float(np.clip(grasp_rotation, -MAX_GRASP_ROT_RAD, MAX_GRASP_ROT_RAD))
    grasp_rotation += GRASP_YAW_BIAS_RAD

    base_quat = get_default_grasp_quat()
    return rotate_tool_quaternion_by_local_z(base_quat, grasp_rotation).tolist()


def _do_sleep(seconds, sleep_fn):
    """执行等待：优先使用可中断 sleep_fn，否则 rospy.sleep。"""
    if sleep_fn is not None:
        return bool(sleep_fn(float(seconds)))
    else:
        rospy.sleep(float(seconds))
        return True


def _call_check(check_continue_fn, stage_name, phase):
    """调用 check_continue_fn(stage_name, phase)，兼容旧无参签名。"""
    if check_continue_fn is None:
        return True
    try:
        return bool(check_continue_fn(stage_name, phase))
    except TypeError:
        return bool(check_continue_fn())


def _make_run_check(check_continue_fn, stage_name):
    """创建运动等待中使用的无参 check，内部以 phase=\"run\" 调用。"""
    if check_continue_fn is None:
        return None

    def _runner():
        try:
            return bool(check_continue_fn(stage_name, "run"))
        except TypeError:
            return bool(check_continue_fn())

    return _runner


def _pose_7d(x, y, z, quat):
    """拼接 7D 位姿 [x, y, z, qx, qy, qz, qw]。"""
    return [float(x), float(y), float(z), float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]


def _wait_motion_or_sleep(target_xyz, wait_motion_done, motion_tolerance,
                        motion_timeout, check_continue_fn, post_wait, sleep_fn):
    """等待 TCP 到位（优先）或固定时长 fallback。"""
    if wait_motion_done:
        return wait_for_tcp_xyz(
            target_xyz,
            tolerance=motion_tolerance,
            timeout=motion_timeout,
            check_continue_fn=check_continue_fn,
        )
    else:
        return _do_sleep(post_wait, sleep_fn)


def _run_stage_sequence(
    src_x, src_y, tgt_x, tgt_y,
    approach_x_offset, approach_y_offset, approach_z, grasp_z, lift_z,
    target_above_z, place_z, leave_z,
    grasp_quat, place_quat,
    move_speed, descend_speed, post_wait,
    settle_before_close, gripper_close_wait, settle_after_lift,
    check_continue_fn, sleep_fn,
    on_stage_start, on_stage_finish,
    after_close_gripper_fn, after_lift_fn, after_open_gripper_fn,
    stage_prefix, return_to_ready,
    wait_motion_done, motion_tolerance, motion_timeout,
    test_phase="full",
    failure_reason_out=None,
):
    """执行完整的 8 阶段序列。"""

    def _fail(reason):
        if failure_reason_out is not None:
            failure_reason_out["reason"] = str(reason)
        return False

    grasp_x = src_x + approach_x_offset
    grasp_y = src_y + approach_y_offset

    # ---- Stage 1: approach_source ----
    if not _call_check(check_continue_fn, "approach_source", "pre"):
        return _fail("interrupted_before_approach")
    _call_stage(on_stage_start, "approach_source")
    _approach_target = [grasp_x, grasp_y, approach_z]
    rospy.logwarn("[PICK_PLACE] %s approach_source: (%.4f, %.4f, %.4f)",
                  stage_prefix, *_approach_target)
    if not movejp_type(_pose_7d(*_approach_target, grasp_quat), move_speed):
        rospy.logerr("[PICK_PLACE] %s approach_source failed", stage_prefix)
        return _fail("motion_failed_approach")
    if not _wait_motion_or_sleep(_approach_target, wait_motion_done, motion_tolerance,
                                 motion_timeout, _make_run_check(check_continue_fn, "approach_source"), post_wait, sleep_fn):
        rospy.logerr("[PICK_PLACE] %s did not reach approach target", stage_prefix)
        return _fail("motion_timeout_approach")
    if not _call_check(check_continue_fn, "approach_source", "post"):
        return _fail("gripper_check_failed_approach_post")
    _call_stage(on_stage_finish, "approach_source")

    # ---- Stage 2: descend_to_grasp ----
    if not _call_check(check_continue_fn, "descend_to_grasp", "pre"):
        return _fail("interrupted_before_descend_grasp")
    _call_stage(on_stage_start, "descend_to_grasp")
    _grasp_target = [grasp_x, grasp_y, grasp_z]
    rospy.logwarn("[PICK_PLACE] %s descend_to_grasp: (%.4f, %.4f, %.4f)",
                  stage_prefix, *_grasp_target)
    if not movel_type(_pose_7d(*_grasp_target, grasp_quat), descend_speed):
        rospy.logerr("[PICK_PLACE] %s descend_to_grasp failed", stage_prefix)
        return _fail("motion_failed_descend_grasp")
    if not _wait_motion_or_sleep(_grasp_target, wait_motion_done, motion_tolerance,
                                 motion_timeout, _make_run_check(check_continue_fn, "descend_to_grasp"), post_wait, sleep_fn):
        rospy.logerr(
            "[PICK_PLACE] %s abort before close_gripper because grasp target was not reached",
            stage_prefix,
        )
        return _fail("motion_timeout_descend_grasp")

    rospy.logwarn(
        "[PICK_PLACE] %s settle_before_close %.2fs at grasp target",
        stage_prefix,
        float(settle_before_close),
    )
    if not _do_sleep(settle_before_close, sleep_fn):
        return _fail("interrupted_settle_before_close")

    if not _call_check(check_continue_fn, "descend_to_grasp", "post"):
        return _fail("gripper_check_failed_descend_grasp_post")

    _call_stage(on_stage_finish, "descend_to_grasp")

    if test_phase == "descend_to_grasp_no_close":
        rospy.logwarn(
            "[PICK_PLACE] %s test_phase=descend_to_grasp_no_close reached grasp target; stop before close_gripper",
            stage_prefix,
        )
        return True

    # ---- Stage 3: close_gripper ----
    if not _call_check(check_continue_fn, "close_gripper", "pre"):
        return _fail("interrupted_before_close_gripper")
    _call_stage(on_stage_start, "close_gripper")
    rospy.logwarn("[PICK_PLACE] %s close_gripper", stage_prefix)
    ret = gripper_close()
    if ret is False:
        rospy.logerr("[PICK_PLACE] %s gripper_close returned False", stage_prefix)
        return _fail("gripper_close_failed")
    rospy.logwarn(
        "[PICK_PLACE] %s waiting gripper_close_wait %.2fs before lift",
        stage_prefix,
        float(gripper_close_wait),
    )
    if not _do_sleep(gripper_close_wait, sleep_fn):
        return _fail("interrupted_gripper_close_wait")
    if not _call_check(check_continue_fn, "close_gripper", "post"):
        return _fail("gripper_check_failed_close_post")
    if not _call_check_hook(after_close_gripper_fn, "after_close_gripper", stage_prefix):
        return _fail("gripper_holding_failed_after_close")
    _call_stage(on_stage_finish, "close_gripper")

    # ---- Stage 4: lift_object ----
    # 竖直从抓取点 grasp_x 上抬，避免低空斜线横移碰撞风险
    if not _call_check(check_continue_fn, "lift_object", "pre"):
        return _fail("interrupted_before_lift")
    _call_stage(on_stage_start, "lift_object")
    _lift_target = [grasp_x, grasp_y, lift_z]
    rospy.logwarn("[PICK_PLACE] %s lift_object: (%.4f, %.4f, %.4f)",
                  stage_prefix, *_lift_target)
    if not movel_type(_pose_7d(*_lift_target, grasp_quat), move_speed):
        rospy.logerr("[PICK_PLACE] %s lift_object failed", stage_prefix)
        return _fail("motion_failed_lift")
    if not _wait_motion_or_sleep(_lift_target, wait_motion_done, motion_tolerance,
                                 motion_timeout, _make_run_check(check_continue_fn, "lift_object"), post_wait, sleep_fn):
        rospy.logerr("[PICK_PLACE] %s did not reach lift target", stage_prefix)
        return _fail("motion_timeout_lift")

    if float(settle_after_lift) > 0:
        rospy.logwarn(
            "[PICK_PLACE] %s settle_after_lift %.2fs",
            stage_prefix,
            float(settle_after_lift),
        )
        if not _do_sleep(settle_after_lift, sleep_fn):
            return _fail("interrupted_settle_after_lift")

    if not _call_check(check_continue_fn, "lift_object", "post"):
        return _fail("gripper_check_failed_lift_post")
    if not _call_check_hook(after_lift_fn, "after_lift", stage_prefix):
        return _fail("gripper_holding_lost_during_transport")
    _call_stage(on_stage_finish, "lift_object")

    # ---- Transit: 竖直抬升到 target_above_z（当目标高度显著高于 lift 高度时） ----
    # 在源位 XY 原地做竖直 MoveJ_P，先到达目标安全高度，再水平横移。
    # 避免 MoveJ_P 同时跨越大高度差斜线 + 水平位移导致规划失败。
    _z_gap = target_above_z - lift_z
    if _z_gap > 0.05:
        if not _call_check(check_continue_fn, "transit_lift", "pre"):
            return _fail("interrupted_before_transit_lift")
        _call_stage(on_stage_start, "transit_lift")
        _transit_target = [grasp_x, grasp_y, target_above_z]
        rospy.logwarn("[PICK_PLACE] %s transit_lift: z_gap=%.3f → (%.4f, %.4f, %.4f)",
                      stage_prefix, _z_gap, *_transit_target)
        if not movejp_type(_pose_7d(*_transit_target, grasp_quat), move_speed):
            rospy.logerr("[PICK_PLACE] %s transit_lift failed", stage_prefix)
            return _fail("motion_failed_transit_lift")
        if not _wait_motion_or_sleep(_transit_target, wait_motion_done, motion_tolerance,
                                     motion_timeout, _make_run_check(check_continue_fn, "transit_lift"), post_wait, sleep_fn):
            rospy.logerr("[PICK_PLACE] %s did not reach transit target", stage_prefix)
            return _fail("motion_timeout_transit_lift")
        _call_stage(on_stage_finish, "transit_lift")

    # ---- Stage 5: move_to_target_above ----
    if not _call_check(check_continue_fn, "move_to_target_above", "pre"):
        return _fail("interrupted_before_move_to_target")
    _call_stage(on_stage_start, "move_to_target_above")
    _target_above = [tgt_x, tgt_y, target_above_z]
    rospy.logwarn("[PICK_PLACE] %s move_to_target_above: (%.4f, %.4f, %.4f)",
                  stage_prefix, *_target_above)
    if not movejp_type(_pose_7d(*_target_above, place_quat), move_speed):
        rospy.logerr("[PICK_PLACE] %s move_to_target_above failed", stage_prefix)
        return _fail("motion_failed_move_to_target")
    if not _wait_motion_or_sleep(_target_above, wait_motion_done, motion_tolerance,
                                 motion_timeout, _make_run_check(check_continue_fn, "move_to_target_above"), post_wait, sleep_fn):
        rospy.logerr("[PICK_PLACE] %s did not reach target above", stage_prefix)
        return _fail("motion_timeout_move_to_target")
    if not _call_check(check_continue_fn, "move_to_target_above", "post"):
        return _fail("gripper_check_failed_move_to_target_post")
    _call_stage(on_stage_finish, "move_to_target_above")

    # ---- Stage 6: descend_to_place ----
    if not _call_check(check_continue_fn, "descend_to_place", "pre"):
        return _fail("interrupted_before_descend_place")
    _call_stage(on_stage_start, "descend_to_place")
    _place_target = [tgt_x, tgt_y, place_z]
    rospy.logwarn("[PICK_PLACE] %s descend_to_place: (%.4f, %.4f, %.4f)",
                  stage_prefix, *_place_target)
    if not movel_type(_pose_7d(*_place_target, place_quat), descend_speed):
        rospy.logerr("[PICK_PLACE] %s descend_to_place failed", stage_prefix)
        return _fail("motion_failed_descend_place")
    if not _wait_motion_or_sleep(_place_target, wait_motion_done, motion_tolerance,
                                 motion_timeout, _make_run_check(check_continue_fn, "descend_to_place"), max(post_wait, 5.0), sleep_fn):
        rospy.logerr("[PICK_PLACE] %s did not reach place target", stage_prefix)
        return _fail("motion_timeout_descend_place")
    if not _call_check(check_continue_fn, "descend_to_place", "post"):
        return _fail("gripper_check_failed_descend_place_post")
    _call_stage(on_stage_finish, "descend_to_place")

    # ---- Stage 7: open_gripper ----
    if not _call_check(check_continue_fn, "open_gripper", "pre"):
        return _fail("interrupted_before_open_gripper")
    _call_stage(on_stage_start, "open_gripper")
    rospy.logwarn("[PICK_PLACE] %s open_gripper", stage_prefix)
    gripper_open()
    if not _do_sleep(post_wait, sleep_fn):
        return _fail("interrupted_open_gripper_wait")
    if not _call_check(check_continue_fn, "open_gripper", "post"):
        return _fail("gripper_check_failed_open_post")
    if not _call_check_hook(after_open_gripper_fn, "after_open_gripper", stage_prefix):
        return _fail("place_failed")
    _call_stage(on_stage_finish, "open_gripper")

    # ---- Stage 8: leave_target ----
    if not _call_check(check_continue_fn, "leave_target", "pre"):
        return _fail("interrupted_before_leave")
    _call_stage(on_stage_start, "leave_target")
    _leave_target = [tgt_x, tgt_y, leave_z]
    rospy.logwarn("[PICK_PLACE] %s leave_target: (%.4f, %.4f, %.4f)",
                  stage_prefix, *_leave_target)
    if not movel_type(_pose_7d(*_leave_target, place_quat), move_speed):
        rospy.logerr("[PICK_PLACE] %s leave_target failed", stage_prefix)
        return _fail("motion_failed_leave")
    if not _wait_motion_or_sleep(_leave_target, wait_motion_done, motion_tolerance,
                                 motion_timeout, _make_run_check(check_continue_fn, "leave_target"), post_wait, sleep_fn):
        rospy.logerr("[PICK_PLACE] %s did not reach leave target", stage_prefix)
        return _fail("motion_timeout_leave")
    _call_stage(on_stage_finish, "leave_target")

    # ---- Stage 9 (optional): return to ready ----
    if return_to_ready:
        if not _call_check(check_continue_fn, "return_to_ready", "pre"):
            return _fail("interrupted_before_return_ready")
        rospy.logwarn("[PICK_PLACE] %s return_to_ready", stage_prefix)
        if not arm_ready_pose():
            rospy.logerr("[PICK_PLACE] %s return_to_ready failed", stage_prefix)
            return _fail("motion_failed_return_ready")
        if not _do_sleep(4.0, sleep_fn):
            return _fail("interrupted_return_ready_wait")

    rospy.logwarn("[PICK_PLACE] %s completed successfully", stage_prefix)
    return True


def _call_check_hook(callback, name, stage_prefix):
    """调用监督 hook，正确处理 (bool, str) 返回值。返回 True 表示通过。"""
    if callback is None:
        return True

    try:
        ret = callback()
        if isinstance(ret, tuple):
            ok = bool(ret[0])
            msg = str(ret[1]) if len(ret) > 1 else ""
        else:
            ok = bool(ret)
            msg = ""

        if not ok:
            rospy.logerr("[PICK_PLACE] %s %s failed: %s", stage_prefix, name, msg)
            return False

        return True

    except Exception as exc:
        rospy.logerr("[PICK_PLACE] %s %s exception: %s", stage_prefix, name, str(exc))
        return False


def _call_stage(callback, stage_name):
    """安全调用阶段回调。"""
    if callback is not None:
        try:
            callback(str(stage_name))
        except Exception:
            pass
