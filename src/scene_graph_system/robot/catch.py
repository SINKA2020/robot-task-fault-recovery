#!/usr/bin/env python3
# -*- coding=UTF-8 -*-


from scene_graph_system.resources import package_resource_path, runtime_data_path
from std_msgs.msg import String, Bool, Empty
import json
import threading
import time
import uuid
import rospy, sys
from rm_msgs.msg import (
    MoveJ_P,
    Arm_Current_State,
    Gripper_Set,
    Gripper_Pick,
    ArmState,
    MoveL,
    MoveJ,
    set_modbus_mode,
    write_register,
    write_single_register,
    Tool_Analog_Output,
)
from geometry_msgs.msg import Pose
import numpy as np
from scipy.spatial.transform import Rotation as R
from blockkit.msg import ObjectInfo
from scene_graph_system.msg import DetectedObjects
from geometry_msgs.msg import TransformStamped, PointStamped
from geometry_msgs.msg import Point, Quaternion
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib import SimpleActionClient
from actionlib import GoalStatus
import os
import importlib.util
from types import SimpleNamespace
from typing import Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

from scene_graph_system.validation.validation_transaction import (
    TOPIC_VALIDATION_REQUEST,
    TOPIC_VALIDATION_RESULT,
    TemporalContext,
    ValidationRequest,
    ValidationResult,
    ValidationStatus,
    result_matches_request,
)
from scene_graph_system.resources import find_package_resource
from scene_graph_system.robot.gripper_command_channel import (
    TOPIC_GRIPPER_COMMAND,
    TOPIC_GRIPPER_STATE,
    publish_gripper_command,
    wait_for_gripper_monitor_sample,
    wait_for_gripper_startup_state,
)


# 相机坐标系到机械臂末端坐标系的旋转矩阵，通过手眼标定得到
rotation_matrix = np.array([
    [0, 1, 0],
    [-1, 0, 0],
    [0, 0, 1],
])

# 相机坐标系到机械臂末端坐标系的平移向量，通过手眼标定得到
translation_vector = np.array([-0.08039019, 0.03225555, -0.09756825])


# =========================
# 全局任务状态
# =========================

task_queue = []
current_task = None
is_executing = False
tasks_completed = False


# =========================
# 验证/恢复闭环状态
# =========================

execution_paused = False
execution_abort_requested = False

# 断点恢复请求。
# 当恢复系统发送 resume_from_step 时：
# - 如果当前没有动作正在执行，立即重置 current_task / task_queue；
# - 如果当前正在执行 catch_and_place，则先让当前动作退出，再重置任务队列。
execution_replan_requested = False
pending_resume_step = None

current_stage_pub = None
last_published_stage = None
status_pub = None
runtime_context_pub = None
recovery_ready_pub = None
recovery_pause_requested = False
pending_recovery_request_id = None
cleanup_result_pub = None
cleanup_active = False
validation_request_pub = None
validation_result_sub = None
validation_condition = threading.Condition()
pending_validation_request = None
pending_validation_result = None
shadow_validation_requests = {}
validation_blocked_status = ""

execution_run_id = ""
execution_attempt_id = 1
execution_stage_seq = 0
execution_actuation_seq = 0

BARRIER_MODE = "off"
VALIDATION_REQUEST_RETRY_SEC = 0.5
VALIDATION_TIMEOUT_SEC = 6.0
GRASP_BARRIER_ENABLED = True
PLACE_BARRIER_ENABLED = True
GRASP_REQUIRED_FRAMES = 1
PLACE_REQUIRED_FRAMES = 2

# 清理控制开关：默认禁用，防止旧 cleanup_pick_place 路径误触发。
# 恢复端新路径（cleanup_executor 直接调用 pick_place_executor）稳定后删除此分支。
# 通过 rosparam ~enable_cleanup_control 控制。
ENABLE_CLEANUP_CONTROL = False  # 在 main() 中通过 rospy.get_param 覆盖

DEFAULT_CURRENT_STAGE_TOPIC = "/task_execution/current_stage"
DEFAULT_STATUS_TOPIC = "/task_execution/status"
DEFAULT_CONTROL_TOPIC = "/task_execution/control"
DEFAULT_RUNTIME_CONTEXT_TOPIC = "/recovery/runtime_context"
DEFAULT_RECOVERY_READY_TOPIC = "/task_execution/recovery_ready"


# =========================
# 观测姿态与抓取姿态参数
# =========================

ARM_READY_JOINTS = [-0.047124, -0.359014, 1.487719, 0.032463, 1.704491, -0.017977]
ARM_READY_TOLERANCE = 0.02

# 清理观测位姿：用于让臂上相机观察整个搭建区域。
# 默认值来自当前已示教的清理观测关节角；如果 cleanup_config.yaml 存在，优先使用配置文件中的 cleanup_observe_joints。
CLEANUP_OBSERVE_JOINTS = [-1.5666086673736572, -0.5103427171707153, 1.342707633972168, 0.031584497541189194, 2.07389760017395, -0.01760704815387726]
DEFAULT_CLEANUP_CONFIG_FILE = package_resource_path('config/cleanup_config.yaml')

GRIP_JOINTS = [0.01094, 0.50146, 1.17590, -0.02993, 1.47406, -0.00206]
GRIP_JOINT_TOLERANCE = 0.02

MAX_GRASP_ROT_RAD = np.deg2rad(90.0)
GRASP_YAW_BIAS_RAD = 0.0


# =========================
# 自定义放置姿态
# =========================
# 这是你手动摆动机器人后，从 /rm_driver/ArmCurrentState 读取到的末端四元数。
# 顺序为 [x, y, z, w]
PLACE_QUAT = [
    0.9996736367553412,
    -0.019567201537117926,
    0.004190573821055128,
    0.01588029254788618,
]

# 放置姿态额外绕工具局部 z 轴旋转 90 度。
# 如果实机测试发现方向反了，把 90.0 改成 -90.0。
# 如果后续要取消额外旋转，把 90.0 改成 0.0。
PLACE_YAW_OFFSET_RAD = np.deg2rad(90.0)


# =========================
# YOLO 消息处理控制
# =========================

accept_yolo_detection = False
YOLO_SETTLE_DELAY = 3.0
YOLO_IGNORE_FIRST_SECONDS = 1.0
yolo_accept_time = 0.0

# Startup handshake: first prove that the monitor publishes, then initialize
# the cold gripper, and only then require a valid command-matched open state.
REQUIRE_GRIPPER_MONITOR = True
GRIPPER_MONITOR_STARTUP_TIMEOUT_SEC = 5.0
GRIPPER_STATE_MAX_AGE_SEC = 0.5
INITIALIZE_GRIPPER_ON_STARTUP = True
GRIPPER_STARTUP_VALIDATION_TIMEOUT_SEC = 8.0
GRIPPER_STARTUP_MIN_STABLE_COUNT = 3
GRIPPER_COMMAND_TOPIC = TOPIC_GRIPPER_COMMAND
GRIPPER_STATE_TOPIC = TOPIC_GRIPPER_STATE
last_gripper_command_id = ""
last_gripper_command_stamp = 0.0


def _mark_actuation() -> int:
    global execution_actuation_seq
    execution_actuation_seq += 1
    return execution_actuation_seq


def _gripper_command_context(actuation_seq: int):
    task = current_task or {}
    return {
        "run_id": execution_run_id,
        "attempt_id": execution_attempt_id,
        "step": task.get("step", -1),
        "stage_seq": execution_stage_seq,
        "actuation_seq": int(actuation_seq),
        "stage_name": str(last_published_stage or ""),
    }


# ============================================================
# 验证/恢复闭环：阶段命名、阶段发布、运行上下文、控制订阅、中断等待
# ============================================================

def build_process_stage_names_for_task(task: dict):
    """
    为单个 catch_and_place 任务生成验证端可识别的 8 个过程阶段名。
    """
    task_step = int(task.get("step", 1))
    target_object_id = task.get("target_object_id")
    target_class = task.get("target_class", "object")
    target_name = target_object_id or target_class
    prefix = f"step_{task_step}_{target_name}"

    return {
        "approach_source": f"{prefix}__approach_source",
        "descend_to_grasp": f"{prefix}__descend_to_grasp",
        "close_gripper": f"{prefix}__close_gripper",
        "lift_object": f"{prefix}__lift_object",
        "move_to_target_above": f"{prefix}__move_to_target_above",
        "descend_to_place": f"{prefix}__descend_to_place",
        "open_gripper": f"{prefix}__open_gripper",
        "leave_target": f"{prefix}__leave_target",
    }


def publish_current_stage(stage_name: str):
    global current_stage_pub, last_published_stage

    stage_name = str(stage_name).strip()
    if not stage_name or current_stage_pub is None:
        return

    if last_published_stage == stage_name:
        return

    current_stage_pub.publish(String(data=stage_name))
    last_published_stage = stage_name
    print(f"[EXEC] current_stage = {stage_name}")


def publish_stage_status(stage_name: str, event: str):
    global status_pub, execution_run_id, execution_attempt_id, execution_stage_seq

    if status_pub is None:
        return

    stage_name = str(stage_name).strip()
    event = str(event).strip()
    if not stage_name or not event:
        return None

    event_stamp = float(rospy.Time.now().to_sec())
    payload = {
        "schema_version": 2,
        "event_type": "stage_event",
        "run_id": execution_run_id,
        "attempt_id": int(execution_attempt_id),
        "stage_seq": int(execution_stage_seq),
        "stage_name": stage_name,
        "event": event,
        "event_stamp": event_stamp,
    }
    status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
    print(f"[EXEC] stage_status = {payload}")
    return event_stamp


def _save_runtime_return_slots_cache(task: dict):
    """
    将执行端记录的 source_return_release_pose 持久化到本地 cache 文件。

    cleanup_planner.py 在生成 return_slots 时优先读取此文件，
    以获取执行端真实抓取前记录的安全释放位姿（比静态 action.py 更准确）。
    """
    cache_path = runtime_data_path("runtime_return_slots.json")

    try:
        step_value = int(task.get("step", 0))
    except Exception:
        return

    release_pose = task.get("source_return_release_pose")
    if release_pose is None or len(release_pose) < 2:
        return

    cache_dir = os.path.dirname(cache_path)
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError:
        pass

    cache = {}
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    cache[str(step_value)] = {
        "step": step_value,
        "target_object_id": task.get("target_object_id"),
        "class_name": task.get("target_class", ""),
        "source_return_release_pose": list(release_pose),
        "source_return_approach_pose": (
            list(task["source_return_approach_pose"])
            if task.get("source_return_approach_pose")
            else None
        ),
        "source_return_lift_pose": (
            list(task["source_return_lift_pose"])
            if task.get("source_return_lift_pose")
            else None
        ),
        "timestamp": time.time(),
    }

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        rospy.logwarn("[EXEC] failed to write runtime return_slots cache: %s", str(exc))


def publish_runtime_context_for_task(task: dict, action_type: str = "", status: str = "active"):
    """
    发布恢复系统所需的运行上下文。

    recovery_manager.py 依赖这个 topic 判断：
      - 当前执行到第几步；
      - 当前目标物体是什么；
      - 故障告警是否属于当前 step；
      - 应该从哪一步恢复。
    """
    global runtime_context_pub, latest_runtime_action_type
    global execution_run_id, execution_attempt_id, execution_stage_seq, execution_actuation_seq
    global last_gripper_command_id, last_gripper_command_stamp

    latest_runtime_action_type = str(action_type or "")

    if runtime_context_pub is None:
        return

    if task is None:
        return

    try:
        step_value = int(task.get("step", 0))
    except Exception:
        step_value = None

    # 持久化 return slot 位姿供 cleanup_planner 读取
    _save_runtime_return_slots_cache(task)

    payload = {
        "run_id": execution_run_id,
        "attempt_id": int(execution_attempt_id),
        "stage_seq": int(execution_stage_seq),
        "actuation_seq": int(execution_actuation_seq),
        "step": step_value,
        "target_object_id": task.get("target_object_id"),
        "target_class": task.get("target_class", ""),
        "target_place": task.get("target_place", []),
        "source_pose": task.get("source_pose"),
        "source_return_approach_pose": task.get("source_return_approach_pose"),
        "source_return_release_pose": task.get("source_return_release_pose"),
        "source_return_lift_pose": task.get("source_return_lift_pose"),
        "gripper_command_id": str(last_gripper_command_id or ""),
        "gripper_command_stamp": float(last_gripper_command_stamp or 0.0),
        "action_type": str(action_type or ""),
        "expected_support_object_id": task.get("expected_support_object_id"),
        "comment": task.get("comment", ""),
        "status": str(status or "active"),
        "timestamp": time.time(),
    }

    runtime_context_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
    print(f"[EXEC] runtime_context = {payload}")

def publish_recovery_ready(
    reason: str = "paused_for_recovery",
    ready: bool = True,
    request_id: str = "",
):
    """
    通知恢复端执行端状态（两阶段握手）。

    ready=False → 已收到暂停请求，正在等待当前动作退出。
    ready=True  → 已让出控制权，恢复端可以执行 safe_retreat。

    request_id 用于匹配恢复端发出的请求，防止 latch 残留误触发。
    """
    global recovery_ready_pub, current_task

    if recovery_ready_pub is None:
        return

    payload = {
        "ready": bool(ready),
        "request_id": str(request_id or ""),
        "state": "ready" if ready else "pending",
        "step": None if current_task is None else current_task.get("step"),
        "target_object_id": None if current_task is None else current_task.get("target_object_id"),
        "target_class": "" if current_task is None else current_task.get("target_class", ""),
        "action_type": str(reason),
        "reason": str(reason),
        "timestamp": time.time(),
    }

    recovery_ready_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
    print(f"[EXEC] recovery_ready = {payload}")


def publish_cleanup_result(request_id, success, reason="", detail=None):
    """
    发布清理动作结果。

    注意：
    - request_id 用于恢复端/测试端过滤旧 latch 消息；
    - cleanup_result 使用 latch=True，方便 wait_for_message 不错过结果；
    - 不影响 recovery_ready 握手。
    """
    global cleanup_result_pub

    payload = {
        "request_id": str(request_id or ""),
        "success": bool(success),
        "reason": str(reason or ""),
        "detail": detail or {},
        "timestamp": time.time(),
    }

    text = json.dumps(payload, ensure_ascii=False)

    if cleanup_result_pub is None:
        rospy.logwarn("[EXEC][CLEANUP] cleanup_result_pub is None; result=%s", text)
        return

    cleanup_result_pub.publish(String(data=text))
    rospy.logwarn("[EXEC][CLEANUP] published cleanup_result: %s", text)


def publish_cleanup_runtime_context(request_id, object_class, source_base_position, target_base_position, action_type):
    """
    发布清理运行上下文。

    如果你的 catch.py 中已有 publish_runtime_context_for_task()，
    这里会复用它；如果没有，则只打印日志，不阻塞清理。
    """
    task_info = {
        "step": None,
        "target_object_id": "cleanup_" + str(object_class or "object"),
        "target_class": str(object_class or ""),
        "target_place_pose": list(target_base_position or []),
        "comment": "cleanup pick-place request_id=%s" % str(request_id or ""),
        "is_cleanup": True,
    }

    try:
        publish_runtime_context_for_task(
            task_info,
            action_type=str(action_type or "cleanup_pick_place"),
            status="cleanup",
        )
    except NameError:
        rospy.logwarn(
            "[EXEC][CLEANUP] publish_runtime_context_for_task not found; "
            "skip runtime context. action_type=%s class=%s source=%s target=%s",
            str(action_type),
            str(object_class),
            str(source_base_position),
            str(target_base_position),
        )
    except Exception as exc:
        rospy.logwarn("[EXEC][CLEANUP] failed to publish cleanup runtime context: %s", str(exc))



def control_callback(msg: String):
    """
    接收恢复系统对执行端的反向控制：
      pause / resume / abort / resume_from_step
    """
    global execution_paused, execution_abort_requested, current_task, tasks_completed
    global execution_replan_requested, pending_resume_step, is_executing
    global recovery_pause_requested, latest_runtime_action_type, pending_recovery_request_id
    global cleanup_active

    raw = str(msg.data).strip()
    if not raw:
        return

    try:
        payload = json.loads(raw)
    except Exception:
        payload = {"cmd": raw}

    cmd = str(payload.get("cmd", "")).strip().lower()

    if cmd == "pause":
        execution_paused = True
        rospy.logwarn("Execution paused by external recovery system.")
        return

    if cmd == "resume":
        execution_paused = False
        execution_abort_requested = False
        # ``pause_for_recovery`` deliberately keeps these flags asserted after
        # an immediate (non-arm-motion) handoff so that no new detection can
        # start while the recovery manager owns the robot.  A matching resume
        # command ends that handoff; leaving either flag set would make the
        # next validation barrier cancel immediately as a stale recovery.
        recovery_pause_requested = False
        execution_replan_requested = False
        pending_resume_step = None
        pending_recovery_request_id = None
        rospy.logwarn("Execution resumed by external recovery system.")
        return

    if cmd == "abort":
        execution_abort_requested = True
        execution_paused = False
        current_task = None
        tasks_completed = True
        recovery_pause_requested = False
        pending_recovery_request_id = None
        rospy.logerr("Execution abort requested by external recovery system.")
        return

    if cmd == "pause_for_recovery":
        # 恢复端请求接管控制权。
        # 这是"暂停并让出控制权"，不是 abort。
        # 当前动作如果正在执行，让 catch_and_place 在最近的 interruptible_sleep / wait_if_paused_or_aborted 处退出。
        recovery_pause_requested = True
        execution_replan_requested = True
        execution_paused = False
        execution_abort_requested = False

        request_id = str(payload.get("request_id", ""))
        pending_recovery_request_id = request_id

        last_action = str(latest_runtime_action_type or "")

        rospy.logwarn(
            "[EXEC] pause_for_recovery requested by recovery system. is_executing=%s last_action=%s request_id=%s",
            str(is_executing),
            last_action,
            request_id,
        )

        if current_task is not None:
            publish_runtime_context_for_task(
                current_task,
                action_type="paused_for_recovery",
                status="recovery",
            )

        # 两阶段握手：
        # - 纯等待阶段（YOLO 检测/已暂停）：立即 ready=true，恢复端可直接接管。
        # - 臂运动/夹爪阶段（MoveL/MoveJ/close_gripper/open_gripper 进行中）：
        #   先 ready=false（ack），等 finally 块确认当前动作退出后再发 ready=true。
        #   夹爪动作期间不立即让出控制权：虽机械臂静止，但夹爪仍在状态变化中，
        #   safe_retreat 的上抬可能产生干涉。
        non_arm_motion_actions = {
            "waiting_detection",
            "waiting_detection_after_resume",
            "paused_for_recovery",
        }

        can_yield_immediately = (not is_executing) or (last_action in non_arm_motion_actions)

        publish_recovery_ready(
            reason=(
                "pause_for_recovery_immediate_non_arm_motion"
                if can_yield_immediately
                else "pause_for_recovery_pending_arm_motion"
            ),
            ready=can_yield_immediately,
            request_id=request_id,
        )
        return

    if cmd in {"resume_from_step", "set_resume_step"}:
        resume_step = payload.get("resume_step", payload.get("step", None))

        if resume_step is None:
            rospy.logerr("[EXEC] resume_from_step command missing resume_step")
            return

        try:
            resume_step = int(resume_step)
        except Exception:
            rospy.logerr("[EXEC] invalid resume_step: %s", str(resume_step))
            return

        if is_executing:
            # 当前 catch_and_place 正在执行中。
            # 不能直接改 task_queue，否则当前动作和新任务队列会冲突。
            # 先设置 replan 标志，让当前动作在 interruptible_sleep / wait_if_paused_or_aborted 中退出；
            # 然后在 object_pose_callback finally 里真正 apply_resume_from_step。
            pending_resume_step = resume_step
            execution_replan_requested = True
            execution_paused = False
            execution_abort_requested = False
            pending_recovery_request_id = None

            rospy.logwarn(
                "[EXEC] resume_from_step pending while executing: step=%d",
                resume_step,
            )
            return

        pending_recovery_request_id = None
        ok = apply_resume_from_step(resume_step, prepare_observation=True)
        if ok:
            rospy.logwarn("[EXEC] resume_from_step applied immediately: step=%d", resume_step)
        else:
            rospy.logerr("[EXEC] resume_from_step failed: step=%d", resume_step)
        return

    if cmd == "cleanup_pick_place":
        if not ENABLE_CLEANUP_CONTROL:
            rospy.logwarn(
                "[EXEC][CLEANUP] cleanup_pick_place received but ENABLE_CLEANUP_CONTROL=False; "
                "ignoring. Use cleanup_executor + pick_place_executor path instead."
            )
            publish_cleanup_result(
                request_id=str(payload.get("request_id", "")),
                success=False,
                reason="cleanup_control_disabled",
                detail={"payload": payload},
            )
            return

        request_id = str(payload.get("request_id", ""))
        object_class = str(payload.get("object_class", ""))

        source_base_position = payload.get("source_base_position", None)
        target_base_position = payload.get("target_base_position", None)

        if not source_base_position or not target_base_position:
            rospy.logerr("[EXEC][CLEANUP] invalid cleanup_pick_place payload: %s", str(payload))
            publish_cleanup_result(
                request_id=request_id,
                success=False,
                reason="invalid_request",
                detail={"payload": payload},
            )
            return

        # 可选安全检查：只允许恢复/暂停阶段执行 cleanup。
        # 如果你当前测试时 catch.py 没有进入 recovery 状态，可以先不启用这段。
        #
        # if not recovery_pause_requested and is_executing:
        #     publish_cleanup_result(
        #         request_id=request_id,
        #         success=False,
        #         reason="execution_busy_not_in_recovery",
        #         detail={"payload": payload},
        #     )
        #     return

        rospy.logwarn("[EXEC][CLEANUP] received cleanup_pick_place payload=%s", str(payload))

        cleanup_pick_place_by_base_pose(
            request_id=request_id,
            object_class=object_class,
            source_base_position=source_base_position,
            target_base_position=target_base_position,
        )

        return

    rospy.logwarn("[EXEC] unknown control cmd: %s payload=%s", cmd, payload)


def wait_if_paused_or_aborted():
    global execution_paused, execution_abort_requested, execution_replan_requested

    while not rospy.is_shutdown():
        if execution_replan_requested:
            rospy.logwarn("Execution replan flag detected.")
            return False

        if execution_abort_requested:
            rospy.logerr("Execution abort flag detected.")
            return False

        if not execution_paused:
            return True

        rospy.sleep(0.1)

    return False


def interruptible_sleep(seconds: float):
    """
    可被 pause/abort/replan 打断的 sleep。
    替代 catch_and_place() 中的 rospy.sleep()，使恢复系统能在动作间隙接管。
    """
    global execution_abort_requested, execution_replan_requested

    deadline = time.monotonic() + float(seconds)

    while not rospy.is_shutdown() and time.monotonic() < deadline:
        if not wait_if_paused_or_aborted():
            return False

        remaining = deadline - time.monotonic()
        rospy.sleep(min(0.1, max(remaining, 0.0)))

    return (not execution_abort_requested) and (not execution_replan_requested)


def _publish_stage_started(stage_names, key, task=None):
    global execution_stage_seq
    if stage_names is None:
        return None
    execution_stage_seq += 1
    publish_current_stage(stage_names[key])
    event_stamp = publish_stage_status(stage_names[key], "started")

    if task is not None:
        publish_runtime_context_for_task(task, action_type=key, status="active")
    return event_stamp


def _publish_stage_finished(stage_names, key):
    if stage_names is None:
        return None
    return publish_stage_status(stage_names[key], "finished")


def validation_result_callback(msg: String):
    global pending_validation_result, shadow_validation_requests
    try:
        result = ValidationResult.from_json(str(msg.data or ""))
    except Exception as exc:
        rospy.logwarn_throttle(2.0, "[EXEC][TX] invalid validation result: %s", str(exc))
        return

    with validation_condition:
        request = pending_validation_request
        if request is not None:
            matched, _ = result_matches_request(request, result)
            if matched:
                pending_validation_result = result
                validation_condition.notify_all()

        if result.request_id in shadow_validation_requests:
            rospy.logwarn(
                "[EXEC][TX][SHADOW] request=%s stage=%s status=%s versions=%s message=%s",
                result.request_id,
                result.context.stage_name,
                result.status,
                str(result.scene_versions),
                result.message,
            )
            if result.terminal:
                shadow_validation_requests.pop(result.request_id, None)


def _publish_validation_request(request: ValidationRequest, event: str = "start"):
    if validation_request_pub is None:
        return
    payload = request.to_dict()
    payload["event"] = str(event)
    validation_request_pub.publish(
        String(data=json.dumps(payload, ensure_ascii=False, sort_keys=True))
    )


def _clear_pending_validation(request_id: str):
    global pending_validation_request, pending_validation_result
    with validation_condition:
        if (
            pending_validation_request is not None
            and pending_validation_request.request_id == str(request_id)
        ):
            pending_validation_request = None
            pending_validation_result = None


def wait_validation_barrier(
    *,
    stage_name: str,
    policy: str,
    not_before: Optional[float],
    required_distinct_frames: int,
    gripper_command_id: str = "",
) -> bool:
    """Request a validation transaction and optionally block execution."""
    global pending_validation_request, pending_validation_result
    global validation_blocked_status, execution_actuation_seq

    if BARRIER_MODE == "off":
        return True

    event_stamp = float(not_before if not_before is not None else rospy.Time.now().to_sec())
    context = TemporalContext(
        run_id=execution_run_id,
        attempt_id=execution_attempt_id,
        stage_seq=execution_stage_seq,
        stage_name=str(stage_name),
        barrier_id="barrier_%s" % uuid.uuid4().hex[:12],
    )
    request = ValidationRequest(
        request_id="validation_%s" % uuid.uuid4().hex[:12],
        context=context,
        policy=str(policy),
        not_before=event_stamp,
        required_distinct_frames=max(1, int(required_distinct_frames)),
        timeout_sec=float(VALIDATION_TIMEOUT_SEC),
        created_stamp=float(rospy.Time.now().to_sec()),
        gripper_command_id=str(gripper_command_id or ""),
    )

    if BARRIER_MODE == "shadow":
        while len(shadow_validation_requests) >= 100:
            shadow_validation_requests.pop(next(iter(shadow_validation_requests)))
        shadow_validation_requests[request.request_id] = request
        _publish_validation_request(request)
        return True

    if BARRIER_MODE != "enforce":
        rospy.logerr("[EXEC][TX] invalid barrier mode: %s", BARRIER_MODE)
        validation_blocked_status = ValidationStatus.INVALID
        return False

    start_monotonic = time.monotonic()
    last_publish_monotonic = 0.0
    starting_actuation_seq = int(execution_actuation_seq)
    with validation_condition:
        pending_validation_request = request
        pending_validation_result = None

    while not rospy.is_shutdown():
        if execution_abort_requested or recovery_pause_requested:
            _publish_validation_request(request, event="cancel")
            _clear_pending_validation(request.request_id)
            validation_blocked_status = ValidationStatus.CANCELLED
            return False
        if int(execution_actuation_seq) != starting_actuation_seq:
            _publish_validation_request(request, event="cancel")
            _clear_pending_validation(request.request_id)
            validation_blocked_status = ValidationStatus.CANCELLED
            rospy.logerr("[EXEC][TX] actuation changed while barrier was active")
            return False

        now_monotonic = time.monotonic()
        if now_monotonic - last_publish_monotonic >= VALIDATION_REQUEST_RETRY_SEC:
            _publish_validation_request(request)
            last_publish_monotonic = now_monotonic

        with validation_condition:
            result = pending_validation_result
            if result is None or not result.terminal:
                validation_condition.wait(timeout=0.1)
                result = pending_validation_result

        if result is not None and result.terminal:
            _clear_pending_validation(request.request_id)
            if result.status == ValidationStatus.PASSED:
                validation_blocked_status = ""
                return True
            validation_blocked_status = result.status
            return False

        if now_monotonic - start_monotonic >= VALIDATION_TIMEOUT_SEC:
            _publish_validation_request(request, event="cancel")
            _clear_pending_validation(request.request_id)
            validation_blocked_status = ValidationStatus.TIMEOUT
            return False

    _clear_pending_validation(request.request_id)
    validation_blocked_status = ValidationStatus.CANCELLED
    return False


# ============================================================
# 观测准备与姿态辅助函数
# ============================================================

def wait_for_arm_ready_pose(timeout=20.0):
    """阻塞等待机械臂实际到达 arm_ready_pose 关节角度"""
    start_monotonic = time.monotonic()
    rate = rospy.Rate(50)
    rospy.loginfo("Waiting for arm to stabilize at ready pose...")

    while not rospy.is_shutdown():
        if not wait_if_paused_or_aborted():
            return False

        try:
            msg = rospy.wait_for_message("/rm_driver/ArmCurrentState", ArmState, timeout=1.0)
            current_joints = getattr(msg, "joint", getattr(msg, "q", None))

            if current_joints and len(current_joints) == 6:
                if all(abs(c - t) < ARM_READY_TOLERANCE for c, t in zip(current_joints, ARM_READY_JOINTS)):
                    rospy.loginfo("Arm has reached ready pose.")
                    return True

        except rospy.ROSException:
            pass

        if time.monotonic() - start_monotonic > timeout:
            rospy.logwarn("Timeout waiting for arm ready pose.")
            return False

        rate.sleep()


def prepare_yolo_observation_for_next_task(timeout=20.0):
    """
    每次准备执行一个新任务前调用：
    1. 禁止处理 YOLO 消息
    2. 等待机械臂实际到达观测位
    3. 等待 YOLO 刷新当前画面
    4. 再允许处理 YOLO 消息
    """
    global accept_yolo_detection, yolo_accept_time

    accept_yolo_detection = False

    rospy.loginfo("Preparing YOLO observation for current task...")
    rospy.loginfo("Waiting for arm to be at observation pose...")

    if not wait_for_arm_ready_pose(timeout=timeout):
        rospy.logerr("Failed to prepare observation: arm did not reach ready pose.")
        return False

    rospy.loginfo(
        "Arm is at observation pose. Waiting %.1f seconds for YOLO to refresh current frame...",
        YOLO_SETTLE_DELAY,
    )
    if not interruptible_sleep(YOLO_SETTLE_DELAY):
        return False

    yolo_accept_time = rospy.Time.now().to_sec() + YOLO_IGNORE_FIRST_SECONDS
    accept_yolo_detection = True

    rospy.loginfo(
        "YOLO detection enabled. Ignoring first %.1f seconds, then accepting current object messages.",
        YOLO_IGNORE_FIRST_SECONDS,
    )

    return True


def wait_for_joint_pose(target_joints, tolerance=0.02, timeout=10.0):
    """阻塞等待机械臂实际到达指定关节角度"""
    start_monotonic = time.monotonic()
    rate = rospy.Rate(50)
    rospy.loginfo("Waiting for arm to stabilize at target joint pose...")

    while not rospy.is_shutdown():
        if not wait_if_paused_or_aborted():
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

        if time.monotonic() - start_monotonic > timeout:
            rospy.logwarn("Timeout waiting for target joint pose.")
            return False

        rate.sleep()


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


def get_reference_grasp_quaternion():
    """
    先运动到你定义的 grip_pose，再读取这一姿态下的真实四元数。
    这样抓取姿态由你设定的抓取参考位姿来决定，而不是由回调瞬间姿态决定。
    """
    if not grip_pose():
        raise RuntimeError("Failed to send grip_pose command")

    if not wait_for_joint_pose(GRIP_JOINTS, tolerance=GRIP_JOINT_TOLERANCE, timeout=10.0):
        raise RuntimeError("Robot did not reach grip_pose joints in time")

    if not interruptible_sleep(0.3):
        raise RuntimeError("Interrupted while waiting after grip_pose")

    return get_current_tool_quaternion(timeout=2.0)


def normalize_half_turn_rad(angle_rad):
    return (angle_rad + np.pi / 2.0) % np.pi - np.pi / 2.0


def rotate_tool_quaternion_by_local_z(base_quaternion, delta_yaw_rad):
    base_rotation = R.from_quat(base_quaternion)
    yaw_rotation = R.from_euler("z", delta_yaw_rad, degrees=False)
    rotated = base_rotation * yaw_rotation
    return rotated.as_quat()


def get_place_quaternion():
    """
    返回放置姿态四元数。
    """
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
# 任务加载与坐标转换
# ============================================================

def load_task_queue_from_action():
    """
    从 action.py 的 ACTION_PLAN 加载任务，并转换成 catch.py 原本使用的 task_queue 格式。
    """
    try:
        default_action_file = find_package_resource("action.py") or package_resource_path("action.py")
        action_file = str(rospy.get_param("~plan_file", default_action_file)).strip()
        if not os.path.exists(action_file):
            rospy.logwarn(f"action.py not found: {action_file}")
            return []

        spec = importlib.util.spec_from_file_location("generated_action_module", action_file)
        if spec is None or spec.loader is None:
            rospy.logwarn("Failed to create spec for action.py")
            return []

        action_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(action_mod)

        direct_queue = getattr(action_mod, "task_queue", None)
        if isinstance(direct_queue, list) and direct_queue:
            rospy.loginfo(f"Loaded {len(direct_queue)} tasks from action.py (task_queue)")
            return direct_queue

        action_plan = getattr(action_mod, "ACTION_PLAN", None)
        if action_plan is None:
            get_action_plan = getattr(action_mod, "get_action_plan", None)
            if callable(get_action_plan):
                action_plan = get_action_plan()
            else:
                rospy.logwarn("No task_queue / ACTION_PLAN / get_action_plan found in action.py")
                return []

        converted_queue = []
        for step in action_plan:
            converted_queue.append({
                "target_object_id": step.get("target_object_id"),
                "target_class": step.get("source_class", ""),
                "target_place": step.get("target_place", [0.0, 0.0, 0.0]),
                "step": step.get("step"),
                "source_pose": step.get("source_pose"),
                "action_type": step.get("action_type"),
                "expected_support_object_id": step.get("expected_support_object_id"),
                "comment": step.get("comment", ""),
            })

        rospy.loginfo(f"Loaded {len(converted_queue)} tasks from action.py (ACTION_PLAN converted)")
        return converted_queue

    except Exception as e:
        rospy.logerr(f"Failed to load task queue from action.py: {e}")
        return []


def _task_step_value(task):
    try:
        return int(task.get("step", 0))
    except Exception:
        return 0


def apply_resume_from_step(resume_step, prepare_observation=True):
    """
    从 action.py 重新加载任务队列，并从 resume_step 开始继续执行。
    """
    global task_queue, current_task, tasks_completed, is_executing
    global execution_paused, execution_abort_requested
    global execution_replan_requested, pending_resume_step
    global accept_yolo_detection, yolo_accept_time
    global last_published_stage
    global validation_blocked_status, execution_attempt_id
    global recovery_pause_requested, pending_recovery_request_id

    try:
        resume_step = int(resume_step)
    except Exception:
        rospy.logerr("[EXEC] invalid resume_step in apply_resume_from_step: %s", str(resume_step))
        return False

    all_tasks = load_task_queue_from_action()
    all_tasks = sorted(all_tasks, key=_task_step_value)

    remaining = [
        task for task in all_tasks
        if _task_step_value(task) >= resume_step
    ]

    if not remaining:
        rospy.logerr(
            "[EXEC] resume_from_step failed: no task with step >= %s",
            str(resume_step),
        )
        return False

    current_task = remaining[0]
    task_queue = remaining[1:]

    tasks_completed = False
    is_executing = False

    execution_paused = False
    execution_abort_requested = False
    execution_replan_requested = False
    pending_resume_step = None
    # Completing resume_from_step also completes any previous recovery
    # handoff.  This is essential for the immediate-handoff path, which has no
    # running object_pose_callback/finally block available to clear the flag.
    recovery_pause_requested = False
    pending_recovery_request_id = None
    validation_blocked_status = ""
    execution_attempt_id += 1

    accept_yolo_detection = False
    yolo_accept_time = 0.0

    # 允许同名阶段重新发布，避免恢复后 validation_monitor 看不到新阶段。
    last_published_stage = None

    rospy.logwarn(
        "[EXEC] resume_from_step applied: resume_step=%d current_task_step=%s remaining=%d",
        resume_step,
        str(current_task.get("step")),
        len(task_queue),
    )

    print("")
    print("[EXEC] RESUME FROM STEP")
    print(f"  resume_step = {resume_step}")
    print(
        "  current_task: step={step} target={target}({cls}) place={place} support={support}".format(
            step=current_task.get("step"),
            target=current_task.get("target_object_id"),
            cls=current_task.get("target_class"),
            place=current_task.get("target_place"),
            support=current_task.get("expected_support_object_id"),
        )
    )
    print("  remaining:")
    for task in task_queue:
        print(
            "    step={step} target={target}({cls}) place={place} support={support}".format(
                step=task.get("step"),
                target=task.get("target_object_id"),
                cls=task.get("target_class"),
                place=task.get("target_place"),
                support=task.get("expected_support_object_id"),
            )
        )
    print("", flush=True)

    publish_runtime_context_for_task(
        current_task,
        action_type="waiting_detection_after_resume",
        status="active",
    )

    if prepare_observation:
        if not prepare_yolo_observation_for_next_task(timeout=20.0):
            rospy.logerr("[EXEC] failed to prepare YOLO observation after resume_from_step")
            return False

    return True


def convert(x, y, z, x1, y1, z1, rx, ry, rz, obj_angle_cam=None):
    """
    相机坐标系物体到机械臂基坐标系转换函数。
    """
    global rotation_matrix, translation_vector

    obj_camera_coordinates = np.array([x, y, z])
    end_effector_pose = np.array([x1, y1, z1, rx, ry, rz])

    T_camera_to_end_effector = np.eye(4)
    T_camera_to_end_effector[:3, :3] = rotation_matrix
    T_camera_to_end_effector[:3, 3] = translation_vector

    position = end_effector_pose[:3]
    orientation = R.from_euler("xyz", end_effector_pose[3:], degrees=False).as_matrix()

    T_base_to_end_effector = np.eye(4)
    T_base_to_end_effector[:3, :3] = orientation
    T_base_to_end_effector[:3, 3] = position

    obj_camera_coordinates_homo = np.append(obj_camera_coordinates, [1])
    obj_end_effector_coordinates_homo = T_camera_to_end_effector.dot(obj_camera_coordinates_homo)
    obj_base_coordinates_homo = T_base_to_end_effector.dot(obj_end_effector_coordinates_homo)
    obj_base_coordinates = obj_base_coordinates_homo[:3]

    obj_orientation_matrix = T_base_to_end_effector[:3, :3].dot(rotation_matrix)
    obj_orientation_euler = R.from_matrix(obj_orientation_matrix).as_euler("xyz", degrees=False)

    if obj_angle_cam is not None:
        r_obj_cam = R.from_euler("z", obj_angle_cam, degrees=False)
        R_camera_to_end = R.from_matrix(rotation_matrix)
        r_base_obj = R.from_matrix(T_base_to_end_effector[:3, :3]) * (R_camera_to_end * r_obj_cam)
        obj_orientation_euler = r_base_obj.as_euler("xyz", degrees=False)

    obj_base_pose = np.hstack((obj_base_coordinates, obj_orientation_euler))
    return obj_base_pose


# ============================================================
# YOLO /object_pose 回调
# ============================================================

def object_pose_callback(data):
    """
    YOLO /object_pose 回调。
    """
    global current_task, is_executing, task_queue, tasks_completed
    global accept_yolo_detection, yolo_accept_time
    global execution_replan_requested, pending_resume_step
    global recovery_pause_requested
    global pending_recovery_request_id
    global validation_blocked_status

    if not accept_yolo_detection:
        return

    capture_lower_bound = getattr(data, "capture_lower_bound", None)
    if capture_lower_bound is not None:
        try:
            if yolo_accept_time > 0.0 and float(capture_lower_bound) < yolo_accept_time:
                return
        except (TypeError, ValueError):
            return
    elif yolo_accept_time > 0.0 and rospy.Time.now().to_sec() < yolo_accept_time:
        # Compatibility path for legacy ObjectInfo, which has no source stamp.
        return

    if not current_task or is_executing or tasks_completed:
        return

    if data.object_class != current_task["target_class"]:
        return

    target_label = current_task.get("target_object_id") or current_task.get("target_class")
    print(f"Detected target object: {target_label} / class={data.object_class} at ({data.x}, {data.y}, {data.z})")

    accept_yolo_detection = False
    is_executing = True

    try:
        if not wait_if_paused_or_aborted():
            return

        arm_pose_msg = rospy.wait_for_message("/rm_driver/Arm_Current_State", Arm_Current_State, timeout=10.0)
        arm_orientation_msg = rospy.wait_for_message("/rm_driver/ArmCurrentState", ArmState, timeout=10.0)

        grasp_angle = getattr(data, "angle", None) if hasattr(data, "angle") else None
        if grasp_angle is not None:
            grasp_angle = normalize_half_turn_rad(float(grasp_angle))
            print(f"Grasp rotation from vision: {np.degrees(grasp_angle):.2f} deg")

        result = convert(
            data.x,
            data.y,
            data.z,
            arm_pose_msg.Pose[0],
            arm_pose_msg.Pose[1],
            arm_pose_msg.Pose[2],
            arm_pose_msg.Pose[3],
            arm_pose_msg.Pose[4],
            arm_pose_msg.Pose[5],
        )

        print(f"Target {target_label} converted pose in base frame: {result[:3]}")

        success = catch_and_place(
            result,
            arm_orientation_msg,
            current_task["target_place"],
            obj_angle=grasp_angle,
            current_task_info=current_task,
        )

        if success:
            print(f"Successfully handled task for {target_label}")

            if task_queue:
                current_task = task_queue.pop(0)
                next_label = current_task.get("target_object_id") or current_task.get("target_class")
                print(
                    f"Moving to next task: Target {next_label}, "
                    f"Class {current_task['target_class']}, Place {current_task['target_place']}"
                )

                publish_runtime_context_for_task(
                    current_task,
                    action_type="waiting_detection",
                    status="active",
                )

                if not prepare_yolo_observation_for_next_task(timeout=20.0):
                    rospy.logerr("Failed to prepare YOLO observation for next task. Stopping.")
                    current_task = None
                    tasks_completed = True
                    pending_recovery_request_id = None

            else:
                publish_runtime_context_for_task(
                    current_task,
                    action_type="completed",
                    status="completed",
                )
                current_task = None
                tasks_completed = True
                pending_recovery_request_id = None
                print("All tasks completed!")

        else:
            if recovery_pause_requested:
                print(
                    f"Task for {target_label} interrupted by pause_for_recovery. "
                    "Will hand over control to recovery manager."
                )
                publish_runtime_context_for_task(
                    current_task,
                    action_type="paused_for_recovery",
                    status="recovery",
                )

                # 不要 tasks_completed=True
                # 不要 current_task=None
                # 不要发布 aborted
            elif execution_replan_requested and pending_resume_step is not None:
                print(
                    f"Task for {target_label} interrupted by resume_from_step. "
                    "Will apply pending resume plan."
                )
            elif validation_blocked_status:
                blocked_status = str(validation_blocked_status)
                runtime_status = (
                    "waiting_recovery"
                    if blocked_status == ValidationStatus.FAILED
                    else "waiting_vision"
                )
                rospy.logerr(
                    "[EXEC][TX] task blocked by validation status=%s; preserving current task",
                    blocked_status,
                )
                publish_runtime_context_for_task(
                    current_task,
                    action_type=blocked_status,
                    status=runtime_status,
                )
            else:
                print(f"Failed to handle task for {target_label}. Stopping.")
                publish_runtime_context_for_task(
                    current_task,
                    action_type="failed",
                    status="aborted",
                )
                tasks_completed = True
                current_task = None
                pending_recovery_request_id = None

    except rospy.ROSException as e:
        print(f"Error waiting for messages during task execution: {e}")

        if recovery_pause_requested:
            rospy.logwarn(
                "[EXEC] ROSException occurred while pause_for_recovery is active; "
                "keep current_task and hand over to recovery manager."
            )
            publish_runtime_context_for_task(
                current_task,
                action_type="paused_for_recovery",
                status="recovery",
            )
        else:
            if task_queue:
                current_task = task_queue.pop(0)
                next_label = current_task.get("target_object_id") or current_task.get("target_class")
                print(
                    f"Skipping to next task: Target {next_label}, "
                    f"Class {current_task['target_class']}, Place {current_task['target_place']}"
                )

                publish_runtime_context_for_task(
                    current_task,
                    action_type="waiting_detection_after_exception",
                    status="active",
                )

                if not prepare_yolo_observation_for_next_task(timeout=20.0):
                    rospy.logerr("Failed to prepare YOLO observation after exception. Stopping.")
                    current_task = None
                    tasks_completed = True
                    pending_recovery_request_id = None
            else:
                current_task = None
                tasks_completed = True
                pending_recovery_request_id = None

    except Exception as e:
        rospy.logerr(f"Unexpected error in object_pose_callback: {e}")

        if recovery_pause_requested:
            rospy.logwarn(
                "[EXEC] exception occurred while pause_for_recovery is active; "
                "keep current_task and hand over to recovery manager."
            )
            publish_runtime_context_for_task(
                current_task,
                action_type="paused_for_recovery",
                status="recovery",
            )
        else:
            publish_runtime_context_for_task(
                current_task,
                action_type="exception",
                status="aborted",
            )
            current_task = None
            tasks_completed = True
            pending_recovery_request_id = None

    finally:
        is_executing = False

        if recovery_pause_requested:
            rospy.logwarn("[EXEC] current action exited for recovery handoff")

            if current_task is not None:
                publish_runtime_context_for_task(
                    current_task,
                    action_type="paused_for_recovery",
                    status="recovery",
                )

            publish_recovery_ready(
                reason="current_action_interrupted_for_recovery",
                ready=True,
                request_id=pending_recovery_request_id or "",
            )

            pending_recovery_request_id = None

            # pause_for_recovery 只是让出控制权，不是断点恢复。
            # resume_from_step 应由 recovery_manager 在 safe_retreat 完成后发送。
            recovery_pause_requested = False
            execution_replan_requested = False

            return

        if execution_replan_requested and pending_resume_step is not None:
            resume_step = pending_resume_step
            rospy.logwarn(
                "[EXEC] applying pending resume_from_step after current action exits: step=%s",
                str(resume_step),
            )
            apply_resume_from_step(resume_step, prepare_observation=True)


# ============================================================
# 机器人底层运动指令
# ============================================================

def stamped_detections_callback(msg: DetectedObjects):
    """Select the current task target from a capture-stamped wrist frame."""
    global current_task, is_executing, tasks_completed

    if not current_task or is_executing or tasks_completed:
        return

    names = list(getattr(msg, "names", []) or [])
    scores = list(getattr(msg, "scores", []) or [])
    positions = list(getattr(msg, "positions_xyz", []) or [])
    angles = list(getattr(msg, "angles", []) or [])
    valid_flags = list(getattr(msg, "position_valid", []) or [])

    count = len(names)
    if len(positions) != count * 3 or len(angles) != count or len(valid_flags) != count:
        rospy.logwarn_throttle(
            2.0,
            "Invalid stamped wrist detection lengths: names=%d positions=%d angles=%d valid=%d",
            count,
            len(positions),
            len(angles),
            len(valid_flags),
        )
        return

    target_class = str(current_task.get("target_class", ""))
    candidates = []
    for index, name in enumerate(names):
        if str(name) != target_class or not bool(valid_flags[index]):
            continue
        score = float(scores[index]) if index < len(scores) else 0.0
        candidates.append((score, index))

    if not candidates:
        return

    _, index = max(candidates, key=lambda item: item[0])
    base = index * 3
    data = SimpleNamespace(
        object_class=target_class,
        x=float(positions[base]),
        y=float(positions[base + 1]),
        z=float(positions[base + 2]),
        angle=float(angles[index]),
        capture_stamp=float(msg.header.stamp.to_sec()),
        capture_lower_bound=float(getattr(msg, "capture_lower_bound", 0.0)),
        source_frame_seq=int(getattr(msg, "frame_seq", 0)),
    )
    object_pose_callback(data)


def movej_type(joint, speed):
    try:
        moveJ_pub = rospy.Publisher("/rm_driver/MoveJ_Cmd", MoveJ, queue_size=1)
        rospy.sleep(0.5)

        move_joint = MoveJ()
        move_joint.joint = joint
        move_joint.speed = speed

        _mark_actuation()
        moveJ_pub.publish(move_joint)
        return True

    except Exception as e:
        rospy.logerr(f"Error publishing MoveJ command: {e}")
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

        _mark_actuation()
        moveJ_P_pub.publish(move_joint_pose)
        return True

    except Exception as e:
        rospy.logerr(f"Error publishing MoveJ_P command: {e}")
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

        _mark_actuation()
        moveL_pub.publish(move_line_pose)
        return True

    except Exception as e:
        rospy.logerr(f"Error publishing MoveL command: {e}")
        return False


def arm_ready_pose():
    try:
        moveJ_pub = rospy.Publisher("/rm_driver/MoveJ_Cmd", MoveJ, queue_size=1)
        rospy.sleep(1)

        pic_joint = MoveJ()
        pic_joint.joint = ARM_READY_JOINTS
        pic_joint.speed = 0.2

        _mark_actuation()
        moveJ_pub.publish(pic_joint)
        return True

    except Exception as e:
        rospy.logerr(f"Error publishing arm_ready_pose command: {e}")
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

    该位姿用于让臂上相机观察整个搭建区域，服务于 cleanup scan-only / 后续自动清理。
    优先从 cleanup_config.yaml 读取 cleanup_observe_joints；读取失败时使用 CLEANUP_OBSERVE_JOINTS。
    """
    try:
        joints = _load_cleanup_observe_joints_from_config(config_file=config_file)

        moveJ_pub = rospy.Publisher("/rm_driver/MoveJ_Cmd", MoveJ, queue_size=1)
        rospy.sleep(1)

        cmd = MoveJ()
        cmd.joint = joints
        cmd.speed = 0.15

        moveJ_pub.publish(cmd)
        print(f"Moved to cleanup observe pose: {joints}")
        return True

    except Exception as e:
        rospy.logerr(f"Error publishing cleanup_observe_pose command: {e}")
        return False


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
        rospy.logerr(f"Error publishing grip_pose command: {e}")
        return False


# ============================================================
# 核心动作：融合版 catch_and_place
# ============================================================

def catch_and_place(
    obj_pose_in_base,
    arm_orientation_msg,
    target_place_pose,
    obj_angle=None,
    current_task_info=None,
):
    try:
        print("Starting catch-and-place sequence...")

        if not wait_if_paused_or_aborted():
            return False

        stage_names = None
        if current_task_info is not None:
            stage_names = build_process_stage_names_for_task(current_task_info)

        grasp_rotation = normalize_half_turn_rad(float(obj_angle)) if obj_angle is not None else 0.0
        grasp_rotation = float(np.clip(grasp_rotation, -MAX_GRASP_ROT_RAD, MAX_GRASP_ROT_RAD))
        grasp_rotation += GRASP_YAW_BIAS_RAD

        print(f"  Applying tool yaw rotation for grasp: {np.degrees(grasp_rotation):.2f} deg")

        base_quat = get_reference_grasp_quaternion()
        target_quat = rotate_tool_quaternion_by_local_z(base_quat, grasp_rotation)

        # 记录"安全放回原抓取位"的 7D 位姿。
        # 恢复端只能使用这些抓取时记录的姿态，禁止用恢复时的当前 TCP 姿态拼接 source_pose。
        if current_task_info is not None:
            src_x = float(obj_pose_in_base[0] - 0.055)
            src_y = float(obj_pose_in_base[1])
            src_z = float(obj_pose_in_base[2])
            current_task_info["source_return_approach_pose"] = [
                src_x, src_y, src_z + 0.10,
                target_quat[0], target_quat[1], target_quat[2], target_quat[3],
            ]
            current_task_info["source_return_release_pose"] = [
                src_x, src_y, src_z + 0.035,
                target_quat[0], target_quat[1], target_quat[2], target_quat[3],
            ]
            current_task_info["source_return_lift_pose"] = [
                src_x, src_y, src_z + 0.10,
                target_quat[0], target_quat[1], target_quat[2], target_quat[3],
            ]

        place_quat = get_place_quaternion()
        print(f"  Using custom place quaternion: {place_quat}")

        # --- Step 1: approach_source，移动到预抓取位置 ---
        if not wait_if_paused_or_aborted():
            return False
        _publish_stage_started(stage_names, "approach_source", current_task_info)

        print(f"  Moving to object near: {obj_pose_in_base[:3]}")
        approach_offset = 0.07

        if not movejp_type([
            obj_pose_in_base[0] - 0.055,
            obj_pose_in_base[1],
            obj_pose_in_base[2] + approach_offset,
            target_quat[0],
            target_quat[1],
            target_quat[2],
            target_quat[3],
        ], 0.2):
            rospy.logerr("Failed to move near object.")
            return False

        if not interruptible_sleep(4.0):
            return False
        _publish_stage_finished(stage_names, "approach_source")

        # --- Step 2: descend_to_grasp，移动到抓取位置 ---
        if not wait_if_paused_or_aborted():
            return False
        _publish_stage_started(stage_names, "descend_to_grasp", current_task_info)

        print(f"  Moving to object: {obj_pose_in_base[:3]}")

        if not movel_type([
            obj_pose_in_base[0] - 0.055,
            obj_pose_in_base[1],
            obj_pose_in_base[2] + 0.03,
            target_quat[0],
            target_quat[1],
            target_quat[2],
            target_quat[3],
        ], 0.15):
            rospy.logerr("Failed to move to object.")
            return False

        if not interruptible_sleep(4.0):
            return False
        _publish_stage_finished(stage_names, "descend_to_grasp")

        # --- Step 3: close_gripper，关闭夹爪 ---
        if not wait_if_paused_or_aborted():
            return False
        _publish_stage_started(stage_names, "close_gripper", current_task_info)

        print("  Closing gripper...")
        if not gripper_close():
            rospy.logerr("Failed to publish close-gripper command with temporal context.")
            return False

        if not interruptible_sleep(4.0):
            return False
        _publish_stage_finished(stage_names, "close_gripper")

        # --- Step 4: lift_object，抓取后上抬 ---
        if not wait_if_paused_or_aborted():
            return False
        _publish_stage_started(stage_names, "lift_object", current_task_info)

        print(f"  Moving to safety height: {obj_pose_in_base[:3]}")
        safety_height_offset = 0.1

        if not movel_type([
            obj_pose_in_base[0],
            obj_pose_in_base[1],
            obj_pose_in_base[2] + safety_height_offset,
            target_quat[0],
            target_quat[1],
            target_quat[2],
            target_quat[3],
        ], 0.2):
            rospy.logerr("Failed to move to safety height.")
            return False

        if not interruptible_sleep(4.0):
            return False
        lift_finished_stamp = _publish_stage_finished(stage_names, "lift_object")
        if (
            GRASP_BARRIER_ENABLED
            and stage_names is not None
            and not bool((current_task_info or {}).get("is_cleanup", False))
            and not wait_validation_barrier(
                stage_name=stage_names["lift_object"],
                policy="grasp_commit",
                not_before=lift_finished_stamp,
                required_distinct_frames=GRASP_REQUIRED_FRAMES,
                gripper_command_id=last_gripper_command_id,
            )
        ):
            rospy.logerr("Grasp validation barrier did not pass.")
            return False

        # --- Step 5: move_to_target_above，移动到目标上方 ---
        if not wait_if_paused_or_aborted():
            return False
        _publish_stage_started(stage_names, "move_to_target_above", current_task_info)

        print(f"  Moving to target place with custom place orientation: {target_place_pose}")
        place_above_offset = 0.1

        if not movejp_type([
            target_place_pose[0],
            target_place_pose[1],
            target_place_pose[2] + place_above_offset,
            place_quat[0],
            place_quat[1],
            place_quat[2],
            place_quat[3],
        ], 0.2):
            rospy.logerr("Failed to move above target place.")
            return False

        if not interruptible_sleep(4.0):
            return False
        _publish_stage_finished(stage_names, "move_to_target_above")

        # --- Step 6: descend_to_place，下降到放置位置 ---
        if not wait_if_paused_or_aborted():
            return False
        _publish_stage_started(stage_names, "descend_to_place", current_task_info)

        print("  Moving down to place with custom place orientation...")

        if not movel_type([
            target_place_pose[0],
            target_place_pose[1],
            target_place_pose[2] + 0.018,
            place_quat[0],
            place_quat[1],
            place_quat[2],
            place_quat[3],
        ], 0.15):
            rospy.logerr("Failed to move down to place.")
            return False

        if not interruptible_sleep(5.0):
            return False
        _publish_stage_finished(stage_names, "descend_to_place")

        # --- Step 7: open_gripper，打开夹爪释放物体 ---
        if not wait_if_paused_or_aborted():
            return False
        _publish_stage_started(stage_names, "open_gripper", current_task_info)

        print("  Opening gripper...")
        if not gripper_open():
            rospy.logerr("Failed to publish open-gripper command with temporal context.")
            return False

        if not interruptible_sleep(4.0):
            return False
        _publish_stage_finished(stage_names, "open_gripper")

        # --- Step 8: leave_target，放置后上抬离开目标 ---
        if not wait_if_paused_or_aborted():
            return False
        _publish_stage_started(stage_names, "leave_target", current_task_info)

        print("  Moving to safety position with custom place orientation...")

        if not movel_type([
            target_place_pose[0],
            target_place_pose[1],
            target_place_pose[2] + 0.04,
            place_quat[0],
            place_quat[1],
            place_quat[2],
            place_quat[3],
        ], 0.2):
            rospy.logerr("Failed to move to safety position.")
            return False

        if not interruptible_sleep(4.0):
            return False
        leave_finished_stamp = _publish_stage_finished(stage_names, "leave_target")
        if (
            PLACE_BARRIER_ENABLED
            and stage_names is not None
            and not bool((current_task_info or {}).get("is_cleanup", False))
            and not wait_validation_barrier(
                stage_name=stage_names["leave_target"],
                policy="place_commit",
                not_before=leave_finished_stamp,
                required_distinct_frames=PLACE_REQUIRED_FRAMES,
                gripper_command_id=last_gripper_command_id,
            )
        ):
            rospy.logerr("Placement validation barrier did not pass.")
            return False

        # --- Step 9: 返回观测位，不作为验证阶段，但仍支持中断 ---
        if not wait_if_paused_or_aborted():
            return False

        print("  Returning to ready pose...")

        if not arm_ready_pose():
            rospy.logerr("Failed to return to ready pose.")
            return False

        if not interruptible_sleep(4.0):
            return False

        print("Catch-and-place sequence completed successfully.")
        return True

    except Exception as e:
        rospy.logerr(f"Error during catch-and-place sequence: {e}")

        try:
            arm_ready_pose()
        except Exception:
            pass

        return False


# ============================================================
# 其他保留函数
# ============================================================

def navigateToGoal(x, y, orientation_z, orientation_w):
    ac = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    ac.wait_for_server(rospy.Duration(5.0))

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.orientation.z = orientation_z
    goal.target_pose.pose.orientation.w = orientation_w

    ac.send_goal(goal)
    ac.wait_for_result()

    state = ac.get_state()
    if state == GoalStatus.SUCCEEDED:
        rospy.loginfo("Successfully reached the goal location")
    else:
        rospy.loginfo("Failed to reach the goal location")


def gripper_open():
    global last_gripper_command_id, last_gripper_command_stamp
    global current_task, latest_runtime_action_type
    try:
        set_pub = rospy.Publisher("/rm_driver/Gripper_Set", Gripper_Set, queue_size=1)
        rospy.sleep(1)

        set_cmd = Gripper_Set()
        set_cmd.position = 1000

        actuation_seq = _mark_actuation()
        last_gripper_command_id, last_gripper_command_stamp = publish_gripper_command(
            "open",
            source="catch",
            context=_gripper_command_context(actuation_seq),
            requested_position=set_cmd.position,
            topic=GRIPPER_COMMAND_TOPIC,
        )
        set_pub.publish(set_cmd)
        if current_task is not None:
            publish_runtime_context_for_task(
                current_task,
                action_type=str(latest_runtime_action_type or "open_gripper"),
                status="active",
            )
        print("Gripper opened.")
        return True

    except Exception as e:
        rospy.logerr(f"Error opening gripper: {e}")
        return False


def gripper_close():
    global last_gripper_command_id, last_gripper_command_stamp
    global current_task, latest_runtime_action_type
    try:
        pick_pub = rospy.Publisher("/rm_driver/Gripper_Pick_On", Gripper_Pick, queue_size=1)
        rospy.sleep(1)

        pick_cmd = Gripper_Pick()
        pick_cmd.speed = 200
        pick_cmd.force = 1000

        actuation_seq = _mark_actuation()
        last_gripper_command_id, last_gripper_command_stamp = publish_gripper_command(
            "close",
            source="catch",
            context=_gripper_command_context(actuation_seq),
            requested_speed=pick_cmd.speed,
            requested_force=pick_cmd.force,
            topic=GRIPPER_COMMAND_TOPIC,
        )
        pick_pub.publish(pick_cmd)
        if current_task is not None:
            publish_runtime_context_for_task(
                current_task,
                action_type=str(latest_runtime_action_type or "close_gripper"),
                status="active",
            )
        print("Gripper closed.")
        return True

    except Exception as e:
        rospy.logerr(f"Error closing gripper: {e}")
        return False


def _make_cleanup_orientation_msg(object_class, source_base_position):
    """
    构造一个最小 arm_orientation_msg，兼容 catch_and_place() 内部可能访问的字段。

    source_base_position 已经是 base 坐标，不会再被 convert。
    """
    p = list(source_base_position or [0.0, 0.0, 0.0])

    return SimpleNamespace(
        object_class=str(object_class or ""),
        x=float(p[0]),
        y=float(p[1]),
        z=float(p[2]),

        # 兼容可能存在的角度字段
        angle=0.0,
        obj_angle=0.0,
        yaw=0.0,
        theta=0.0,
        rotation_z=0.0,
    )


def cleanup_pick_place_by_base_pose(
    request_id,
    object_class,
    source_base_position,
    target_base_position,
):
    """
    清理专用 pick-place。

    设计原则：
    - 复用 catch_and_place() 的成熟抓放动作；
    - source_base_position 已经是机械臂 base 坐标，不再 convert；
    - 不修改 current_task；
    - 不 pop task_queue；
    - 不设置 tasks_completed；
    - 不触发 resume_from_step；
    - 清理成功/失败通过 /task_execution/cleanup_result 返回。
    """
    global cleanup_active, is_executing, accept_yolo_detection

    cleanup_active = True
    is_executing = True
    accept_yolo_detection = False

    try:
        if source_base_position is None or len(source_base_position) < 3:
            raise ValueError("invalid source_base_position=%s" % str(source_base_position))

        if target_base_position is None or len(target_base_position) < 3:
            raise ValueError("invalid target_base_position=%s" % str(target_base_position))

        obj_pose_in_base = [
            float(source_base_position[0]),
            float(source_base_position[1]),
            float(source_base_position[2]),
        ]

        target_place_pose = [
            float(target_base_position[0]),
            float(target_base_position[1]),
            float(target_base_position[2]),
        ]

        object_class = str(object_class or "").strip()

        rospy.logwarn(
            "[EXEC][CLEANUP] start cleanup_pick_place request_id=%s class=%s source=%s target=%s",
            str(request_id),
            object_class,
            str([round(v, 4) for v in obj_pose_in_base]),
            str([round(v, 4) for v in target_place_pose]),
        )

        publish_cleanup_runtime_context(
            request_id=request_id,
            object_class=object_class,
            source_base_position=obj_pose_in_base,
            target_base_position=target_place_pose,
            action_type="cleanup_pick_place_start",
        )

        cleanup_task_info = {
            "step": 0,
            "target_object_id": "cleanup_" + object_class,
            "target_class": object_class,
            "target_place": target_place_pose,
            "target_place_pose": target_place_pose,
            "comment": "cleanup pick-place",
            "is_cleanup": True,
        }

        arm_orientation_msg = _make_cleanup_orientation_msg(
            object_class=object_class,
            source_base_position=obj_pose_in_base,
        )

        # 关键：复用执行端 catch_and_place，不走 cleanup_executor 独立轨迹。
        try:
            ok = catch_and_place(
                obj_pose_in_base=obj_pose_in_base,
                arm_orientation_msg=arm_orientation_msg,
                target_place_pose=target_place_pose,
                obj_angle=0.0,
                current_task_info=cleanup_task_info,
            )
        except TypeError:
            # 兼容 catch_and_place 不支持关键字参数的版本。
            ok = catch_and_place(
                obj_pose_in_base,
                arm_orientation_msg,
                target_place_pose,
                0.0,
                cleanup_task_info,
            )

        publish_cleanup_runtime_context(
            request_id=request_id,
            object_class=object_class,
            source_base_position=obj_pose_in_base,
            target_base_position=target_place_pose,
            action_type="cleanup_pick_place_done" if ok else "cleanup_pick_place_failed",
        )

        publish_cleanup_result(
            request_id=request_id,
            success=bool(ok),
            reason="cleanup_pick_place_done" if ok else "cleanup_pick_place_failed",
            detail={
                "object_class": object_class,
                "source_base_position": obj_pose_in_base,
                "target_base_position": target_place_pose,
            },
        )

        return bool(ok)

    except Exception as exc:
        rospy.logerr("[EXEC][CLEANUP] cleanup_pick_place exception: %s", str(exc))

        publish_cleanup_result(
            request_id=request_id,
            success=False,
            reason="exception",
            detail={
                "object_class": str(object_class or ""),
                "source_base_position": list(source_base_position or []),
                "target_base_position": list(target_base_position or []),
                "error": str(exc),
            },
        )

        return False

    finally:
        is_executing = False
        cleanup_active = False


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    rospy.init_node("object_catch", anonymous=True)

    execution_run_id = "run_%s" % uuid.uuid4().hex[:12]
    execution_attempt_id = 1
    execution_stage_seq = 0
    execution_actuation_seq = 0
    validation_blocked_status = ""

    BARRIER_MODE = str(rospy.get_param("~barrier_mode", "enforce")).strip().lower()
    if BARRIER_MODE not in {"off", "shadow", "enforce"}:
        rospy.logfatal("Invalid ~barrier_mode=%s", BARRIER_MODE)
        sys.exit(1)
    VALIDATION_REQUEST_RETRY_SEC = max(
        0.1,
        float(rospy.get_param("~validation_request_retry_sec", 0.5)),
    )
    VALIDATION_TIMEOUT_SEC = max(
        0.5,
        float(rospy.get_param("~validation_timeout_sec", 6.0)),
    )
    GRASP_BARRIER_ENABLED = bool(rospy.get_param("~grasp_barrier_enabled", True))
    PLACE_BARRIER_ENABLED = bool(rospy.get_param("~place_barrier_enabled", True))
    GRASP_REQUIRED_FRAMES = max(1, int(rospy.get_param("~grasp_required_frames", 1)))
    PLACE_REQUIRED_FRAMES = max(1, int(rospy.get_param("~place_required_frames", 2)))

    REQUIRE_GRIPPER_MONITOR = bool(rospy.get_param("~require_gripper_monitor", True))
    GRIPPER_MONITOR_STARTUP_TIMEOUT_SEC = max(
        0.5, float(rospy.get_param("~gripper_monitor_startup_timeout_sec", 5.0))
    )
    GRIPPER_STATE_MAX_AGE_SEC = max(
        0.1, float(rospy.get_param("~gripper_state_max_age_sec", 0.5))
    )
    INITIALIZE_GRIPPER_ON_STARTUP = bool(
        rospy.get_param("~initialize_gripper_on_startup", True)
    )
    GRIPPER_STARTUP_VALIDATION_TIMEOUT_SEC = max(
        0.5,
        float(rospy.get_param("~gripper_startup_validation_timeout_sec", 8.0)),
    )
    GRIPPER_STARTUP_MIN_STABLE_COUNT = max(
        1, int(rospy.get_param("~gripper_startup_min_stable_count", 3))
    )
    GRIPPER_COMMAND_TOPIC = str(
        rospy.get_param("~gripper_command_topic", TOPIC_GRIPPER_COMMAND)
    ).strip()
    GRIPPER_STATE_TOPIC = str(
        rospy.get_param("~gripper_state_topic", TOPIC_GRIPPER_STATE)
    ).strip()

    ENABLE_CLEANUP_CONTROL = bool(rospy.get_param("~enable_cleanup_control", False))

    current_stage_topic = str(rospy.get_param("~current_stage_topic", DEFAULT_CURRENT_STAGE_TOPIC))
    status_topic = str(rospy.get_param("~status_topic", DEFAULT_STATUS_TOPIC))
    control_topic = str(rospy.get_param("~control_topic", DEFAULT_CONTROL_TOPIC))
    runtime_context_topic = str(rospy.get_param("~runtime_context_topic", DEFAULT_RUNTIME_CONTEXT_TOPIC))
    recovery_ready_topic = str(rospy.get_param("~recovery_ready_topic", DEFAULT_RECOVERY_READY_TOPIC))
    validation_request_topic = str(
        rospy.get_param("~validation_request_topic", TOPIC_VALIDATION_REQUEST)
    )
    validation_result_topic = str(
        rospy.get_param("~validation_result_topic", TOPIC_VALIDATION_RESULT)
    )

    current_stage_pub = rospy.Publisher(current_stage_topic, String, queue_size=10)
    status_pub = rospy.Publisher(status_topic, String, queue_size=20)
    runtime_context_pub = rospy.Publisher(runtime_context_topic, String, queue_size=20)
    validation_request_pub = rospy.Publisher(
        validation_request_topic,
        String,
        queue_size=10,
        latch=False,
    )
    validation_result_sub = rospy.Subscriber(
        validation_result_topic,
        String,
        validation_result_callback,
        queue_size=10,
    )
    # latch=True: 避免 rospy.wait_for_message 临时订阅者错过消息。
    # 配合两阶段握手协议（ready=false ack → ready=true handoff）防止 latch 残留误触发。
    recovery_ready_pub = rospy.Publisher(recovery_ready_topic, String, queue_size=10, latch=True)

    cleanup_result_pub = rospy.Publisher(
        "/task_execution/cleanup_result",
        String,
        queue_size=10,
        latch=True,
    )

    rospy.Subscriber(control_topic, String, control_callback, queue_size=10)

    rospy.loginfo("Execution stage topic: %s", current_stage_topic)
    rospy.loginfo("Execution status topic: %s", status_topic)
    rospy.loginfo("Execution control topic: %s", control_topic)
    rospy.loginfo("Recovery runtime context topic: %s", runtime_context_topic)
    rospy.loginfo("Execution recovery ready topic: %s", recovery_ready_topic)

    pub_arm_pose = rospy.Publisher("/rm_driver/GetCurrentArmState", Empty, queue_size=1)

    if REQUIRE_GRIPPER_MONITOR:
        # A cold RealMan gripper can publish an all-zero invalid SDK state
        # until its first control command.  At this point require only a
        # current monitor sample, not valid=True, to avoid a startup deadlock.
        state_msg, state_reason = wait_for_gripper_monitor_sample(
            timeout_sec=GRIPPER_MONITOR_STARTUP_TIMEOUT_SEC,
            max_age_sec=GRIPPER_STATE_MAX_AGE_SEC,
            topic=GRIPPER_STATE_TOPIC,
        )
        if state_msg is None:
            rospy.logfatal(
                "Required gripper state monitor is not publishing a current sample: "
                "reason=%s topic=%s",
                state_reason,
                GRIPPER_STATE_TOPIC,
            )
            sys.exit(1)
        rospy.loginfo(
            "Gripper state monitor online: instance=%s sample_seq=%s valid=%s "
            "states=%s reason=%s",
            state_msg.monitor_instance_id,
            str(state_msg.sample_seq),
            str(bool(state_msg.valid)),
            str(list(state_msg.states)),
            str(state_msg.reason or ""),
        )

    if INITIALIZE_GRIPPER_ON_STARTUP:
        if not gripper_open():
            rospy.logfatal("Failed to send the synchronized startup gripper-open command.")
            sys.exit(1)

        if REQUIRE_GRIPPER_MONITOR:
            state_msg, state_reason = wait_for_gripper_startup_state(
                timeout_sec=GRIPPER_STARTUP_VALIDATION_TIMEOUT_SEC,
                max_age_sec=GRIPPER_STATE_MAX_AGE_SEC,
                required_state="open",
                min_stable_count=GRIPPER_STARTUP_MIN_STABLE_COUNT,
                command_id=last_gripper_command_id,
                not_before=last_gripper_command_stamp,
                topic=GRIPPER_STATE_TOPIC,
            )
            if state_msg is None:
                rospy.logfatal(
                    "Startup gripper-open command was not confirmed: reason=%s "
                    "command_id=%s topic=%s",
                    state_reason,
                    last_gripper_command_id,
                    GRIPPER_STATE_TOPIC,
                )
                sys.exit(1)
            rospy.loginfo(
                "Startup gripper-open confirmed: command_id=%s sample_seq=%s "
                "stable_count=%s states=%s",
                last_gripper_command_id,
                str(state_msg.sample_seq),
                str(state_msg.stable_count),
                str(list(state_msg.states)),
            )
    else:
        if not REQUIRE_GRIPPER_MONITOR:
            rospy.logfatal(
                "Invalid startup configuration: initialize_gripper_on_startup=false "
                "requires require_gripper_monitor=true"
            )
            sys.exit(1)
        state_msg, state_reason = wait_for_gripper_startup_state(
            timeout_sec=GRIPPER_STARTUP_VALIDATION_TIMEOUT_SEC,
            max_age_sec=GRIPPER_STATE_MAX_AGE_SEC,
            required_state="open",
            min_stable_count=GRIPPER_STARTUP_MIN_STABLE_COUNT,
            topic=GRIPPER_STATE_TOPIC,
        )
        if state_msg is None:
            rospy.logfatal(
                "Startup initialization is disabled, but no stable open gripper "
                "state was observed: reason=%s topic=%s",
                state_reason,
                GRIPPER_STATE_TOPIC,
            )
            sys.exit(1)
        rospy.loginfo(
            "Startup gripper already open; initialization command skipped: "
            "sample_seq=%s stable_count=%s",
            str(state_msg.sample_seq),
            str(state_msg.stable_count),
        )

    # 1. 发送观测位姿指令
    arm_ready_pose()

    # 2. 阻塞等待机械臂实际到达观测位
    if not wait_for_arm_ready_pose(timeout=20.0):
        rospy.logerr("Failed to reach ready pose within timeout. Exiting.")
        sys.exit(1)

    # 3. 加载任务队列
    task_queue = load_task_queue_from_action()

    if task_queue:
        current_task = task_queue.pop(0)
        first_label = current_task.get("target_object_id") or current_task.get("target_class")
        print(
            f"Setting first task: Target {first_label}, "
            f"Class {current_task['target_class']}, Place {current_task['target_place']}"
        )

        publish_runtime_context_for_task(
            current_task,
            action_type="waiting_detection",
            status="active",
        )
    else:
        print("Task queue is empty!")
        tasks_completed = True

    # 4. Register the capture-stamped wrist detection as the authoritative
    # execution input. The legacy ObjectInfo path is optional because it has
    # no source timestamp and cannot provide temporal guarantees.
    sub_detected_objects = rospy.Subscriber(
        "/wrist/detected_objects",
        DetectedObjects,
        stamped_detections_callback,
        queue_size=1,
    )
    sub_object_pose = None
    if bool(rospy.get_param("~enable_legacy_object_pose_subscription", False)):
        sub_object_pose = rospy.Subscriber(
            "/object_pose",
            ObjectInfo,
            object_pose_callback,
            queue_size=1,
        )

    rospy.loginfo("YOLO topic subscription registered. Detection processing is currently disabled.")

    # 5. 第一次任务前，也统一走"观测准备流程"
    if not tasks_completed:
        if not prepare_yolo_observation_for_next_task(timeout=20.0):
            rospy.logerr("Failed to prepare YOLO observation for first task. Exiting.")
            sys.exit(1)

    rospy.spin()
