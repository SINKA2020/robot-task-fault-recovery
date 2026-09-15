#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import json
import inspect
import time
from typing import Any, Dict, List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))



import numpy as np
import rospy

from scene_graph_system.robot.catch import CatchExecutor, convert, wait_for_current_arm_state
try:
    from scene_graph_system.robot.catch import load_grasp_check_config_from_ros_params
except ImportError:
    load_grasp_check_config_from_ros_params = None

from scene_graph_system.planning.generate_expected_state_ai import build_expected_state_for_execution_step
from scene_graph_system.planning.configurable_expected_state_builder import (
    build_configured_expected_state_for_execution_step,
    compare_configured_expected_state,
)
from scene_graph_system.scene_graph.object_id_utils import canonical_object_id, normalize_object_class_name
from scene_graph_system.validation.scene_graph_comparator import SceneGraphComparator
from scene_graph_system.scene_graph.scene_graph_builder import SceneGraphBuilder
from scene_graph_system.planning.task_plan_adapter import build_internal_plan, build_internal_plan_from_file, load_task_plan, summarize_internal_plan


DEFAULT_REALMAN_IP = '192.168.0.18'
DEFAULT_REALMAN_PORT = 8080
DEFAULT_REALMAN_THREAD_MODE = 'RM_TRIPLE_MODE_E'
DEFAULT_TASK_PROFILE = 'block_building'


def _supports_keyword_argument(callable_obj, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False

    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return keyword in signature.parameters


def load_grasp_check_config_compat(param_prefix: str = '~'):
    if load_grasp_check_config_from_ros_params is None:
        rospy.logwarn_once('catch.py does not expose load_grasp_check_config_from_ros_params; using default grasp check behavior.')
        return None
    return load_grasp_check_config_from_ros_params(param_prefix)


def create_realman_provider_from_runtime():
    rospy.logwarn_once(
        'Direct gripper SDK access is disabled; use gripper_state_monitor on /gripper/state.'
    )
    return None, None


def create_scene_graph_builder_compat(provider):
    builder_kwargs = {'init_node': False}
    if provider is not None and _supports_keyword_argument(SceneGraphBuilder.__init__, 'gripper_state_provider'):
        builder_kwargs['gripper_state_provider'] = provider
    return SceneGraphBuilder(**builder_kwargs)


def create_catch_executor_compat(provider, grasp_check_config):
    executor_kwargs = {'init_node': False}
    if provider is not None and _supports_keyword_argument(CatchExecutor.__init__, 'gripper_state_provider'):
        executor_kwargs['gripper_state_provider'] = provider
    if grasp_check_config is not None and _supports_keyword_argument(CatchExecutor.__init__, 'grasp_check_config'):
        executor_kwargs['grasp_check_config'] = grasp_check_config
    return CatchExecutor(**executor_kwargs)


def load_runtime_plan() -> Dict[str, Any]:
    default_pose_tolerance = float(rospy.get_param('~default_pose_tolerance', 0.05))
    task_plan_ref = str(rospy.get_param('~task_plan_ref', '')).strip()
    task_plan_root = str(rospy.get_param('~task_plan_root', '')).strip() or None
    if task_plan_ref:
        return load_task_plan(
            task_plan_ref,
            task_plan_root=task_plan_root,
            default_pose_tolerance=default_pose_tolerance,
        )

    plan_file = str(rospy.get_param('~plan_file', '')).strip()
    if plan_file:
        return build_internal_plan_from_file(plan_file, default_pose_tolerance=default_pose_tolerance)

    if rospy.has_param('~internal_plan'):
        return build_internal_plan(
            rospy.get_param('~internal_plan'),
            default_pose_tolerance=default_pose_tolerance,
        )

    if rospy.has_param('~action_plan'):
        return build_internal_plan(
            {'ACTION_PLAN': rospy.get_param('~action_plan')},
            default_pose_tolerance=default_pose_tolerance,
        )

    if rospy.has_param('~task_queue'):
        return build_internal_plan(
            {'task_queue': rospy.get_param('~task_queue')},
            default_pose_tolerance=default_pose_tolerance,
        )

    raise ValueError('Missing plan input. Provide ~plan_file, ~internal_plan, ~action_plan or ~task_queue.')


def wait_for_initial_scene_graph(builder, wait_timeout: float, poll_interval: float):
    deadline = time.time() + wait_timeout
    while not rospy.is_shutdown() and time.time() < deadline:
        if builder.has_latest_scene_graph():
            return builder.get_latest_scene_graph(), builder.latest_scene_graph_stamp
        time.sleep(poll_interval)
    return None, None


def scene_graph_signature(scene_graph) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    node_signature = tuple(
        sorted(
            (
                node.uid(),
                node.name,
                tuple(sorted(getattr(node, 'states', []) or [])),
            )
            for node in scene_graph.nodes
        )
    )
    relation_signature = tuple(
        sorted((edge.start.uid(), edge.end.uid(), edge.edge_type) for edge in scene_graph.get_relations())
    )
    return node_signature, relation_signature


def wait_for_scene_graph_stability(builder, previous_stamp, wait_timeout: float, poll_interval: float, stable_count: int):
    deadline = time.time() + wait_timeout
    update_seen = False
    last_signature = None
    stable_hits = 0
    latest_graph = None
    latest_stamp = previous_stamp

    while not rospy.is_shutdown() and time.time() < deadline:
        if builder.has_latest_scene_graph():
            current_stamp = builder.latest_scene_graph_stamp
            if current_stamp is None:
                time.sleep(poll_interval)
                continue

            if previous_stamp is not None and current_stamp == previous_stamp and not update_seen:
                time.sleep(poll_interval)
                continue

            latest_graph = builder.get_latest_scene_graph()
            latest_stamp = current_stamp
            signature = scene_graph_signature(latest_graph)

            if not update_seen:
                update_seen = True
                last_signature = signature
                stable_hits = 1
            else:
                if signature == last_signature:
                    stable_hits += 1
                else:
                    last_signature = signature
                    stable_hits = 1

            if stable_hits >= max(1, stable_count):
                return latest_graph, latest_stamp

        time.sleep(poll_interval)

    return latest_graph, latest_stamp


def find_node_by_object_id(scene_graph, object_id: Optional[str]):
    if not object_id:
        return None
    normalized = canonical_object_id(object_id)
    for node in scene_graph.nodes:
        if node.uid() == normalized:
            return node
    return None


def list_nodes_by_class(scene_graph, target_class: str):
    normalized_class = normalize_object_class_name(target_class)
    return [node for node in scene_graph.nodes if normalize_object_class_name(node.name) == normalized_class]


def convert_node_to_base_pose(node, arm_pose_msg):
    if node is None or node.pos3d_np is None:
        return None

    return convert(
        float(node.pos3d_np[0]),
        float(node.pos3d_np[1]),
        float(node.pos3d_np[2]),
        arm_pose_msg.Pose[0],
        arm_pose_msg.Pose[1],
        arm_pose_msg.Pose[2],
        arm_pose_msg.Pose[3],
        arm_pose_msg.Pose[4],
        arm_pose_msg.Pose[5],
    )


def resolve_source_node(scene_graph, step_dict, arm_pose_msg):
    explicit_object_id = step_dict.get('target_object_id')
    if explicit_object_id:
        node = find_node_by_object_id(scene_graph, explicit_object_id)
        if node is not None:
            return node, node.uid(), 'object_id'

    candidates = list_nodes_by_class(scene_graph, step_dict['target_class'])
    if not candidates:
        return None, explicit_object_id, 'missing'

    if len(candidates) == 1:
        return candidates[0], candidates[0].uid(), 'class_unique'

    source_pose = step_dict.get('source_pose')
    if source_pose is None:
        return None, explicit_object_id, 'ambiguous_class'

    source_xyz = np.asarray(source_pose[:3], dtype=float)
    best_candidate = None
    best_distance = None
    for candidate in candidates:
        candidate_pose = convert_node_to_base_pose(candidate, arm_pose_msg)
        if candidate_pose is None:
            continue
        distance = float(np.linalg.norm(candidate_pose[:3] - source_xyz))
        if best_distance is None or distance < best_distance:
            best_candidate = candidate
            best_distance = distance

    if best_candidate is None:
        return None, explicit_object_id, 'ambiguous_class'
    return best_candidate, best_candidate.uid(), 'class_nearest_source_pose'


def resolve_postcheck_node(scene_graph, step_dict, preferred_object_id: Optional[str], arm_pose_msg):
    preferred_node = find_node_by_object_id(scene_graph, preferred_object_id)
    if preferred_node is not None:
        base_pose = convert_node_to_base_pose(preferred_node, arm_pose_msg)
        return preferred_node, base_pose, 'preferred_object_id'

    candidates = list_nodes_by_class(scene_graph, step_dict['target_class'])
    if not candidates:
        return None, None, 'missing'

    target_place = np.asarray(step_dict['target_place'], dtype=float)
    best_candidate = None
    best_pose = None
    best_distance = None
    for candidate in candidates:
        candidate_pose = convert_node_to_base_pose(candidate, arm_pose_msg)
        if candidate_pose is None:
            continue
        distance = float(np.linalg.norm(candidate_pose[:3] - target_place))
        if best_distance is None or distance < best_distance:
            best_candidate = candidate
            best_pose = candidate_pose
            best_distance = distance

    if best_candidate is None:
        return None, None, 'missing'
    return best_candidate, best_pose, 'nearest_target_place'


def validate_target_place(step_dict, observed_node, observed_base_pose):
    if observed_node is None or observed_base_pose is None:
        return {
            'passed': False,
            'message': 'Target node not found in post-check scene graph.',
            'distance': None,
            'tolerance': float(step_dict.get('pose_tolerance', 0.05)),
        }

    target_place = np.asarray(step_dict['target_place'], dtype=float)
    distance = float(np.linalg.norm(observed_base_pose[:3] - target_place))
    tolerance = float(step_dict.get('pose_tolerance', 0.05))
    passed = distance <= tolerance
    return {
        'passed': passed,
        'message': (
            f"distance={distance:.4f}m <= tolerance={tolerance:.4f}m"
            if passed else
            f"distance={distance:.4f}m exceeds tolerance={tolerance:.4f}m"
        ),
        'distance': distance,
        'tolerance': tolerance,
        'observed_object_id': observed_node.uid(),
        'observed_class': observed_node.name,
        'observed_base_pose': [float(value) for value in observed_base_pose.tolist()],
    }


def infer_step_stage_name(step_dict) -> str:
    explicit = str(step_dict.get('stage_name', '') or '').strip()
    if explicit:
        return explicit
    try:
        step_num = int(step_dict.get('step', 1))
    except Exception:
        step_num = 1
    target_class = str(step_dict.get('target_class') or step_dict.get('class_name') or 'object').strip() or 'object'
    return f"step_{step_num}_catch_and_place_{target_class}"


def build_and_compare_post_expected_state(
    step_dict,
    expected_object_id,
    updated_graph,
    comparator,
    *,
    task_profile: str,
    task_profile_root: Optional[str],
    enable_configured_validation: bool,
):
    if enable_configured_validation:
        try:
            expected_state = build_configured_expected_state_for_execution_step(
                step_dict,
                resolved_object_id=expected_object_id,
                profile_name=task_profile or DEFAULT_TASK_PROFILE,
                profile_root=task_profile_root,
                phase='post',
            )
            comparison = compare_configured_expected_state(expected_state, updated_graph, comparator)
            return expected_state, comparison, 'configured'
        except Exception as exc:
            rospy.logwarn(
                'Configured post validation failed (%s); fallback to legacy expected state.',
                str(exc),
            )

    expected_state = build_expected_state_for_execution_step(step_dict, resolved_object_id=expected_object_id)
    comparison = comparator.compare(expected_state, updated_graph)
    return expected_state, comparison, 'legacy'


def execute_plan_step(
    builder,
    catch_executor,
    comparator,
    step_dict,
    wait_timeout,
    poll_interval,
    stable_count,
    *,
    task_profile: str = DEFAULT_TASK_PROFILE,
    task_profile_root: Optional[str] = None,
    enable_configured_validation: bool = True,
):
    latest_graph = builder.get_latest_scene_graph()
    if latest_graph is None:
        raise RuntimeError('No scene graph available before execution.')

    arm_pose_msg, arm_orientation_msg = wait_for_current_arm_state(timeout=10.0)
    source_node, resolved_object_id, source_resolution = resolve_source_node(latest_graph, step_dict, arm_pose_msg)
    pre_execution_stamp = builder.latest_scene_graph_stamp

    execution_mode = 'base_pose'
    if step_dict.get('source_pose') is not None:
        success = catch_executor.execute_one_step_from_base_pose(
            step_dict['source_pose'],
            step_dict['target_place'],
            arm_orientation_msg=arm_orientation_msg,
            step=step_dict.get('step'),
            stage_name=infer_step_stage_name(step_dict),
            target_object_id=step_dict.get('target_object_id'),
            target_class=step_dict.get('target_class'),
            action_type=step_dict.get('action_type', 'catch_and_place'),
        )
    else:
        raise RuntimeError(
            "纯执行版 catch.py 不再支持 scene graph 驱动抓取。"
            "请在 action.py 中为每一步提供 source_pose。"
        )

    updated_graph, updated_stamp = wait_for_scene_graph_stability(
        builder,
        pre_execution_stamp,
        wait_timeout,
        poll_interval,
        stable_count,
    )
    if updated_graph is None:
        raise RuntimeError('No updated scene graph received after execution.')

    post_arm_pose_msg, _ = wait_for_current_arm_state(timeout=10.0)
    observed_node, observed_base_pose, postcheck_resolution = resolve_postcheck_node(
        updated_graph,
        step_dict,
        preferred_object_id=resolved_object_id,
        arm_pose_msg=post_arm_pose_msg,
    )

    expected_object_id = resolved_object_id or (observed_node.uid() if observed_node is not None else None)
    expected_state, comparison, validation_mode = build_and_compare_post_expected_state(
        step_dict,
        expected_object_id,
        updated_graph,
        comparator,
        task_profile=task_profile,
        task_profile_root=task_profile_root,
        enable_configured_validation=enable_configured_validation,
    )
    pose_check = validate_target_place(step_dict, observed_node, observed_base_pose)
    passed = bool(success) and comparison.passed and pose_check['passed']

    return {
        'passed': passed,
        'execution_success': bool(success),
        'execution_mode': execution_mode,
        'source_resolution': source_resolution,
        'resolved_object_id': resolved_object_id,
        'postcheck_resolution': postcheck_resolution,
        'expected_stage_name': expected_state.stage_name,
        'validation_mode': validation_mode,
        'comparison': comparison.summary_dict(),
        'comparison_pretty': comparison.pretty_print(),
        'pose_check': pose_check,
        'post_execution_stamp': str(updated_stamp),
    }


def run_plan_closed_loop(plan_dict):
    wait_timeout = float(rospy.get_param('~wait_timeout', 15.0))
    poll_interval = float(rospy.get_param('~poll_interval', 0.2))
    stable_count = int(rospy.get_param('~stable_count', 2))
    retry_limit = int(rospy.get_param('~retry_limit', 1))
    task_profile = str(rospy.get_param('~task_profile', DEFAULT_TASK_PROFILE)).strip()
    task_profile_root = str(rospy.get_param('~task_profile_root', '')).strip() or None
    enable_configured_validation = bool(rospy.get_param('~enable_configured_validation', True))
    grasp_check_config = load_grasp_check_config_compat('~')

    provider, arm = create_realman_provider_from_runtime()
    rospy.logwarn("Legacy runner consumes the independent gripper state topic.")
    builder = create_scene_graph_builder_compat(provider)
    catch_executor = create_catch_executor_compat(provider, grasp_check_config)
    comparator = SceneGraphComparator()

    try:
        catch_executor.prepare()
        initial_graph, initial_stamp = wait_for_initial_scene_graph(builder, wait_timeout, poll_interval)
        if initial_graph is None:
            raise RuntimeError('No initial scene graph received before timeout.')

        rospy.loginfo('Loaded plan summary: %s', json.dumps(summarize_internal_plan(plan_dict), ensure_ascii=False))
        rospy.loginfo('Initial scene graph stamp: %s', str(initial_stamp))

        step_records = []
        all_passed = True

        for step_dict in plan_dict['execution_steps']:
            step_record = {
                'step': step_dict['step'],
                'stage_name': infer_step_stage_name(step_dict),
                'target_object_id': step_dict.get('target_object_id'),
                'target_class': step_dict['target_class'],
                'target_place': list(step_dict['target_place']),
                'attempts': [],
                'passed': False,
            }

            for attempt in range(retry_limit + 1):
                rospy.loginfo(
                    'Execute step=%s attempt=%s target_class=%s target_place=%s',
                    step_dict['step'],
                    attempt + 1,
                    step_dict['target_class'],
                    str(step_dict['target_place']),
                )
                try:
                    attempt_record = execute_plan_step(
                        builder,
                        catch_executor,
                        comparator,
                        step_dict,
                        wait_timeout,
                        poll_interval,
                        stable_count,
                        task_profile=task_profile or DEFAULT_TASK_PROFILE,
                        task_profile_root=task_profile_root,
                        enable_configured_validation=enable_configured_validation,
                    )
                except Exception as exc:
                    attempt_record = {
                        'passed': False,
                        'execution_success': False,
                        'error': str(exc),
                    }

                attempt_record['attempt'] = attempt + 1
                step_record['attempts'].append(attempt_record)

                if attempt_record.get('passed'):
                    step_record['passed'] = True
                    break

                if attempt < retry_limit:
                    rospy.logwarn(
                        'Step=%s failed on attempt=%s, preparing retry.',
                        step_dict['step'],
                        attempt + 1,
                    )
                    catch_executor.prepare()

            step_records.append(step_record)
            if not step_record['passed']:
                all_passed = False
                break

        summary = {
            'task_goal': plan_dict.get('task_goal', ''),
            'plan_metadata': dict(plan_dict.get('plan_metadata', {})),
            'all_passed': all_passed,
            'step_records': step_records,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if all_passed else 1
    finally:
        # SDK connection ownership belongs to gripper_state_monitor.
        pass


def main():
    rospy.init_node('run_task_plan_closed_loop', anonymous=True)
    plan_dict = load_runtime_plan()
    return run_plan_closed_loop(plan_dict)


if __name__ == '__main__':
    sys.exit(main())
