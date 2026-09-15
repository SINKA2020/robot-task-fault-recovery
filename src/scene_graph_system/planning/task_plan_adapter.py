#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compatibility wrapper for task-plan loading.

Historically the rest of the closed-loop system imported task-plan helpers from
``task_plan_adapter.py``.  The implementation now lives in
``task_plan_loader.py`` so it can also serve the profile UI and dry-run compiler
without importing ROS or robot recovery code.
"""

from __future__ import annotations

from scene_graph_system.planning.task_plan_loader import (  # noqa: F401
    build_internal_plan,
    build_internal_plan_from_file,
    get_default_task_plan_root,
    list_task_plans,
    load_task_plan,
    resolve_task_plan_path,
    summarize_internal_plan,
)


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Load and summarize an action/task plan.")
    parser.add_argument("plan_file", nargs="?", default="")
    parser.add_argument("--task-plan", default="")
    parser.add_argument("--task-plan-root", default=None)
    args = parser.parse_args()

    if args.task_plan:
        plan = load_task_plan(args.task_plan, task_plan_root=args.task_plan_root)
    elif args.plan_file:
        plan = build_internal_plan_from_file(args.plan_file)
    else:
        parser.error("provide a plan_file or --task-plan")

    print(json.dumps(summarize_internal_plan(plan), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
