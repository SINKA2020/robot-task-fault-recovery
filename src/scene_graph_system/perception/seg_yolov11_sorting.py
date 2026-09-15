#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Sorting-specific external-camera YOLO segmentation node.

This node reuses the existing ``seg_yolov11.GlobalSegYoloV11`` camera,
segmentation, depth and coordinate-transform pipeline.  Its only task-specific
addition is image-space classification against multiple sorting-region ROIs.

Only one external-camera detector should run at a time.  The building detector
publishes ``/global_camera/seg_detections`` while this node defaults to the
separate ``/global_camera/sorting_detections`` topic.
"""

from __future__ import annotations

from scene_graph_system.resources import package_resource_path

import copy
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rospy
import yaml

from scene_graph_system.perception.seg_yolov11 import (
    GlobalSegYoloV11,
    classify_mask_build_region,
    make_roi_mask,
    scale_roi_polygon_if_needed,
)


DEFAULT_SORTING_REGIONS_YAML = (
    package_resource_path('config/global_sorting_regions.yaml')
)
DEFAULT_SORTING_OUTPUT_TOPIC = "/global_camera/sorting_detections"

SORTING_OUTSIDE_STATE = "outside_sort_regions"
SORTING_REGION_STATE_PREFIX = "in_sort_region_"

REGION_COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
]


def load_sorting_regions(path: str) -> Dict[str, Any]:
    """Load and validate a multi-region image-space sorting ROI file."""
    if not path:
        raise ValueError("sorting regions yaml path is empty")
    if not os.path.exists(path):
        raise FileNotFoundError("sorting regions yaml not found: %s" % path)

    with open(path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    if not isinstance(data, dict):
        raise ValueError("sorting regions yaml root must be a mapping")

    raw_regions = data.get("regions", [])
    if not isinstance(raw_regions, list) or not raw_regions:
        raise ValueError("sorting regions yaml requires a non-empty regions list")

    image_width = int(data.get("image_width", 0) or 0)
    image_height = int(data.get("image_height", 0) or 0)
    regions: List[Dict[str, Any]] = []
    names = set()
    states = set()

    for index, raw in enumerate(raw_regions):
        if not isinstance(raw, dict):
            raise ValueError("regions[%d] must be a mapping" % index)

        region_name = str(raw.get("region_name", "") or "").strip()
        state = str(raw.get("state", "") or "").strip()
        polygon = raw.get("polygon")

        if not region_name:
            raise ValueError("regions[%d].region_name is required" % index)
        if region_name in names:
            raise ValueError("duplicate sorting region_name: %s" % region_name)
        if not state.startswith(SORTING_REGION_STATE_PREFIX):
            raise ValueError(
                "regions[%d].state must start with %s"
                % (index, SORTING_REGION_STATE_PREFIX)
            )
        if state in states:
            raise ValueError("duplicate sorting region state: %s" % state)
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError("regions[%d].polygon needs at least 3 points" % index)

        polygon_np = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if not np.all(np.isfinite(polygon_np)):
            raise ValueError("regions[%d].polygon contains non-finite values" % index)

        min_mask_overlap = float(raw.get("min_mask_overlap", 0.20))
        if not 0.0 <= min_mask_overlap <= 1.0:
            raise ValueError(
                "regions[%d].min_mask_overlap must be within [0, 1]" % index
            )
        if abs(float(cv2.contourArea(polygon_np))) < 1.0:
            raise ValueError("regions[%d].polygon area is zero" % index)

        names.add(region_name)
        states.add(state)
        regions.append(
            {
                "region_name": region_name,
                "state": state,
                "polygon": polygon_np,
                "min_mask_overlap": min_mask_overlap,
                "min_center_in_roi": bool(raw.get("min_center_in_roi", True)),
                "color": REGION_COLORS[index % len(REGION_COLORS)],
            }
        )

    return {
        "image_width": image_width,
        "image_height": image_height,
        "regions": regions,
        "yaml_path": path,
    }


class SortingSegYoloV11(GlobalSegYoloV11):
    """Global segmentation detector with mutually-exclusive sorting ROIs."""

    def __init__(self):
        self.sorting_regions_cfg: Optional[Dict[str, Any]] = None
        self.sorting_region_runtime: List[Dict[str, Any]] = []
        self.sorting_region_image_size: Optional[Tuple[int, int]] = None
        super().__init__()

        configured_regions = (
            list(self.sorting_regions_cfg.get("regions", []) or [])
            if self.sorting_regions_cfg is not None
            else []
        )
        rospy.loginfo(
            "seg_yolov11_sorting initialized: topic=%s regions=%d yaml=%s",
            self.output_topic,
            len(self.sorting_region_runtime or configured_regions),
            self.sorting_regions_path,
        )

    def _load_params(self):
        super()._load_params()
        self.output_topic = str(
            rospy.get_param("~output_topic", DEFAULT_SORTING_OUTPUT_TOPIC)
        ).strip()
        self.enable_sorting_regions = bool(
            rospy.get_param("~enable_sorting_regions", True)
        )
        self.sorting_regions_path = str(
            rospy.get_param("~sorting_regions_path", DEFAULT_SORTING_REGIONS_YAML)
        ).strip()
        self.draw_sorting_regions = bool(
            rospy.get_param("~draw_sorting_regions", True)
        )

        # The sorting ROIs themselves form the legacy umbrella build region.
        # Do not load the block-building polygon unless explicitly requested.
        self.enable_build_region_roi = bool(
            rospy.get_param("~enable_build_region_roi", False)
        )

    def _load_build_region_roi(self):
        super()._load_build_region_roi()
        self._load_sorting_regions()

    def _load_sorting_regions(self):
        if not self.enable_sorting_regions:
            rospy.logwarn("Sorting region classification disabled.")
            self.sorting_regions_cfg = None
            return

        cfg = load_sorting_regions(self.sorting_regions_path)
        self.sorting_regions_cfg = cfg
        rospy.loginfo(
            "Loaded sorting regions from %s: image=%dx%d regions=%s",
            cfg["yaml_path"],
            int(cfg["image_width"]),
            int(cfg["image_height"]),
            [region["region_name"] for region in cfg["regions"]],
        )

    def _ensure_build_region_mask(self, color_image: np.ndarray):
        super()._ensure_build_region_mask(color_image)
        self._ensure_sorting_region_masks(color_image)

    def _ensure_sorting_region_masks(self, color_image: np.ndarray):
        if self.sorting_regions_cfg is None:
            self.sorting_region_runtime = []
            self.sorting_region_image_size = None
            return

        height, width = color_image.shape[:2]
        image_size = (int(width), int(height))
        if self.sorting_region_runtime and self.sorting_region_image_size == image_size:
            return

        runtime_regions: List[Dict[str, Any]] = []
        yaml_width = int(self.sorting_regions_cfg.get("image_width", 0) or 0)
        yaml_height = int(self.sorting_regions_cfg.get("image_height", 0) or 0)

        for region in self.sorting_regions_cfg["regions"]:
            polygon = scale_roi_polygon_if_needed(
                polygon=region["polygon"],
                yaml_width=yaml_width,
                yaml_height=yaml_height,
                current_width=width,
                current_height=height,
            )
            runtime = copy.deepcopy(region)
            runtime["polygon"] = polygon
            runtime["roi_mask"] = make_roi_mask(color_image.shape, polygon)
            runtime_regions.append(runtime)

        self.sorting_region_runtime = runtime_regions
        self.sorting_region_image_size = image_size
        rospy.loginfo(
            "Sorting ROI masks ready: image=%dx%d regions=%s",
            width,
            height,
            {
                region["region_name"]: region["polygon"].astype(int).tolist()
                for region in runtime_regions
            },
        )

    def _classify_sorting_mask(self, mask: np.ndarray) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        for region in self.sorting_region_runtime:
            info = classify_mask_build_region(
                mask=mask,
                roi_mask=region["roi_mask"],
                min_mask_overlap=float(region["min_mask_overlap"]),
                min_center_in_roi=bool(region["min_center_in_roi"]),
            )
            results.append(
                {
                    "region_name": region["region_name"],
                    "state": region["state"],
                    "matched": bool(info["in_build_region"]),
                    "overlap_ratio": float(info["overlap_ratio"]),
                    "center_in_roi": bool(info["center_in_roi"]),
                    "mask_center_px": info["mask_center_px"],
                }
            )

        matched = [item for item in results if item["matched"]]
        selected = None
        if matched:
            selected = max(
                matched,
                key=lambda item: (
                    float(item["overlap_ratio"]),
                    int(bool(item["center_in_roi"])),
                ),
            )

        return {
            "selected": selected,
            "outside": selected is None,
            "results": results,
        }

    def _extract_segmentation_instances(
        self,
        result0,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        depth_intrin,
    ) -> List[Dict[str, Any]]:
        detections = super()._extract_segmentation_instances(
            result0=result0,
            color_image=color_image,
            depth_image=depth_image,
            depth_intrin=depth_intrin,
        )

        if not self.enable_sorting_regions:
            return detections
        if not self.sorting_region_runtime:
            rospy.logerr_throttle(
                2.0,
                "Sorting regions are enabled but no runtime ROI masks are available.",
            )
            return detections

        masks_obj = getattr(result0, "masks", None)
        if masks_obj is None:
            return detections

        masks = masks_obj.data.cpu().numpy()
        height, width = color_image.shape[:2]

        for det in detections:
            if str(det.get("class_name", "")).strip().lower() == "gripper":
                continue

            mask_index = int(det.get("mask_index", -1))
            if mask_index < 0 or mask_index >= len(masks):
                continue

            mask = masks[mask_index]
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
            classification = self._classify_sorting_mask(mask > 0.5)
            selected = classification["selected"]

            states = [
                str(state)
                for state in list(det.get("states", []) or [])
                if not str(state).startswith(SORTING_REGION_STATE_PREFIX)
                and str(state) != SORTING_OUTSIDE_STATE
                and str(state) not in {"in_build_region", "outside_build_region"}
            ]

            if selected is None:
                region_name = "none"
                region_state = SORTING_OUTSIDE_STATE
                overlap_ratio = 0.0
                center_in_roi = False
                center_px = (
                    classification["results"][0]["mask_center_px"]
                    if classification["results"]
                    else None
                )
                states.extend(["outside_build_region", SORTING_OUTSIDE_STATE])
            else:
                region_name = str(selected["region_name"])
                region_state = str(selected["state"])
                overlap_ratio = float(selected["overlap_ratio"])
                center_in_roi = bool(selected["center_in_roi"])
                center_px = selected["mask_center_px"]
                states.extend(["in_build_region", region_state])

            overlaps = {
                item["region_name"]: float(item["overlap_ratio"])
                for item in classification["results"]
            }
            details = {
                item["region_name"]: {
                    "state": item["state"],
                    "matched": bool(item["matched"]),
                    "overlap_ratio": float(item["overlap_ratio"]),
                    "center_in_roi": bool(item["center_in_roi"]),
                }
                for item in classification["results"]
            }

            det["states"] = list(dict.fromkeys(states))
            det["sorting_region"] = region_name
            det["sorting_region_state"] = region_state
            det["sorting_region_overlap"] = overlap_ratio
            det["sorting_region_overlaps"] = overlaps

            # Preserve the existing umbrella fields for current graph filters.
            det["in_build_region"] = selected is not None
            det["build_region_overlap"] = overlap_ratio
            det["center_in_build_region"] = center_in_roi
            det["build_region_center_px"] = center_px
            det["region_name"] = region_name

            attributes = dict(det.get("attributes", {}) or {})
            attributes["source"] = "seg_yolov11_sorting"
            attributes["detector_type"] = "yolo_segmentation_sorting"
            attributes["sorting_region"] = {
                "enabled": bool(self.enable_sorting_regions),
                "region_name": region_name,
                "state": region_state,
                "overlap_ratio": overlap_ratio,
                "overlaps": overlaps,
                "details": details,
                "mask_center_px": center_px,
            }
            det["attributes"] = attributes

        return detections

    def _make_payload(self, detections, **kwargs):
        payload = super()._make_payload(detections, **kwargs)
        payload["source"] = "seg_yolov11_sorting"
        payload["detector_type"] = "yolo_segmentation_sorting"
        payload["sorting_regions"] = {
            "enabled": bool(self.enable_sorting_regions),
            "yaml_path": self.sorting_regions_path,
            "regions": [
                {
                    "region_name": region["region_name"],
                    "state": region["state"],
                    "polygon": region["polygon"].astype(int).tolist(),
                    "min_mask_overlap": float(region["min_mask_overlap"]),
                    "min_center_in_roi": bool(region["min_center_in_roi"]),
                }
                for region in self.sorting_region_runtime
            ],
        }
        return payload

    def _show_debug_window(self, color_image, detections, result0=None):
        vis = color_image.copy()

        if self.draw_masks and result0 is not None and getattr(result0, "masks", None) is not None:
            try:
                masks = result0.masks.data.cpu().numpy()
                height, width = vis.shape[:2]
                overlay = vis.copy()
                for mask in masks:
                    if mask.shape[:2] != (height, width):
                        mask = cv2.resize(
                            mask.astype(np.float32),
                            (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    mask_bool = mask > 0.5
                    overlay[mask_bool] = (
                        0.5 * overlay[mask_bool] + 0.5 * np.array([0, 180, 0])
                    ).astype(np.uint8)
                vis = cv2.addWeighted(overlay, 0.55, vis, 0.45, 0.0)
            except Exception:
                pass

        colors_by_name = {}
        if self.draw_sorting_regions:
            for region in self.sorting_region_runtime:
                polygon = np.asarray(region["polygon"], dtype=np.int32)
                color = tuple(int(value) for value in region["color"])
                colors_by_name[region["region_name"]] = color
                cv2.polylines(vis, [polygon], isClosed=True, color=color, thickness=2)
                label_position = tuple(polygon[0].astype(int).tolist())
                cv2.putText(
                    vis,
                    str(region["region_name"]),
                    label_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    color,
                    2,
                )

        for det in detections:
            bbox = det.get("bbox2d")
            if bbox is None:
                continue
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
            region_name = str(det.get("sorting_region", "none") or "none")
            color = colors_by_name.get(region_name, (128, 128, 128))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            class_name = str(det.get("class_name", "object"))
            score = float(det.get("score", 0.0))
            overlap = float(det.get("sorting_region_overlap", 0.0))
            label = "%s %.2f %s ov=%.2f" % (
                class_name,
                score,
                region_name,
                overlap,
            )
            cv2.putText(
                vis,
                label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
            )

            center_px = det.get("build_region_center_px")
            if isinstance(center_px, (list, tuple)) and len(center_px) == 2:
                cv2.circle(
                    vis,
                    (int(center_px[0]), int(center_px[1])),
                    3,
                    color,
                    -1,
                )

        cv2.imshow("seg_yolov11_sorting", vis)
        cv2.waitKey(1)


def main():
    if not rospy.core.is_initialized():
        rospy.init_node("seg_yolov11_sorting", anonymous=False)
    SortingSegYoloV11()
    rospy.spin()


if __name__ == "__main__":
    main()
