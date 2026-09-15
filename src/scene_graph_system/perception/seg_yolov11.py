#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from scene_graph_system.resources import package_resource_path

import json
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
import rospy
import yaml
from std_msgs.msg import String
from ultralytics import YOLO


# ============================================================
# Paths / defaults
# ============================================================

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

from scene_graph_system.validation.validation_transaction import (
    CaptureInterval,
    TOPIC_OBSERVATION_REQUEST,
    ValidationRequest,
    processing_cycle_due,
)


DEFAULT_GLOBAL_CAMERA_SERIAL = "313522071466"
DEFAULT_GLOBAL_MODEL_PATH = package_resource_path('scripts/global_seg_best.pt')
DEFAULT_TABLE_TRANSFORM_YAML = package_resource_path('config/global_table_frame.yaml')
DEFAULT_BUILD_REGION_ROI_YAML = package_resource_path('config/global_build_region_roi.yaml')
DEFAULT_OUTPUT_TOPIC = "/global_camera/seg_detections"


# ============================================================
# Helper functions
# ============================================================

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def normalize_class_name(name: Any) -> str:
    text = str(name).strip().lower()
    text = text.replace(" ", "_").replace("-", "_")

    aliases = {
        "rect": "cuboid",
        "rectangle": "cuboid",
        "rectangular": "cuboid",
        "rectangular_prism": "cuboid",
        "block": "cuboid",
        "box": "cuboid",
        "tri": "triangle",
        "triangular": "triangle",
        "arch_block": "arch",
        "gripper_hand": "gripper",
        "hand": "gripper",
        "clipper": "gripper",
    }
    return aliases.get(text, text)


def load_matrix4x4_from_yaml(path: str, key: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError("%s not found: %s" % (key, path))

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if key not in data:
        raise KeyError("%s not found in %s" % (key, path))

    T = np.asarray(data[key], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("%s must be 4x4, got %s" % (key, T.shape))

    return T


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    homo = np.ones((points.shape[0], 4), dtype=np.float64)
    homo[:, :3] = points

    out = (T @ homo.T).T[:, :3]
    return out


def compute_aabb_corners(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    mn = np.min(points, axis=0)
    mx = np.max(points, axis=0)

    corners = np.array(
        [
            [mn[0], mn[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mx[1], mx[2]],
            [mx[0], mn[1], mn[2]],
            [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mn[2]],
            [mx[0], mx[1], mx[2]],
        ],
        dtype=np.float64,
    )
    return corners


def maybe_downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    points = np.asarray(points)
    max_points = int(max_points)

    if max_points <= 0 or len(points) <= max_points:
        return points

    idx = np.random.choice(len(points), size=max_points, replace=False)
    return points[idx]


# ============================================================
# Build-region ROI helpers
# ============================================================

def load_build_region_roi(path: str) -> Optional[Dict[str, Any]]:
    if not path:
        return None

    if not os.path.exists(path):
        rospy.logwarn("Build region ROI yaml not found: %s", path)
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Invalid build region ROI yaml: root must be dict")

    polygon = data.get("polygon", None)
    if polygon is None or len(polygon) < 3:
        raise ValueError("Build region ROI needs at least 3 polygon points")

    polygon_np = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)

    cfg = {
        "image_width": int(data.get("image_width", 0) or 0),
        "image_height": int(data.get("image_height", 0) or 0),
        "region_name": str(data.get("region_name", "build_area")),
        "polygon": polygon_np,
        "min_mask_overlap": float(data.get("min_mask_overlap", 0.20)),
        "min_center_in_roi": bool(data.get("min_center_in_roi", True)),
        "yaml_path": path,
    }
    return cfg


def scale_roi_polygon_if_needed(
    polygon: np.ndarray,
    yaml_width: int,
    yaml_height: int,
    current_width: int,
    current_height: int,
) -> np.ndarray:
    polygon = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)

    if yaml_width <= 0 or yaml_height <= 0:
        return np.round(polygon).astype(np.int32)

    if yaml_width == current_width and yaml_height == current_height:
        return np.round(polygon).astype(np.int32)

    sx = float(current_width) / float(yaml_width)
    sy = float(current_height) / float(yaml_height)

    scaled = polygon.copy()
    scaled[:, 0] *= sx
    scaled[:, 1] *= sy

    rospy.logwarn(
        "Build ROI image size differs: yaml=%dx%d current=%dx%d, polygon scaled by sx=%.4f sy=%.4f",
        yaml_width,
        yaml_height,
        current_width,
        current_height,
        sx,
        sy,
    )

    return np.round(scaled).astype(np.int32)


def make_roi_mask(image_shape: Tuple[int, int, int], polygon: np.ndarray) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 1)
    return mask


def classify_mask_build_region(
    mask: np.ndarray,
    roi_mask: Optional[np.ndarray],
    min_mask_overlap: float = 0.20,
    min_center_in_roi: bool = True,
) -> Dict[str, Any]:
    if roi_mask is None:
        return {
            "in_build_region": False,
            "overlap_ratio": 0.0,
            "center_in_roi": False,
            "mask_center_px": None,
        }

    mask_bool = np.asarray(mask).astype(bool)
    roi_bool = np.asarray(roi_mask).astype(bool)

    mask_area = int(mask_bool.sum())
    if mask_area <= 0:
        return {
            "in_build_region": False,
            "overlap_ratio": 0.0,
            "center_in_roi": False,
            "mask_center_px": None,
        }

    overlap = int(np.logical_and(mask_bool, roi_bool).sum())
    overlap_ratio = float(overlap) / float(mask_area)

    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        center_in_roi = False
        center_px = None
    else:
        cx = int(round(float(np.mean(xs))))
        cy = int(round(float(np.mean(ys))))
        center_px = [cx, cy]

        h, w = roi_mask.shape[:2]
        center_in_roi = bool(0 <= cx < w and 0 <= cy < h and roi_mask[cy, cx] > 0)

    if min_center_in_roi:
        in_build_region = bool(center_in_roi or overlap_ratio >= float(min_mask_overlap))
    else:
        in_build_region = bool(overlap_ratio >= float(min_mask_overlap))

    return {
        "in_build_region": in_build_region,
        "overlap_ratio": float(overlap_ratio),
        "center_in_roi": bool(center_in_roi),
        "mask_center_px": center_px,
    }


# ============================================================
# Detector node
# ============================================================

class GlobalSegYoloV11:
    """
    外部 RealSense + YOLO-seg 检测节点。

    输出：
      /global_camera/seg_detections std_msgs/String(JSON)

    每个 detection 主要字段：
      class_name
      raw_label
      score
      bbox2d
      center_camera
      center_table
      center_sg
      points_camera
      points_table
      points_sg
      corners_sg
      depth_values
      point_count
      in_build_region
      build_region_overlap
      center_in_build_region
      region_name
      attributes
    """

    def __init__(self):
        if not rospy.core.is_initialized():
            rospy.init_node("seg_yolov11", anonymous=True)

        self.pipeline = None
        self.align = None
        self.depth_scale = 0.001
        self.model = None
        self.T_table_camera = None

        self.frame_index = 0
        self.is_processing = False
        self.processing_lock = threading.Lock()
        self.transaction_lock = threading.Lock()
        self.pending_validation_request = None
        self.pending_validation_started_monotonic = None
        self.last_processing_monotonic = None
        self.last_saved_first_frame = False

        self.build_region_cfg: Optional[Dict[str, Any]] = None
        self.build_region_polygon: Optional[np.ndarray] = None
        self.build_region_roi_mask: Optional[np.ndarray] = None

        self._load_params()
        self._load_table_transform()
        self._load_build_region_roi()
        self._load_model()
        self._start_camera()

        self.pub = rospy.Publisher(self.output_topic, String, queue_size=self.output_queue_size)
        self.observation_request_topic = str(
            rospy.get_param("~observation_request_topic", TOPIC_OBSERVATION_REQUEST)
        ).strip()
        self.observation_request_sub = rospy.Subscriber(
            self.observation_request_topic,
            String,
            self._observation_request_cb,
            queue_size=10,
        )

        self.timer = rospy.Timer(
            rospy.Duration(1.0 / max(self.transaction_process_hz, self.process_hz, 0.1)),
            self._timer_cb,
        )

        rospy.on_shutdown(self.shutdown)

        rospy.loginfo(
            "seg_yolov11 initialized: topic=%s model=%s serial=%s",
            self.output_topic,
            self.model_path,
            self.camera_serial,
        )

        if self.enable_scene_graph_mirror_x:
            rospy.loginfo(
                "Scene graph axis mapping: scene_graph_xyz = [2*mirror_center_x - table_x, table_z, table_y], mirror_center_x=%.4f",
                self.mirror_center_x,
            )
        else:
            rospy.loginfo(
                "Scene graph axis mapping: scene_graph_xyz = [table_x, table_z, table_y]"
            )

    # ============================================================
    # Params
    # ============================================================

    def _load_params(self):
        self.output_topic = str(
            rospy.get_param("~output_topic", DEFAULT_OUTPUT_TOPIC)
        ).strip()

        self.camera_serial = str(
            rospy.get_param("~global_camera_serial", DEFAULT_GLOBAL_CAMERA_SERIAL)
        ).strip()

        self.model_path = str(
            rospy.get_param("~global_model_path", DEFAULT_GLOBAL_MODEL_PATH)
        ).strip()

        self.table_transform_yaml = str(
            rospy.get_param("~table_transform_yaml", DEFAULT_TABLE_TRANSFORM_YAML)
        ).strip()

        self.image_width = int(rospy.get_param("~global_image_width", 640))
        self.image_height = int(rospy.get_param("~global_image_height", 480))
        self.image_fps = int(rospy.get_param("~global_image_fps", 15))
        self.capture_uncertainty_sec = max(
            0.0,
            float(
                rospy.get_param(
                    "~capture_uncertainty_sec",
                    max(0.1, 2.0 / max(float(self.image_fps), 1.0)),
                )
            ),
        )

        self.conf_threshold = float(rospy.get_param("~global_conf_threshold", 0.35))
        self.process_hz = float(rospy.get_param("~global_process_hz", 1.0))
        self.transaction_process_hz = max(
            self.process_hz,
            float(rospy.get_param("~global_transaction_process_hz", 3.0)),
        )
        self.output_queue_size = max(1, int(rospy.get_param("~output_queue_size", 1)))
        self.show_window = bool(rospy.get_param("~show_window", True))

        self.min_depth = float(rospy.get_param("~global_min_depth", 0.2))
        self.max_depth = float(rospy.get_param("~global_max_depth", 2.0))
        self.min_points = int(rospy.get_param("~global_min_points", 30))
        self.erode_iter = int(rospy.get_param("~global_erode_iter", 1))
        self.depth_band_m = float(rospy.get_param("~global_depth_band_m", 0.015))
        self.max_mask_points = int(rospy.get_param("~global_max_mask_points", 2500))
        self.max_publish_points = int(rospy.get_param("~max_publish_points", 800))

        allowed_classes = rospy.get_param(
            "~global_allowed_classes",
            ["arch", "cube", "cuboid", "triangle", "gripper"],
        )
        self.allowed_classes = set(
            normalize_class_name(x) for x in allowed_classes
            if str(x).strip()
        )

        self.publish_empty_detections = bool(
            rospy.get_param("~publish_empty_detections", True)
        )

        # table -> scene_graph coordinate mapping
        self.enable_scene_graph_mirror_x = bool(
            rospy.get_param("~enable_scene_graph_mirror_x", True)
        )
        self.mirror_center_x = float(rospy.get_param("~mirror_center_x", 0.0))

        # build region ROI
        self.enable_build_region_roi = bool(
            rospy.get_param("~enable_build_region_roi", True)
        )
        self.build_region_roi_path = str(
            rospy.get_param("~build_region_roi_path", DEFAULT_BUILD_REGION_ROI_YAML)
        ).strip()

        # save first frame for ROI marking
        self.save_first_frame_path = str(
            rospy.get_param("~save_first_frame_path", "")
        ).strip()

        self.draw_masks = bool(rospy.get_param("~draw_masks", True))
        self.draw_build_region = bool(rospy.get_param("~draw_build_region", True))

    def _load_table_transform(self):
        self.T_table_camera = load_matrix4x4_from_yaml(
            self.table_transform_yaml,
            "T_table_camera",
        )

        R = self.T_table_camera[:3, :3]
        det_R = float(np.linalg.det(R))
        orth_error = float(np.linalg.norm(R.T @ R - np.eye(3)))

        rospy.loginfo("Loaded T_table_camera from %s", self.table_transform_yaml)
        rospy.loginfo("T_table_camera det(R)=%.6f orth_error=%.6e", det_R, orth_error)

    def _load_build_region_roi(self):
        if not self.enable_build_region_roi:
            rospy.loginfo("Build region ROI disabled.")
            self.build_region_cfg = None
            return

        try:
            cfg = load_build_region_roi(self.build_region_roi_path)
        except Exception as exc:
            rospy.logwarn("Failed to load build region ROI: %s", str(exc))
            cfg = None

        self.build_region_cfg = cfg

        if cfg is not None:
            rospy.loginfo(
                "Loaded build region ROI from %s: region_name=%s points=%d min_mask_overlap=%.3f min_center_in_roi=%s",
                cfg["yaml_path"],
                cfg["region_name"],
                len(cfg["polygon"]),
                float(cfg["min_mask_overlap"]),
                str(cfg["min_center_in_roi"]),
            )

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("YOLO segmentation model not found: %s" % self.model_path)

        self.model = YOLO(self.model_path)
        rospy.loginfo("Loaded YOLO model: %s", self.model_path)
        rospy.loginfo("YOLO task: %s", getattr(self.model, "task", "unknown"))

    def _start_camera(self):
        pipeline = rs.pipeline()
        config = rs.config()

        if self.camera_serial:
            rospy.loginfo("Opening global RealSense serial=%s", self.camera_serial)
            config.enable_device(str(self.camera_serial))
        else:
            rospy.logwarn("No ~global_camera_serial specified. First available RealSense may be used.")

        config.enable_stream(
            rs.stream.depth,
            self.image_width,
            self.image_height,
            rs.format.z16,
            self.image_fps,
        )
        config.enable_stream(
            rs.stream.color,
            self.image_width,
            self.image_height,
            rs.format.bgr8,
            self.image_fps,
        )

        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)

        device = profile.get_device()
        opened_serial = device.get_info(rs.camera_info.serial_number)
        opened_name = device.get_info(rs.camera_info.name)

        depth_sensor = device.first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())

        rospy.loginfo("Opened global RealSense: name=%s serial=%s", opened_name, opened_serial)
        rospy.loginfo("Depth scale: %.8f", self.depth_scale)

        if self.camera_serial and str(opened_serial) != str(self.camera_serial):
            rospy.logwarn(
                "Opened serial=%s differs from requested serial=%s",
                opened_serial,
                self.camera_serial,
            )

        self.pipeline = pipeline
        self.align = align

        for _ in range(10):
            self._get_aligned_images()

    # ============================================================
    # Shutdown
    # ============================================================

    def shutdown(self):
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
                self.pipeline = None
        except Exception:
            pass

        if self.show_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    # ============================================================
    # Frames
    # ============================================================

    def _get_aligned_images(self):
        if self.pipeline is None or self.align is None:
            return None, None, None, None

        frames = self.pipeline.wait_for_frames()
        host_receive_stamp = float(rospy.Time.now().to_sec())
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None, None, None

        try:
            color_source_timestamp = float(color_frame.get_timestamp())
        except Exception:
            color_source_timestamp = None
        try:
            depth_source_timestamp = float(depth_frame.get_timestamp())
        except Exception:
            depth_source_timestamp = None
        try:
            timestamp_domain = str(color_frame.get_frame_timestamp_domain()).split(".")[-1]
        except Exception:
            timestamp_domain = "unknown"

        color_depth_skew_sec = None
        if color_source_timestamp is not None and depth_source_timestamp is not None:
            color_depth_skew_sec = abs(color_source_timestamp - depth_source_timestamp) / 1000.0

        interval_width = max(
            float(self.capture_uncertainty_sec),
            0.0 if color_depth_skew_sec is None else float(color_depth_skew_sec),
        )
        capture_interval = CaptureInterval.from_host_receive(
            host_receive_stamp=host_receive_stamp,
            uncertainty_sec=interval_width,
            timestamp_source="host_receive",
            source_timestamp=color_source_timestamp,
            source_timestamp_domain=timestamp_domain,
        )

        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        capture_metadata = capture_interval.to_dict()
        capture_metadata.update(
            {
                "color_source_timestamp": color_source_timestamp,
                "depth_source_timestamp": depth_source_timestamp,
                "color_depth_skew_sec": color_depth_skew_sec,
            }
        )

        return color_image, depth_image, depth_intrin, capture_metadata

    def _ensure_build_region_mask(self, color_image: np.ndarray):
        if self.build_region_cfg is None:
            return

        if self.build_region_roi_mask is not None:
            return

        h, w = color_image.shape[:2]

        polygon = scale_roi_polygon_if_needed(
            polygon=self.build_region_cfg["polygon"],
            yaml_width=int(self.build_region_cfg.get("image_width", 0)),
            yaml_height=int(self.build_region_cfg.get("image_height", 0)),
            current_width=w,
            current_height=h,
        )

        self.build_region_polygon = polygon
        self.build_region_roi_mask = make_roi_mask(color_image.shape, polygon)

        rospy.loginfo(
            "Build region ROI mask ready: image=%dx%d polygon=%s",
            w,
            h,
            polygon.astype(int).tolist(),
        )

    # ============================================================
    # Timer
    # ============================================================

    def _observation_request_cb(self, msg: String):
        try:
            payload = json.loads(str(msg.data or "{}"))
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Invalid observation request JSON: %s", str(exc))
            return

        event = str(payload.get("event", "start") or "start").strip().lower()
        if event == "cancel":
            barrier_id = str(payload.get("barrier_id", "") or "").strip()
            with self.transaction_lock:
                current = self.pending_validation_request
                if current is not None and current.context.barrier_id == barrier_id:
                    self.pending_validation_request = None
                    self.pending_validation_started_monotonic = None
            return

        try:
            request = ValidationRequest.from_dict(payload)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Rejected observation request: %s", str(exc))
            return

        with self.transaction_lock:
            current = self.pending_validation_request
            if current is not None and current.request_id == request.request_id:
                return
            if current is not None and self.pending_validation_started_monotonic is not None:
                if time.monotonic() - self.pending_validation_started_monotonic < current.timeout_sec:
                    rospy.logwarn_throttle(
                        2.0,
                        "Ignoring observation request %s while %s is active",
                        request.request_id,
                        current.request_id,
                    )
                    return
            self.pending_validation_request = request
            self.pending_validation_started_monotonic = time.monotonic()

    def _transaction_context_for_capture(self, capture_metadata: Dict[str, Any]) -> Dict[str, Any]:
        with self.transaction_lock:
            request = self.pending_validation_request
            started = self.pending_validation_started_monotonic
            if request is None or started is None:
                return {}
            if time.monotonic() - started >= request.timeout_sec:
                self.pending_validation_request = None
                self.pending_validation_started_monotonic = None
                return {}
            try:
                capture_lower_bound = float(capture_metadata.get("capture_lower_bound"))
            except (TypeError, ValueError):
                return {}
            if capture_lower_bound < request.not_before:
                return {}

            context = request.context.to_dict()
            context.update(
                {
                    "request_id": request.request_id,
                    "validation_policy": request.policy,
                    "not_before": request.not_before,
                    "required_distinct_frames": request.required_distinct_frames,
                }
            )
            return context

    def _timer_cb(self, _event):
        now_monotonic = time.monotonic()
        with self.transaction_lock:
            request = self.pending_validation_request
            started = self.pending_validation_started_monotonic
            transaction_active = bool(
                request is not None
                and started is not None
                and now_monotonic - started < request.timeout_sec
            )
        desired_rate = self.transaction_process_hz if transaction_active else self.process_hz
        if not processing_cycle_due(
            self.last_processing_monotonic,
            now_monotonic,
            desired_rate,
        ):
            return
        if not self.processing_lock.acquire(False):
            return

        self.last_processing_monotonic = now_monotonic
        self.is_processing = True
        try:
            self._process_one_frame()
        except Exception as exc:
            rospy.logerr_throttle(2.0, "seg_yolov11 processing failed: %s", str(exc))
        finally:
            self.is_processing = False
            self.processing_lock.release()

    def _process_one_frame(self):
        self.frame_index += 1

        color_image, depth_image, depth_intrin, capture_metadata = self._get_aligned_images()
        if color_image is None or depth_image is None:
            rospy.logwarn_throttle(2.0, "Skipping: empty RealSense frame")
            return

        self._ensure_build_region_mask(color_image)

        if self.save_first_frame_path and not self.last_saved_first_frame:
            try:
                cv2.imwrite(self.save_first_frame_path, color_image)
                rospy.loginfo("Saved first global camera frame: %s", self.save_first_frame_path)
            except Exception as exc:
                rospy.logwarn("Failed to save first frame: %s", str(exc))
            self.last_saved_first_frame = True

        inference_start_stamp = float(rospy.Time.now().to_sec())
        results = self.model.predict(
            color_image,
            conf=self.conf_threshold,
            verbose=False,
        )
        inference_end_stamp = float(rospy.Time.now().to_sec())

        detections: List[Dict[str, Any]] = []
        result0 = None

        if results and len(results) > 0:
            result0 = results[0]
            if getattr(result0, "masks", None) is not None and result0.masks is not None:
                detections = self._extract_segmentation_instances(
                    result0=result0,
                    color_image=color_image,
                    depth_image=depth_image,
                    depth_intrin=depth_intrin,
                )
            else:
                rospy.logwarn_throttle(
                    2.0,
                    "YOLO result has no masks. Please use a segmentation model.",
                )

        if detections or self.publish_empty_detections:
            transaction_context = self._transaction_context_for_capture(capture_metadata)
            payload = self._make_payload(
                detections,
                capture_metadata=capture_metadata,
                inference_start_stamp=inference_start_stamp,
                inference_end_stamp=inference_end_stamp,
                transaction_context=transaction_context,
            )
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self.pub.publish(msg)

        rospy.loginfo_throttle(
            1.0,
            "seg_yolov11 published detections: count=%d in_build_region=%d",
            len(detections),
            sum(1 for d in detections if bool(d.get("in_build_region", False))),
        )

        if self.show_window:
            self._show_debug_window(color_image, detections, result0)

    # ============================================================
    # Detection extraction
    # ============================================================

    def _extract_segmentation_instances(
        self,
        result0,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        depth_intrin,
    ) -> List[Dict[str, Any]]:
        boxes_obj = result0.boxes
        masks_obj = result0.masks

        if boxes_obj is None or masks_obj is None:
            return []

        boxes = boxes_obj.xyxy.cpu().numpy().astype(np.float32)
        classes = boxes_obj.cls.cpu().numpy().astype(np.int32)
        scores = boxes_obj.conf.cpu().numpy().astype(np.float32)

        masks = masks_obj.data.cpu().numpy()
        names = getattr(self.model, "names", {}) or {}

        h, w = color_image.shape[:2]
        out: List[Dict[str, Any]] = []

        for i in range(len(boxes)):
            score = float(scores[i])
            cls_id = int(classes[i])

            raw_label = str(names.get(cls_id, cls_id))
            class_name = normalize_class_name(raw_label)

            if self.allowed_classes and class_name not in self.allowed_classes:
                continue

            mask = masks[i]
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)

            mask_bool_original = mask > 0.5
            if int(mask_bool_original.sum()) <= 0:
                continue

            mask_bool = mask_bool_original.copy()

            if self.erode_iter > 0:
                kernel = np.ones((3, 3), dtype=np.uint8)
                mask_uint8 = mask_bool.astype(np.uint8)
                mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=int(self.erode_iter))
                mask_bool = mask_uint8.astype(bool)

            geometry = self._extract_mask_geometry(
                mask_bool=mask_bool,
                depth_image=depth_image,
                depth_intrin=depth_intrin,
            )

            if geometry is None:
                continue

            if geometry["point_count"] < self.min_points:
                continue

            region = self._classify_detection_region(mask_bool_original)

            x1, y1, x2, y2 = [float(v) for v in boxes[i].tolist()]

            states = []
            if region["in_build_region"]:
                states.append("in_build_region")
            else:
                states.append("outside_build_region")

            det = {
                "class_name": class_name,
                "raw_label": raw_label,
                "score": score,
                "bbox2d": [x1, y1, x2, y2],
                "mask_index": int(i),

                "center_camera": geometry["center_camera"],
                "center_table": geometry["center_table"],
                "center_sg": geometry["center_sg"],

                "points_camera": geometry["points_camera"],
                "points_table": geometry["points_table"],
                "points_sg": geometry["points_sg"],
                "corners_sg": geometry["corners_sg"],

                "depth_values": geometry["depth_values"],
                "point_count": int(geometry["point_count"]),

                "in_build_region": bool(region["in_build_region"]),
                "build_region_overlap": float(region["overlap_ratio"]),
                "center_in_build_region": bool(region["center_in_roi"]),
                "build_region_center_px": region["mask_center_px"],
                "region_name": region["region_name"],

                "states": states,
                "attributes": {
                    "source": "seg_yolov11",
                    "camera_source": "global",
                    "detector_type": "yolo_segmentation",
                    "raw_label": raw_label,
                    "score": score,
                    "mask_index": int(i),
                    "bbox2d": [x1, y1, x2, y2],
                    "axis_mapping": self._axis_mapping_description(),
                    "build_region": {
                        "enabled": bool(self.enable_build_region_roi),
                        "roi_loaded": bool(self.build_region_roi_mask is not None),
                        "in_build_region": bool(region["in_build_region"]),
                        "overlap_ratio": float(region["overlap_ratio"]),
                        "center_in_roi": bool(region["center_in_roi"]),
                        "mask_center_px": region["mask_center_px"],
                        "region_name": region["region_name"],
                    },
                },
            }

            out.append(det)

        return out

    def _extract_mask_geometry(
        self,
        mask_bool: np.ndarray,
        depth_image: np.ndarray,
        depth_intrin,
    ) -> Optional[Dict[str, Any]]:
        h, w = depth_image.shape[:2]

        ys, xs = np.where(mask_bool)
        if len(xs) == 0:
            return None

        depths_raw = depth_image[ys, xs].astype(np.float32)
        depths_m = depths_raw * float(self.depth_scale)

        valid = np.isfinite(depths_m)
        valid &= depths_m >= float(self.min_depth)
        valid &= depths_m <= float(self.max_depth)

        if not np.any(valid):
            return None

        xs = xs[valid]
        ys = ys[valid]
        depths_m = depths_m[valid]

        if len(depths_m) <= 0:
            return None

        # Remove depth outliers around median, useful when mask covers background.
        if self.depth_band_m > 0 and len(depths_m) >= 10:
            med = float(np.median(depths_m))
            band_valid = np.abs(depths_m - med) <= float(self.depth_band_m)

            # If band filtering is too aggressive, keep original valid points.
            if int(np.sum(band_valid)) >= max(self.min_points, 10):
                xs = xs[band_valid]
                ys = ys[band_valid]
                depths_m = depths_m[band_valid]

        if len(depths_m) < self.min_points:
            return None

        if len(depths_m) > self.max_mask_points:
            idx = np.random.choice(len(depths_m), size=self.max_mask_points, replace=False)
            xs = xs[idx]
            ys = ys[idx]
            depths_m = depths_m[idx]

        points_camera = []
        for px, py, depth_m in zip(xs, ys, depths_m):
            try:
                p = rs.rs2_deproject_pixel_to_point(
                    depth_intrin,
                    [float(px), float(py)],
                    float(depth_m),
                )
            except Exception:
                continue

            if p is None or len(p) != 3:
                continue

            p = [float(p[0]), float(p[1]), float(p[2])]
            if all(np.isfinite(p)):
                points_camera.append(p)

        if len(points_camera) < self.min_points:
            return None

        points_camera_np = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
        points_table_np = transform_points(self.T_table_camera, points_camera_np)
        points_sg_np = self._table_points_to_scene_graph_points(points_table_np)

        center_camera = np.median(points_camera_np, axis=0)
        center_table = np.median(points_table_np, axis=0)
        center_sg = np.median(points_sg_np, axis=0)

        corners_sg = compute_aabb_corners(points_sg_np)

        points_camera_pub = maybe_downsample_points(points_camera_np, self.max_publish_points)
        points_table_pub = maybe_downsample_points(points_table_np, self.max_publish_points)
        points_sg_pub = maybe_downsample_points(points_sg_np, self.max_publish_points)

        depth_values_pub = maybe_downsample_points(depths_m.reshape(-1, 1), min(self.max_publish_points, 300)).reshape(-1)

        return {
            "center_camera": center_camera.astype(float).tolist(),
            "center_table": center_table.astype(float).tolist(),
            "center_sg": center_sg.astype(float).tolist(),

            "points_camera": points_camera_pub.astype(float).tolist(),
            "points_table": points_table_pub.astype(float).tolist(),
            "points_sg": points_sg_pub.astype(float).tolist(),

            "corners_sg": corners_sg.astype(float).tolist(),
            "depth_values": depth_values_pub.astype(float).tolist(),
            "point_count": int(len(points_camera_np)),
        }

    def _classify_detection_region(self, mask_bool_original: np.ndarray) -> Dict[str, Any]:
        if self.build_region_cfg is None or self.build_region_roi_mask is None:
            return {
                "in_build_region": False,
                "overlap_ratio": 0.0,
                "center_in_roi": False,
                "mask_center_px": None,
                "region_name": "none",
            }

        info = classify_mask_build_region(
            mask=mask_bool_original,
            roi_mask=self.build_region_roi_mask,
            min_mask_overlap=float(self.build_region_cfg["min_mask_overlap"]),
            min_center_in_roi=bool(self.build_region_cfg["min_center_in_roi"]),
        )

        info["region_name"] = str(self.build_region_cfg.get("region_name", "build_area"))
        return info

    # ============================================================
    # Coordinate mapping
    # ============================================================

    def _table_points_to_scene_graph_points(self, points_table: np.ndarray) -> np.ndarray:
        points_table = np.asarray(points_table, dtype=np.float64).reshape(-1, 3)
        if points_table.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        table_x = points_table[:, 0]
        table_y = points_table[:, 1]
        table_z = points_table[:, 2]

        if self.enable_scene_graph_mirror_x:
            sg_x = 2.0 * float(self.mirror_center_x) - table_x
        else:
            sg_x = table_x

        # scene_graph.py uses Y as vertical axis.
        sg_y = table_z
        sg_z = table_y

        return np.stack([sg_x, sg_y, sg_z], axis=1)

    def _axis_mapping_description(self) -> str:
        if self.enable_scene_graph_mirror_x:
            return "scene_graph_xyz = [2*mirror_center_x - table_x, table_z, table_y]"
        return "scene_graph_xyz = [table_x, table_z, table_y]"

    # ============================================================
    # Payload
    # ============================================================

    def _make_payload(
        self,
        detections: List[Dict[str, Any]],
        *,
        capture_metadata: Dict[str, Any],
        inference_start_stamp: float,
        inference_end_stamp: float,
        transaction_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        publish_stamp = float(rospy.Time.now().to_sec())
        capture_stamp = float(capture_metadata["capture_stamp"])

        roi_payload = {
            "enabled": bool(self.enable_build_region_roi),
            "roi_loaded": bool(self.build_region_roi_mask is not None),
            "region_name": "none",
            "polygon": [],
            "min_mask_overlap": None,
            "min_center_in_roi": None,
        }

        if self.build_region_cfg is not None:
            roi_payload.update(
                {
                    "region_name": str(self.build_region_cfg.get("region_name", "build_area")),
                    "polygon": (
                        self.build_region_polygon.astype(int).tolist()
                        if self.build_region_polygon is not None
                        else np.asarray(self.build_region_cfg["polygon"], dtype=np.int32).tolist()
                    ),
                    "min_mask_overlap": float(self.build_region_cfg["min_mask_overlap"]),
                    "min_center_in_roi": bool(self.build_region_cfg["min_center_in_roi"]),
                }
            )

        payload = {
            # Legacy consumers read "stamp".  It now intentionally means
            # physical capture time, never inference/publish completion time.
            "stamp": capture_stamp,
            "capture_stamp": capture_stamp,
            "capture_lower_bound": float(capture_metadata["capture_lower_bound"]),
            "capture_upper_bound": float(capture_metadata["capture_upper_bound"]),
            "capture_uncertainty_sec": float(capture_metadata["capture_uncertainty_sec"]),
            "capture_timestamp_source": str(capture_metadata["capture_timestamp_source"]),
            "source_timestamp": capture_metadata.get("source_timestamp"),
            "source_timestamp_domain": str(capture_metadata.get("source_timestamp_domain", "unknown")),
            "color_source_timestamp": capture_metadata.get("color_source_timestamp"),
            "depth_source_timestamp": capture_metadata.get("depth_source_timestamp"),
            "color_depth_skew_sec": capture_metadata.get("color_depth_skew_sec"),
            "inference_start_stamp": float(inference_start_stamp),
            "inference_end_stamp": float(inference_end_stamp),
            "publish_stamp": publish_stamp,
            "frame_index": int(self.frame_index),
            "source_frame_seq": int(self.frame_index),
            "source": "seg_yolov11",
            "camera_source": "global",
            "detector_type": "yolo_segmentation",
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "coordinate_frames": {
                "camera_frame": "global_realsense_color_optical_frame",
                "table_frame": "global_table_frame",
                "scene_graph_frame": "scene_graph_frame",
                "axis_mapping": self._axis_mapping_description(),
                "mirror_center_x": float(self.mirror_center_x),
                "enable_scene_graph_mirror_x": bool(self.enable_scene_graph_mirror_x),
            },
            "build_region": roi_payload,
            "detections": list(detections or []),
        }
        payload.update(dict(transaction_context or {}))
        return payload

    # ============================================================
    # Debug display
    # ============================================================

    def _show_debug_window(
        self,
        color_image: np.ndarray,
        detections: List[Dict[str, Any]],
        result0: Any = None,
    ):
        vis = color_image.copy()

        # Draw ROI first.
        if self.draw_build_region and self.build_region_polygon is not None:
            polygon = np.asarray(self.build_region_polygon, dtype=np.int32)
            cv2.polylines(vis, [polygon], isClosed=True, color=(0, 255, 255), thickness=2)

            label_pos = tuple(polygon[0].astype(int).tolist())
            cv2.putText(
                vis,
                "BUILD_REGION",
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

        # Optional masks overlay from ultralytics result.
        if self.draw_masks and result0 is not None and getattr(result0, "masks", None) is not None:
            try:
                masks = result0.masks.data.cpu().numpy()
                h, w = vis.shape[:2]
                overlay = vis.copy()

                for i, mask in enumerate(masks):
                    if mask.shape[:2] != (h, w):
                        mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                    mask_bool = mask > 0.5
                    overlay[mask_bool] = (0.5 * overlay[mask_bool] + 0.5 * np.array([0, 180, 0])).astype(np.uint8)

                vis = cv2.addWeighted(overlay, 0.55, vis, 0.45, 0.0)
            except Exception:
                pass

        for det in detections:
            bbox = det.get("bbox2d", None)
            if bbox is None:
                continue

            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            in_region = bool(det.get("in_build_region", False))

            color = (0, 255, 0) if in_region else (0, 0, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            cls = str(det.get("class_name", "object"))
            score = float(det.get("score", 0.0))
            overlap = float(det.get("build_region_overlap", 0.0))
            region_text = "IN" if in_region else "OUT"

            label = f"{cls} {score:.2f} {region_text} ov={overlap:.2f}"
            cv2.putText(
                vis,
                label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                2,
            )

            center_px = det.get("build_region_center_px", None)
            if center_px is not None and len(center_px) == 2:
                cv2.circle(vis, (int(center_px[0]), int(center_px[1])), 3, color, -1)

        cv2.imshow("seg_yolov11_global", vis)
        cv2.waitKey(1)


# ============================================================
# Main
# ============================================================

def main():
    node = GlobalSegYoloV11()
    rospy.spin()


if __name__ == "__main__":
    main()
