#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from scene_graph_system.resources import package_resource_path
import os
import sys

import math
import cv2
import numpy as np
import pyrealsense2 as rs
import rospy
from blockkit.msg import ObjectInfo
from cv_bridge import CvBridge
from scene_graph_system.msg import DetectedObjects
from sensor_msgs.msg import CameraInfo, Image
from ultralytics import YOLO

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

from scene_graph_system.resources import find_package_resource
from scene_graph_system.validation.validation_transaction import CaptureInterval


# ============================================================
# 抓取角度计算常量（可通过 ROS param 覆盖）
# ============================================================

GRIPPER_FINGER_AXIS_RAD = math.pi / 2.0
GRASP_ANGLE_BIAS_RAD = 0.0


# ============================================================
# 固定臂上相机配置
# ============================================================

# 如果实际臂上相机不是这个 serial，只改这里即可。
WRIST_CAMERA_SERIAL = "215322077258"

WRIST_CAMERA_NAME = "wrist"

WRIST_DETECTED_OBJECTS_TOPIC = "/wrist/detected_objects"
WRIST_DEPTH_TOPIC = "/wrist/camera/aligned_depth_to_color/image_raw"
WRIST_CAMERA_INFO_TOPIC = "/wrist/camera/color/camera_info"

# catch.py 仍然订阅 /object_pose，所以臂上相机必须继续发布这个 topic。
WRIST_OBJECT_POSE_TOPIC = "/object_pose"

WRIST_FRAME_ID = "wrist_camera_color_optical_frame"

DEFAULT_MODEL_PATH = package_resource_path('scripts/best.pt')


bridge = CvBridge()


def create_realsense_pipeline(width, height, fps):
    """
    固定打开臂上 RealSense 相机。
    双相机环境下，必须用 serial 固定设备，否则可能打开外部相机或出现设备占用。
    """
    pipeline = rs.pipeline()
    config = rs.config()

    rospy.loginfo("Opening wrist RealSense camera serial=%s", WRIST_CAMERA_SERIAL)
    config.enable_device(str(WRIST_CAMERA_SERIAL))

    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)

    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    color_intr = color_profile.get_intrinsics()

    try:
        device = profile.get_device()
        opened_serial = device.get_info(rs.camera_info.serial_number)
        opened_name = device.get_info(rs.camera_info.name)

        rospy.loginfo("Opened wrist RealSense: name=%s serial=%s", opened_name, opened_serial)

        if str(opened_serial) != str(WRIST_CAMERA_SERIAL):
            rospy.logwarn(
                "Opened camera serial=%s differs from expected wrist serial=%s",
                opened_serial,
                WRIST_CAMERA_SERIAL,
            )

    except Exception as exc:
        rospy.logwarn("Failed to query RealSense device info: %s", str(exc))

    return pipeline, align, color_intr


def get_aligned_images(pipeline, align, capture_uncertainty_sec):
    frames = pipeline.wait_for_frames()
    host_receive_stamp = float(rospy.Time.now().to_sec())
    aligned_frames = align.process(frames)

    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not aligned_depth_frame or not color_frame:
        return None, None, None, None, None, None

    try:
        color_source_timestamp = float(color_frame.get_timestamp())
    except Exception:
        color_source_timestamp = None
    try:
        timestamp_domain = str(color_frame.get_frame_timestamp_domain()).split(".")[-1]
    except Exception:
        timestamp_domain = "unknown"

    capture_interval = CaptureInterval.from_host_receive(
        host_receive_stamp=host_receive_stamp,
        uncertainty_sec=float(capture_uncertainty_sec),
        timestamp_source="host_receive",
        source_timestamp=color_source_timestamp,
        source_timestamp_domain=timestamp_domain,
    )

    intr = color_frame.profile.as_video_stream_profile().intrinsics
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    return intr, depth_intrin, color_image, depth_image, aligned_depth_frame, capture_interval


def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin):
    x = int(depth_pixel[0])
    y = int(depth_pixel[1])

    distance = aligned_depth_frame.get_distance(x, y)

    if distance <= 0:
        return distance, [0.0, 0.0, 0.0]

    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], distance)
    return distance, camera_coordinate


def normalize_half_turn_rad(angle_rad):
    return (angle_rad + math.pi / 2.0) % math.pi - math.pi / 2.0


def compute_long_edge_angle_rad(corners):
    corners = corners.astype(np.float32)
    edge_vectors = np.roll(corners, -1, axis=0) - corners
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    long_edge_index = int(np.argmax(edge_lengths))
    long_edge_vector = edge_vectors[long_edge_index]
    long_edge_angle_rad = math.atan2(float(long_edge_vector[1]), float(long_edge_vector[0]))
    return normalize_half_turn_rad(long_edge_angle_rad)


def compute_grasp_rotation_rad(long_edge_angle_rad):
    return normalize_half_turn_rad(
        long_edge_angle_rad - GRIPPER_FINGER_AXIS_RAD + GRASP_ANGLE_BIAS_RAD
    )


def draw_direction_arrow(image, center, angle_rad, length, color, label):
    center = np.asarray(center, dtype=np.float32)
    direction = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=np.float32)
    start = center - direction * (length * 0.45)
    end = center + direction * (length * 0.55)
    cv2.arrowedLine(
        image,
        (int(start[0]), int(start[1])),
        (int(end[0]), int(end[1])),
        color,
        3,
        cv2.LINE_AA,
        0,
        0.2,
    )
    label_pos = (int(end[0] + 6), int(end[1] - 6))
    cv2.putText(
        image,
        label,
        label_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_angle_label(image, corners, long_edge_deg, grasp_deg):
    corners = corners.astype(int)
    top_left_index = np.argmin(corners[:, 0] + corners[:, 1])
    label_anchor = corners[top_left_index].copy()
    label_anchor[1] = max(32, label_anchor[1] - 24)
    angle_text = f"long: {long_edge_deg:.1f} deg"
    grasp_text = f"grasp: {grasp_deg:.1f} deg"
    cv2.putText(
        image,
        angle_text,
        (int(label_anchor[0]), int(label_anchor[1])),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        grasp_text,
        (int(label_anchor[0]), int(label_anchor[1] + 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 165, 255),
        2,
        cv2.LINE_AA,
    )


def extract_detections(result):
    """
    统一提取检测结果，兼容：
    1. 普通 YOLO 检测模型：result.boxes
    2. OBB 旋转框模型：result.obb
    3. 无检测结果：返回空数组
    """
    detected_boxes = np.empty((0, 4), dtype=np.float32)
    detected_classes = np.empty((0,), dtype=np.int32)
    detected_scores = np.empty((0,), dtype=np.float32)

    # 普通检测模型
    if getattr(result, "boxes", None) is not None:
        boxes_obj = result.boxes

        if getattr(boxes_obj, "xyxy", None) is not None:
            xyxy = boxes_obj.xyxy
            if xyxy is not None and len(xyxy) > 0:
                detected_boxes = xyxy.cpu().numpy().astype(np.float32)

        if getattr(boxes_obj, "cls", None) is not None:
            cls = boxes_obj.cls
            if cls is not None and len(cls) > 0:
                detected_classes = cls.cpu().numpy().astype(np.int32)

        if getattr(boxes_obj, "conf", None) is not None:
            conf = boxes_obj.conf
            if conf is not None and len(conf) > 0:
                detected_scores = conf.cpu().numpy().astype(np.float32)

        return detected_boxes, detected_classes, detected_scores

    # OBB 旋转框模型
    if getattr(result, "obb", None) is not None:
        obb_obj = result.obb

        if getattr(obb_obj, "xyxyxyxy", None) is not None:
            corners = obb_obj.xyxyxyxy
            if corners is not None and len(corners) > 0:
                corners_np = corners.cpu().numpy()
                x_min = corners_np[:, :, 0].min(axis=1)
                y_min = corners_np[:, :, 1].min(axis=1)
                x_max = corners_np[:, :, 0].max(axis=1)
                y_max = corners_np[:, :, 1].max(axis=1)
                detected_boxes = np.stack(
                    [x_min, y_min, x_max, y_max],
                    axis=1,
                ).astype(np.float32)

        if getattr(obb_obj, "cls", None) is not None:
            cls = obb_obj.cls
            if cls is not None and len(cls) > 0:
                detected_classes = cls.cpu().numpy().astype(np.int32)

        if getattr(obb_obj, "conf", None) is not None:
            conf = obb_obj.conf
            if conf is not None and len(conf) > 0:
                detected_scores = conf.cpu().numpy().astype(np.float32)

        return detected_boxes, detected_classes, detected_scores

    return detected_boxes, detected_classes, detected_scores


def get_obb_corners(result):
    if result is None:
        return None
    obb = getattr(result, "obb", None)
    if obb is None:
        return None
    corners = getattr(obb, "xyxyxyxy", None)
    if corners is None or len(corners) == 0:
        return None
    return corners.cpu().numpy()


def resolve_class_name(result, cls_id):
    cls_id = int(cls_id)
    if result is not None and hasattr(result, "names") and cls_id in result.names:
        return str(result.names[cls_id])
    return str(cls_id)


def build_camera_info_msg(color_intr):
    msg = CameraInfo()

    msg.height = int(color_intr.height)
    msg.width = int(color_intr.width)
    msg.distortion_model = "plumb_bob"

    msg.D = [0.0, 0.0, 0.0, 0.0, 0.0]

    msg.K = [
        color_intr.fx, 0.0, color_intr.ppx,
        0.0, color_intr.fy, color_intr.ppy,
        0.0, 0.0, 1.0,
    ]

    msg.R = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]

    msg.P = [
        color_intr.fx, 0.0, color_intr.ppx, 0.0,
        0.0, color_intr.fy, color_intr.ppy, 0.0,
        0.0, 0.0, 1.0, 0.0,
    ]

    msg.header.frame_id = WRIST_FRAME_ID
    return msg


def clamp_box_to_image(box, image_shape):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = map(int, box.tolist())

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    return x1, y1, x2, y2


def main():
    rospy.init_node("wrist_object_detect", anonymous=True)

    image_width = int(rospy.get_param("~image_width", 640))
    image_height = int(rospy.get_param("~image_height", 480))
    image_fps = int(rospy.get_param("~image_fps", 15))
    capture_uncertainty_sec = max(
        0.0,
        float(
            rospy.get_param(
                "~capture_uncertainty_sec",
                max(0.1, 2.0 / max(float(image_fps), 1.0)),
            )
        ),
    )

    conf_threshold = float(rospy.get_param("~conf_threshold", 0.5))
    show_window = bool(rospy.get_param("~show_window", True))

    gripper_finger_axis_deg = float(rospy.get_param("~gripper_finger_axis_deg", 90.0))
    grasp_angle_bias_deg = float(rospy.get_param("~grasp_angle_bias_deg", 0.0))

    global GRIPPER_FINGER_AXIS_RAD, GRASP_ANGLE_BIAS_RAD
    GRIPPER_FINGER_AXIS_RAD = math.radians(gripper_finger_axis_deg)
    GRASP_ANGLE_BIAS_RAD = math.radians(grasp_angle_bias_deg)

    model_path = rospy.get_param("~model_path", DEFAULT_MODEL_PATH)
    if not model_path:
        model_path = find_package_resource("best.pt") or package_resource_path("best.pt")

    rospy.loginfo("=" * 70)
    rospy.loginfo("Starting wrist YOLO detector")
    rospy.loginfo("wrist serial: %s", WRIST_CAMERA_SERIAL)
    rospy.loginfo("model_path: %s", model_path)
    rospy.loginfo("detected_objects_topic: %s", WRIST_DETECTED_OBJECTS_TOPIC)
    rospy.loginfo("depth_topic: %s", WRIST_DEPTH_TOPIC)
    rospy.loginfo("camera_info_topic: %s", WRIST_CAMERA_INFO_TOPIC)
    rospy.loginfo("object_pose_topic: %s", WRIST_OBJECT_POSE_TOPIC)
    rospy.loginfo("=" * 70)

    pipeline = None

    try:
        pipeline, align, color_intr = create_realsense_pipeline(
            image_width,
            image_height,
            image_fps,
        )

        model = YOLO(model_path)

        print("[INFO] 完成 YOLO 模型加载")
        print("[INFO] model_path =", model_path)
        print("[INFO] model task =", getattr(model, "task", "unknown"))

        detected_objects_pub = rospy.Publisher(
            WRIST_DETECTED_OBJECTS_TOPIC,
            DetectedObjects,
            queue_size=10,
        )

        depth_pub = rospy.Publisher(
            WRIST_DEPTH_TOPIC,
            Image,
            queue_size=10,
        )

        camera_info_pub = rospy.Publisher(
            WRIST_CAMERA_INFO_TOPIC,
            CameraInfo,
            queue_size=10,
        )

        object_pose_pub = rospy.Publisher(
            WRIST_OBJECT_POSE_TOPIC,
            ObjectInfo,
            queue_size=10,
        )

        camera_info_msg = build_camera_info_msg(color_intr)

        if show_window:
            cv2.namedWindow(
                "wrist_detection",
                flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED,
            )

        rate = rospy.Rate(min(max(image_fps, 1), 30))
        frame_seq = 0

        while not rospy.is_shutdown():
            intr, depth_intrin, color_image, depth_image, aligned_depth_frame, capture_interval = get_aligned_images(
                pipeline,
                align,
                capture_uncertainty_sec,
            )

            if color_image is None or depth_image is None:
                rospy.logwarn_throttle(2.0, "Empty RealSense frame, skipping.")
                continue

            frame_seq += 1
            inference_start_stamp = rospy.Time.now()
            results = model.predict(color_image, conf=conf_threshold, verbose=False)
            inference_end_stamp = rospy.Time.now()

            if results is None or len(results) == 0:
                result0 = None
                canvas = color_image.copy()
                detected_boxes = np.empty((0, 4), dtype=np.float32)
                detected_classes = np.empty((0,), dtype=np.int32)
                detected_scores = np.empty((0,), dtype=np.float32)
            else:
                result0 = results[0]
                canvas = result0.plot()
                detected_boxes, detected_classes, detected_scores = extract_detections(result0)

            obb_corners_list = get_obb_corners(result0)

            capture_ros_stamp = rospy.Time.from_sec(capture_interval.capture_stamp)
            publish_stamp = rospy.Time.now()

            detected_msg = DetectedObjects()
            detected_msg.header.stamp = capture_ros_stamp
            detected_msg.header.frame_id = WRIST_FRAME_ID
            detected_msg.frame_seq = int(frame_seq)
            detected_msg.inference_start_stamp = inference_start_stamp
            detected_msg.inference_end_stamp = inference_end_stamp
            detected_msg.publish_stamp = publish_stamp
            detected_msg.capture_lower_bound = float(capture_interval.lower_bound)
            detected_msg.capture_upper_bound = float(capture_interval.upper_bound)
            detected_msg.capture_uncertainty_sec = float(capture_interval.uncertainty_sec)
            detected_msg.capture_timestamp_source = str(capture_interval.timestamp_source)
            detected_msg.source_timestamp = float(capture_interval.source_timestamp or 0.0)
            detected_msg.source_timestamp_domain = str(capture_interval.source_timestamp_domain)

            names = []
            boxes = []
            scores = []
            positions_xyz = []
            angles = []
            position_valid = []

            if len(detected_boxes) != len(detected_classes) or len(detected_boxes) != len(detected_scores):
                rospy.logwarn(
                    "Detection result length mismatch: boxes=%d, classes=%d, scores=%d",
                    len(detected_boxes),
                    len(detected_classes),
                    len(detected_scores),
                )
                min_len = min(len(detected_boxes), len(detected_classes), len(detected_scores))
                detected_boxes = detected_boxes[:min_len]
                detected_classes = detected_classes[:min_len]
                detected_scores = detected_scores[:min_len]
                if obb_corners_list is not None and len(obb_corners_list) > min_len:
                    obb_corners_list = obb_corners_list[:min_len]

            for idx, (box, cls_id, confidence) in enumerate(zip(detected_boxes, detected_classes, detected_scores)):
                x1, y1, x2, y2 = clamp_box_to_image(box, color_image.shape)

                if x2 <= x1 or y2 <= y1:
                    continue

                name = resolve_class_name(result0, cls_id)

                ux = int((x1 + x2) / 2)
                uy = int((y1 + y2) / 2)

                distance, camera_coordinate = get_3d_camera_coordinate(
                    [ux, uy],
                    aligned_depth_frame,
                    depth_intrin,
                )

                if distance <= 0:
                    formatted_camera_coordinate = "(invalid_depth)"
                    rospy.logwarn_throttle(
                        1.0,
                        "Detected %s, but center depth is invalid at pixel (%d, %d)",
                        name,
                        ux,
                        uy,
                    )
                    positions_xyz.extend([0.0, 0.0, 0.0])
                    angles.append(0.0)
                    position_valid.append(False)
                else:
                    formatted_camera_coordinate = (
                        f"({camera_coordinate[0]:.2f}, "
                        f"{camera_coordinate[1]:.2f}, "
                        f"{camera_coordinate[2]:.2f})"
                    )

                    rospy.loginfo_throttle(
                        1.0,
                        "Detected %s at wrist camera coordinate: %.3f, %.3f, %.3f",
                        name,
                        camera_coordinate[0],
                        camera_coordinate[1],
                        camera_coordinate[2],
                    )

                    object_msg = ObjectInfo()
                    object_msg.object_class = str(name)
                    object_msg.x = float(camera_coordinate[0])
                    object_msg.y = float(camera_coordinate[1])
                    object_msg.z = float(camera_coordinate[2])

                    if hasattr(object_msg, "angle"):
                        if obb_corners_list is not None and idx < len(obb_corners_list):
                            long_edge_rad = compute_long_edge_angle_rad(obb_corners_list[idx])
                            grasp_rad = compute_grasp_rotation_rad(long_edge_rad)
                            object_msg.angle = float(grasp_rad)

                            long_edge_deg = math.degrees(long_edge_rad)
                            grasp_deg = math.degrees(grasp_rad)
                            finger_angle_rad = normalize_half_turn_rad(
                                GRIPPER_FINGER_AXIS_RAD + grasp_rad
                            )
                            draw_direction_arrow(canvas, [ux, uy], long_edge_rad, 60, (0, 255, 0), "long")
                            draw_direction_arrow(canvas, [ux, uy], finger_angle_rad, 42, (0, 165, 255), "grip")
                            draw_angle_label(canvas, obb_corners_list[idx], long_edge_deg, grasp_deg)
                        else:
                            object_msg.angle = 0.0

                    positions_xyz.extend(
                        [
                            float(camera_coordinate[0]),
                            float(camera_coordinate[1]),
                            float(camera_coordinate[2]),
                        ]
                    )
                    angles.append(float(getattr(object_msg, "angle", 0.0)))
                    position_valid.append(True)

                    object_pose_pub.publish(object_msg)

                cv2.circle(canvas, (ux, uy), 4, (255, 255, 255), 5)
                cv2.putText(
                    canvas,
                    str(formatted_camera_coordinate),
                    (ux + 20, uy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    [225, 255, 255],
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

                names.append(str(name))
                boxes.extend([int(x1), int(y1), int(x2), int(y2)])
                scores.append(float(confidence))

            detected_msg.names = names
            detected_msg.boxes = boxes
            detected_msg.scores = scores
            detected_msg.positions_xyz = positions_xyz
            detected_msg.angles = angles
            detected_msg.position_valid = position_valid
            detected_objects_pub.publish(detected_msg)

            camera_info_msg.header.stamp = capture_ros_stamp
            camera_info_msg.header.frame_id = WRIST_FRAME_ID
            camera_info_pub.publish(camera_info_msg)

            depth_msg = bridge.cv2_to_imgmsg(depth_image, encoding="passthrough")
            depth_msg.header.stamp = capture_ros_stamp
            depth_msg.header.frame_id = WRIST_FRAME_ID
            depth_pub.publish(depth_msg)

            if show_window:
                cv2.imshow("wrist_detection", canvas)
                key = cv2.waitKey(1)
                if key & 0xFF == ord("q") or key == 27:
                    break

            rate.sleep()

    except KeyboardInterrupt:
        rospy.loginfo("Shutting down wrist object detector")

    except Exception as exc:
        rospy.logerr("wrist object detector failed: %s", str(exc))
        raise

    finally:
        if pipeline is not None:
            pipeline.stop()
        if show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
