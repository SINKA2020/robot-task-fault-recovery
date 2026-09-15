#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from scene_graph_system.resources import package_resource_path
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml


DEFAULT_SERIAL = "313522071466"


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate global_table_frame from external RealSense table plane.")

    parser.add_argument("--serial", type=str, default=DEFAULT_SERIAL)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)

    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        default=[30, 160, 270, 260],  # [左，上，右，下],
        help="Table ROI in pixels: u_min v_min u_max v_max",
    )

    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--min_depth", type=float, default=0.2)
    parser.add_argument("--max_depth", type=float, default=2.0)

    parser.add_argument(
        "--output",
        type=str,
        default=package_resource_path('config/global_table_frame.yaml'),
    )

    parser.add_argument(
        "--flip_normal",
        action="store_true",
        help="Flip table z direction manually if object height becomes negative.",
    )

    return parser.parse_args()


def create_pipeline(serial, width, height, fps):
    pipeline = rs.pipeline()
    config = rs.config()

    if serial:
        config.enable_device(str(serial))
        print(f"[INFO] Using RealSense serial: {serial}")

    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    print(f"[INFO] Device name: {device.get_info(rs.camera_info.name)}")
    print(f"[INFO] Device serial: {device.get_info(rs.camera_info.serial_number)}")
    print(f"[INFO] Depth scale: {depth_scale}")

    return pipeline, align, depth_scale


def get_aligned_images(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)

    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()

    if not depth_frame or not color_frame:
        return None, None, None

    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    return color_image, depth_image, depth_intrin


def deproject_depth_roi_to_points(
    depth_image,
    depth_intrin,
    roi,
    depth_scale,
    min_depth,
    max_depth,
    stride,
):
    u_min, v_min, u_max, v_max = roi
    h, w = depth_image.shape[:2]

    u_min = max(0, min(u_min, w - 1))
    u_max = max(0, min(u_max, w - 1))
    v_min = max(0, min(v_min, h - 1))
    v_max = max(0, min(v_max, h - 1))

    points = []

    for v in range(v_min, v_max, stride):
        for u in range(u_min, u_max, stride):
            z = float(depth_image[v, u]) * float(depth_scale)
            if z <= min_depth or z >= max_depth:
                continue

            p = rs.rs2_deproject_pixel_to_point(
                depth_intrin,
                [float(u), float(v)],
                float(z),
            )
            points.append(p)

    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)

    return np.asarray(points, dtype=np.float64)


def fit_plane_pca(points):
    """
    用 PCA 拟合桌面平面。
    返回:
      origin: 桌面中心点，camera frame 下坐标
      normal: 桌面法向量，camera frame 下方向
      rmse: 点到平面的均方根误差
    """
    points = np.asarray(points, dtype=np.float64)

    if len(points) < 30:
        raise RuntimeError(f"Not enough table points: {len(points)}")

    origin = points.mean(axis=0)
    centered = points - origin

    _, _, vh = np.linalg.svd(centered, full_matrices=False)

    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)

    distances = centered @ normal
    rmse = float(np.sqrt(np.mean(distances ** 2)))

    return origin, normal, rmse


def build_table_transform(origin_camera, normal_camera, flip_normal=False):
    """
    构造 T_table_camera，使:
      p_table = T_table_camera @ p_camera_h

    table_z 取桌面法向量方向。
    默认假设相机在桌面上方，桌面向上的方向大致朝向相机，也就是 -camera_z。
    """
    origin_camera = np.asarray(origin_camera, dtype=np.float64)
    z_axis = np.asarray(normal_camera, dtype=np.float64)
    z_axis = z_axis / np.linalg.norm(z_axis)

    camera_forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # 让 table_z 大致朝向相机一侧。
    # 如果后续测试发现物体高度为负，可用 --flip_normal 反转。
    if np.dot(z_axis, camera_forward) > 0:
        z_axis = -z_axis

    if flip_normal:
        z_axis = -z_axis

    # 将相机 x 轴投影到桌面平面，作为 table_x
    camera_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = camera_x - np.dot(camera_x, z_axis) * z_axis

    if np.linalg.norm(x_axis) < 1e-6:
        camera_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = camera_y - np.dot(camera_y, z_axis) * z_axis

    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # 再正交化一次
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    R_table_camera = np.vstack([x_axis, y_axis, z_axis])
    t_table_camera = -R_table_camera @ origin_camera

    T_table_camera = np.eye(4, dtype=np.float64)
    T_table_camera[:3, :3] = R_table_camera
    T_table_camera[:3, 3] = t_table_camera

    return T_table_camera


def save_yaml(output_path, T_table_camera, origin_camera, normal_camera, roi, rmse):
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    data = {
        "frame_name": "global_table_frame",
        "source_frame": "global_camera_color_optical_frame",
        "note": "p_table = T_table_camera @ p_camera_h",
        "roi": [int(x) for x in roi],
        "plane_origin_camera": [float(x) for x in origin_camera],
        "plane_normal_camera": [float(x) for x in normal_camera],
        "plane_rmse_m": float(rmse),
        "T_table_camera": T_table_camera.tolist(),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"[SAVE] {output_path}")


def draw_roi(image, roi):
    u_min, v_min, u_max, v_max = roi
    canvas = image.copy()
    cv2.rectangle(canvas, (u_min, v_min), (u_max, v_max), (0, 255, 0), 2)
    cv2.putText(
        canvas,
        "ROI must cover clean empty table. Press C to calibrate, Q/Esc to quit.",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )
    return canvas


def main():
    args = parse_args()

    pipeline = None

    try:
        pipeline, align, depth_scale = create_pipeline(
            args.serial,
            args.width,
            args.height,
            args.fps,
        )

        # 预热
        for _ in range(15):
            get_aligned_images(pipeline, align)

        window_name = "Calibrate global_table_frame"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("=" * 80)
        print("[INFO] 请先清空桌面 ROI 区域，保证绿色框内是干净桌面。")
        print("[INFO] 按 c 执行桌面平面标定。")
        print("[INFO] 按 q 或 Esc 退出。")
        print(f"[INFO] ROI = {args.roi}")
        print(f"[INFO] Output = {args.output}")
        print("=" * 80)

        while True:
            color_image, depth_image, depth_intrin = get_aligned_images(pipeline, align)

            if color_image is None:
                print("[WARN] Empty frame")
                continue

            display = draw_roi(color_image, args.roi)
            cv2.imshow(window_name, display)

            key = cv2.waitKeyEx(30)
            key_low = key & 0xFF

            if key_low in [ord("q"), ord("Q"), 27]:
                print("[INFO] Exit")
                break

            if key_low in [ord("c"), ord("C")]:
                points = deproject_depth_roi_to_points(
                    depth_image=depth_image,
                    depth_intrin=depth_intrin,
                    roi=args.roi,
                    depth_scale=depth_scale,
                    min_depth=args.min_depth,
                    max_depth=args.max_depth,
                    stride=args.stride,
                )

                print(f"[INFO] Sampled table points: {len(points)}")

                origin, normal, rmse = fit_plane_pca(points)
                T_table_camera = build_table_transform(
                    origin,
                    normal,
                    flip_normal=args.flip_normal,
                )

                print("[RESULT] plane_origin_camera =", origin)
                print("[RESULT] plane_normal_camera =", normal)
                print("[RESULT] plane_rmse_m =", rmse)
                print("[RESULT] T_table_camera =")
                print(T_table_camera)

                save_yaml(
                    args.output,
                    T_table_camera,
                    origin,
                    normal,
                    args.roi,
                    rmse,
                )

                print("[INFO] 标定完成。建议下一步放一个积木测试 table_z 是否为正。")
                break

    finally:
        if pipeline is not None:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()