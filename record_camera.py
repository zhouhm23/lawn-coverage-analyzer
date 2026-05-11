#!/usr/bin/env python3
"""
record_camera.py — 俯视摄像头录屏脚本，支持两种模式:

模式一：先录制再处理 (--record)
    python record_camera.py --record --output demo.mp4
    python record_camera.py --record --output demo.mp4 --mask mask.png --analyze --visualize

模式二：边录制边显示覆盖率 (--live)
    python record_camera.py --live                                    # 全区域可通行
    python record_camera.py --live --mask mask.png                    # 文件蒙版
    python record_camera.py --live --color-lower 35,50,50 --color-upper 85,255,255  # 颜色蒙版

依赖: pip install opencv-python numpy matplotlib
"""

import argparse
import os
import sys
import time
import csv
from collections import deque
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── PIL 用于中文渲染 ────────────────────────────────────────────────
try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# ── 中文字体查找 ────────────────────────────────────────────────────
_CHINESE_FONT = None
if _HAS_PIL:
    _FONT_CANDIDATES = [
        "C:/Windows/Fonts/simhei.ttf",       # 黑体
        "C:/Windows/Fonts/msyh.ttc",          # 微软雅黑
        "C:/Windows/Fonts/simsun.ttc",        # 宋体
        "C:/Windows/Fonts/simkai.ttf",        # 楷体
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Linux
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/System/Library/Fonts/PingFang.ttc",  # macOS
    ]
    for _fp in _FONT_CANDIDATES:
        if os.path.exists(_fp):
            _CHINESE_FONT = _fp
            break


def _put_chinese_text(img: np.ndarray, text: str, pos: tuple,
                      font_size: int = 20, color: tuple = (0, 255, 0),
                      thickness: int = 2) -> np.ndarray:
    """在 OpenCV 图像上绘制中文文本（PIL 回退）。

    如果 PIL 不可用或没有中文字体，回退到 cv2.putText（仅英文/数字）。
    """
    if not _HAS_PIL or _CHINESE_FONT is None:
        # 回退：只显示 ASCII 字符
        ascii_text = ''.join(c if ord(c) < 128 else '?' for c in text)
        cv2.putText(img, ascii_text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, thickness)
        return img

    try:
        font = ImageFont.truetype(_CHINESE_FONT, font_size)
        # BGR → RGB → PIL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        # PIL 颜色是 RGB 元组
        pil_color = (color[2], color[1], color[0])
        draw.text(pos, text, font=font, fill=pil_color)
        # 转回 BGR
        img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        ascii_text = ''.join(c if ord(c) < 128 else '?' for c in text)
        cv2.putText(img, ascii_text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, thickness)
    return img

# ── 导入 camera_coverage 核心模块 ────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from camera_coverage import (
    CameraCoverageConfig,
    CameraCoverageAnalyzer,
    detect_aruco_markers,
    detect_corner_markers,
    detect_robot_marker,
    compute_homography,
    image_to_paper,
    load_mask,
    mask_to_passable_grid,
    create_mask_from_color,
    create_all_passable_mask,
    generate_overlay_image,
)

# ═══════════════════════════════════════════════════════════════════════════
# 公共参数
# ═══════════════════════════════════════════════════════════════════════════

CAMERA_INDEX = 1
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
FPS = 30.0
SCREEN_W, SCREEN_H = 1920, 1080


# ═══════════════════════════════════════════════════════════════════════════
# 模式一：先录制再处理
# ═══════════════════════════════════════════════════════════════════════════

def record_only(output_path: str, camera_index: int = CAMERA_INDEX):
    """录制视频到文件，按 q 停止。返回录制是否成功。"""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"无法打开摄像头 (index={camera_index})")
        return False

    # 获取摄像头分辨率
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头分辨率: {cam_w}×{cam_h}")

    writer = cv2.VideoWriter(output_path, FOURCC, FPS, (cam_w, cam_h))
    if not writer.isOpened():
        print(f"无法创建视频文件: {output_path}")
        cap.release()
        return False

    print(f"录制中... 按 'q' 停止 → {output_path}")
    frame_count = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ 摄像头帧读取失败")
            break

        writer.write(frame)
        frame_count += 1

        # 显示录制画面（缩放到屏幕）
        disp = _resize_to_screen(frame)
        cv2.imshow("Recording — 按 'q' 停止", disp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    elapsed = time.time() - t_start
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # 打印录制信息
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n录制完成: {frame_count} 帧, {elapsed:.1f}s, "
          f"{file_size_mb:.1f} MB")
    print(f"实际帧率: {frame_count / max(elapsed, 0.01):.1f} fps")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# 模式二：边录制边显示覆盖率
# ═══════════════════════════════════════════════════════════════════════════

class LiveCoverageRecorder:
    """实时录制并在画面右下角叠加覆盖率信息。"""

    def __init__(self, config: CameraCoverageConfig, interpolate: bool = False):
        self.config = config
        self.interpolate = interpolate  # ArUco 丢失时是否线性插值
        self._homography: Optional[np.ndarray] = None
        self._passable_mask: Optional[np.ndarray] = None
        self._covered_count: Optional[np.ndarray] = None
        self._total_passable: int = 0
        self._trajectory_points: int = 0
        self._trajectory_len: float = 0.0
        self._last_pt: Optional[Tuple[float, float]] = None
        self._rad_cells: int = 0

        # 工作计时器：从首次检测到机器人开始计时
        self._work_start_time: Optional[float] = None
        self._work_elapsed: float = 0.0

        # ArUco 插值状态
        self._last_valid_pt: Optional[Tuple[float, float]] = None
        self._lost_frames_count: int = 0
        self._max_interp_frames: int = 5  # 最多插值帧数

        # 论文坐标轨迹（用于导出叠加图）
        self._paper_trajectory: List[Tuple[float, float, float]] = []
        self._bg_frame: Optional[np.ndarray] = None  # 首帧（用于叠加图背景）

        # 校准状态
        self._calibrated = False
        self._calib_buffer: Deque[Dict[int, Tuple[float, float]]] = deque(maxlen=config.calib_frames)
        self._lost_consecutive: int = 0
        self._hud_text: List[str] = []

    def calibrate(self, frame: np.ndarray) -> bool:
        """从当前帧检测角点 ArUco 并累积标定缓冲区。

        需要连续 calib_frames 帧全部检测到 4 角，然后取平均坐标计算单应矩阵。
        这样即使单帧抖动，最终单应矩阵也是稳定的。

        Returns:
            True 表示标定完成（此后不再需要调用此方法）
        """
        corners = detect_corner_markers(frame, self.config)
        if corners is None:
            self._calib_buffer.clear()  # 丢失一帧就清空缓冲，强制连续稳定
            return False

        self._calib_buffer.append(corners)

        if len(self._calib_buffer) < self.config.calib_frames:
            return False  # 缓冲未满

        # 取平均
        avg_corners: Dict[int, Tuple[float, float]] = {}
        for cid in self.config.corner_ids:
            xs = [self._calib_buffer[i][cid][0] for i in range(len(self._calib_buffer))]
            ys = [self._calib_buffer[i][cid][1] for i in range(len(self._calib_buffer))]
            avg_corners[cid] = (float(np.mean(xs)), float(np.mean(ys)))

        img_pts = [avg_corners[cid] for cid in self.config.corner_ids]
        paper_pts = list(self.config.corner_paper_xy)
        self._homography = compute_homography(img_pts, paper_pts)

        for cid in self.config.corner_ids:
            std_x = float(np.std([self._calib_buffer[i][cid][0] for i in range(len(self._calib_buffer))]))
            std_y = float(np.std([self._calib_buffer[i][cid][1] for i in range(len(self._calib_buffer))]))
            print(f"  ✓ 角点 ID={cid}: avg=({avg_corners[cid][0]:.1f},"
                  f"{avg_corners[cid][1]:.1f})  std=({std_x:.2f},{std_y:.2f})px")

        print(f"✓ 单应矩阵标定成功 (基于 {self.config.calib_frames} 帧平均)")
        self._calibrated = True
        self._calib_buffer.clear()  # 释放内存
        return True

    def init_grid(self, mask_img: np.ndarray):
        """初始化覆盖网格。"""
        cfg = self.config
        self._passable_mask = mask_to_passable_grid(mask_img, self._homography, cfg)
        self._covered_count = np.zeros(self._passable_mask.shape, dtype=np.int32)
        self._total_passable = int(np.sum(self._passable_mask))
        self._rad_cells = int(np.ceil(cfg.coverage_radius / cfg.resolution))
        print(f"✓ 覆盖网格初始化: {cfg.grid_w}×{cfg.grid_h}, "
              f"可通行 {self._total_passable} 格 "
              f"(≈ {self._total_passable * cfg.resolution**2:.3f} m²)")

    def update(self, t_sec: float, robot_pt: Optional[Tuple[float, float]]):
        """用当前帧的机器人位置更新覆盖。

        支持 ArUco 丢失时线性插值（如果 self.interpolate=True）。
        """
        # ── 插值路径：ArUco 丢失但启用插值 ─────────────────────────
        if robot_pt is None and self.interpolate and self._last_valid_pt is not None:
            self._lost_frames_count += 1
            if self._lost_frames_count <= self._max_interp_frames:
                # 复用上一个有效位置（零阶保持）
                robot_pt = self._last_valid_pt
            else:
                self._lost_consecutive += 1
                self._hud_text = ["⚠ ArUco 丢失 (插值超限)"]
                return
        elif robot_pt is None:
            self._lost_consecutive += 1
            self._hud_text = ["⚠ ArUco 丢失"]
            return
        else:
            self._lost_frames_count = 0

        self._lost_consecutive = 0

        # ── 工作计时器：首次检测到机器人时启动 ────────────────────
        if self._work_start_time is None:
            self._work_start_time = t_sec

        paper_pt = image_to_paper(robot_pt, self._homography)
        self._last_valid_pt = robot_pt
        cfg = self.config

        # 范围检查
        if not (-0.1 <= paper_pt[0] <= cfg.paper_width + 0.1 and
                -0.1 <= paper_pt[1] <= cfg.paper_height + 0.1):
            self._hud_text = ["⚠ 坐标超出范围"]
            return

        px, py = paper_pt
        cx = int(px / cfg.resolution)
        cy = int(py / cfg.resolution)
        x0 = max(cx - self._rad_cells, 0)
        x1 = min(cx + self._rad_cells + 1, cfg.grid_w)
        y0 = max(cy - self._rad_cells, 0)
        y1 = min(cy + self._rad_cells + 1, cfg.grid_h)

        if x0 < x1 and y0 < y1:
            xs = (np.arange(x0, x1, dtype=np.float64) + 0.5) * cfg.resolution
            ys = (np.arange(y0, y1, dtype=np.float64) + 0.5) * cfg.resolution
            X, Y = np.meshgrid(xs, ys)
            circle = (X - px) ** 2 + (Y - py) ** 2 <= (cfg.coverage_radius ** 2)
            self._covered_count[y0:y1, x0:x1] += circle.astype(np.int32)

        self._trajectory_points += 1
        if self._last_pt is not None:
            d = np.hypot(px - self._last_pt[0], py - self._last_pt[1])
            self._trajectory_len += d
        self._last_pt = paper_pt

        # 记录论文坐标轨迹（用于导出叠加图，每 3 帧存一次以节省内存）
        if self._trajectory_points % 3 == 0:
            self._paper_trajectory.append((0.0, px, py))

        # ── 计算实时指标（条带面积法）─────────────────────
        # strip_area = trajectory_len × strip_width
        # unique_area = N_total × resolution²
        # repeat_coverage = (strip_area - unique_area) / strip_area
        res2 = cfg.resolution ** 2
        strip_w = 2.0 * cfg.coverage_radius
        if self._total_passable > 0:
            N_total = int(np.sum((self._covered_count > 0) & self._passable_mask))
            area_cov = N_total / self._total_passable

            unique_a = N_total * res2
            strip_a = self._trajectory_len * strip_w
            rep_cov = max(0.0, (strip_a - unique_a) / strip_a) if strip_a > 1e-9 else 0.0

            eff = area_cov / max(self._trajectory_len, 0.01)
        else:
            area_cov = rep_cov = eff = 0.0

        # 工作时间
        work_sec = t_sec - (self._work_start_time or t_sec)

        self._hud_text = [
            f"区域覆盖率: {area_cov*100:5.1f}%",
            f"重复覆盖率: {rep_cov*100:5.1f}%",
            f"覆盖效率:   {eff:.3f} m-1",
            f"轨迹长度:   {self._trajectory_len:.2f} m",
            f"轨迹点数:   {self._trajectory_points}",
            f"工作时间:   {work_sec:.1f} s",
        ]
        if self.interpolate and self._lost_frames_count > 0:
            self._hud_text.append(f"插值续接:   {self._lost_frames_count} 帧")

    def draw_hud(self, frame: np.ndarray) -> np.ndarray:
        """在画面上叠加 HUD 文本面板（支持中文）。"""
        overlay = frame.copy()
        panel_w, panel_h = 340, 20 + len(self._hud_text) * 28
        margin = 20
        x0 = frame.shape[1] - panel_w - margin
        y0 = frame.shape[0] - panel_h - margin

        # 半透明背景
        cv2.rectangle(overlay, (x0, y0),
                      (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # 中文文本
        for i, txt in enumerate(self._hud_text):
            _put_chinese_text(frame, txt, (x0 + 12, y0 + 16 + i * 28),
                              font_size=18, color=(0, 255, 0), thickness=2)

        # 状态指示
        status = "CALIBRATED" if self._calibrated else "NO CALIB"
        color = (0, 255, 0) if self._calibrated else (0, 0, 255)
        cv2.putText(frame, status, (margin, margin + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame


def live_coverage(config: CameraCoverageConfig,
                  camera_index: int = CAMERA_INDEX,
                  output_video: Optional[str] = None,
                  mask_path: Optional[str] = None,
                  color_lower: Optional[Tuple[int, int, int]] = None,
                  color_upper: Optional[Tuple[int, int, int]] = None,
                  interpolate: bool = False):
    """边录制边显示覆盖率。

    蒙版优先级: mask_path > color_range > 全矩形可通行
    """
    # 1. 获取蒙版（稍后在标定时从首帧生成）
    # 如果是文件蒙版，先加载；颜色蒙版或全通行在标定后生成
    mask_img: Optional[np.ndarray] = None
    if mask_path:
        print(f"[1/3] 加载蒙版文件: {mask_path}")
        mask_img = load_mask(mask_path, config.mask_threshold)
    elif color_lower and color_upper:
        print(f"[1/3] 颜色蒙版模式: HSV[{color_lower} ~ {color_upper}]")
        mask_img = None  # 标定时从首帧生成
    else:
        print(f"[1/3] 全区域可通行模式")

    # 2. 打开摄像头
    print(f"[2/3] 打开摄像头 (index={camera_index})...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"无法打开摄像头 (index={camera_index})")
        sys.exit(1)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"      分辨率: {cam_w}×{cam_h}")

    # 可选录制输出
    writer = None
    if output_video:
        writer = cv2.VideoWriter(output_video, FOURCC, FPS, (cam_w, cam_h))
        if not writer.isOpened():
            print(f"⚠ 无法创建输出视频: {output_video}，将跳过录制")
            writer = None
        else:
            print(f"      输出视频: {output_video}")

    # 3. 初始化实时覆盖记录器
    print(f"[3/3] 实时覆盖模式启动...")
    recorder = LiveCoverageRecorder(config, interpolate=interpolate)

    aruco_dict = cv2.aruco.getPredefinedDictionary(config.aruco_dict)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    print(f"等待 ArUco 角点稳定标定 (ID=0,1,2,3, 需连续 {config.calib_frames} 帧)... 按 'q' 直接退出 / 'f' 完成任务")
    calibration_attempts = 0
    frame_idx = 0
    t_start = time.time()
    exit_mode = "quit"  # "quit" 或 "finish"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ 摄像头帧读取失败")
            break

        t_sec = time.time() - t_start

        if not recorder._calibrated:
            # 标定阶段
            calibration_attempts += 1
            if recorder.calibrate(frame):
                # 确定蒙版来源
                if mask_img is None and color_lower and color_upper:
                    mask_img = create_mask_from_color(frame, color_lower, color_upper)
                    print(f"      颜色蒙版生成: 可通行={np.sum(mask_img>=128)} px")
                elif mask_img is None:
                    mask_img = create_all_passable_mask(frame.shape)
                recorder.init_grid(mask_img)
                # 保存标定完成时的帧作为叠加图背景
                recorder._bg_frame = frame.copy()
                print("开始实时覆盖分析...")

            # 显示标定状态
            if writer is not None:
                writer.write(frame.copy())
            disp = _resize_to_screen(frame)
            buf_len = len(recorder._calib_buffer) if hasattr(recorder, '_calib_buffer') else 0
            total = recorder.config.calib_frames
            _put_chinese_text(disp,
                f"CALIBRATING [{buf_len}/{total}]  (attempt {calibration_attempts})",
                (20, 20), font_size=22, color=(0, 255, 255), thickness=2)
            cv2.imshow("Live Coverage — 按 'q' 退出 / 'f' 完成任务", disp)
        else:
            # 检测机器人 ArUco
            corners, ids, _ = aruco_detector.detectMarkers(frame)
            robot_pt = None
            if ids is not None:
                for i, mid in enumerate(ids.flatten()):
                    if int(mid) == config.robot_id:
                        c = corners[i][0]
                        robot_pt = (float(np.mean(c[:, 0])),
                                    float(np.mean(c[:, 1])))
                        break

            # 更新覆盖
            recorder.update(t_sec, robot_pt)

            # ── 录制原始帧（在标注前，保留给离线分析）──────────
            if writer is not None:
                writer.write(frame.copy())

            # 画 ArUco 标注
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # 绘制 HUD（仅在显示画面上叠加）
            frame = recorder.draw_hud(frame)

            # 显示
            disp = _resize_to_screen(frame)
            cv2.imshow("Live Coverage — 按 'q' 退出 / 'f' 完成任务", disp)

        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            exit_mode = "quit"
            break
        elif key == ord('f'):
            exit_mode = "finish"
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - t_start

    # ── 按 f 完成任务：视频加时间后缀 + 保存临时结果 + 导出叠加图 ─
    if exit_mode == "finish" and output_video:
        ts = datetime.now().strftime("%m%d%H%M")
        dirname = os.path.dirname(output_video) or "."
        basename = os.path.splitext(os.path.basename(output_video))[0]

        # 视频加时间后缀
        if os.path.exists(output_video):
            stamped_path = os.path.join(dirname, f"{basename}{ts}.mp4")
            os.rename(output_video, stamped_path)
            print(f"\n✓ 视频已保存: {stamped_path}")
        else:
            stamped_path = output_video

        # 保存临时测试结果
        _save_temp_results(recorder, elapsed, frame_idx, stamped_path)

        # 导出论文叠加图
        if recorder._bg_frame is not None and recorder._homography is not None \
                and recorder._covered_count is not None and recorder._passable_mask is not None \
                and len(recorder._paper_trajectory) > 1:
            overlay_path = os.path.join(dirname, f"{basename}{ts}_overlay.png")
            generate_overlay_image(
                bg_frame=recorder._bg_frame.copy(),
                homography=recorder._homography,
                trajectory=recorder._paper_trajectory,
                covered_count=recorder._covered_count,
                passable_mask=recorder._passable_mask,
                config=recorder.config,
                output_path=overlay_path,
                traj_len=recorder._trajectory_len,
            )

    # 打印最终统计
    _print_live_summary(recorder, elapsed, frame_idx, exit_mode)


def _print_live_summary(recorder: LiveCoverageRecorder, elapsed: float,
                        total_frames: int, exit_mode: str = "quit"):
    """打印实时覆盖的最终汇总。"""
    work_sec = recorder._work_elapsed
    if recorder._work_start_time is not None:
        work_sec = elapsed - recorder._work_start_time
    print(f"\n{'='*50}")
    mode_label = "任务完成" if exit_mode == "finish" else "手动退出"
    print(f"  实时覆盖录制汇总 ({mode_label})")
    print(f"{'='*50}")
    print(f"  总录制时长:   {elapsed:.1f} s")
    if recorder._work_start_time is not None:
        print(f"  有效工作时长: {work_sec:.1f} s")
    print(f"  总帧数:       {total_frames}")
    print(f"  有效轨迹点:   {recorder._trajectory_points}")
    if recorder._total_passable > 0:
        cfg = recorder.config
        N_total = int(np.sum((recorder._covered_count > 0) & recorder._passable_mask))
        res2 = cfg.resolution ** 2
        strip_w = 2.0 * cfg.coverage_radius
        unique_a = N_total * res2
        strip_a = recorder._trajectory_len * strip_w
        rep_cov = max(0.0, (strip_a - unique_a) / strip_a) if strip_a > 1e-9 else 0.0
        print(f"  区域覆盖率:   {N_total/recorder._total_passable*100:.1f}%")
        print(f"  重复覆盖率:   {rep_cov*100:.1f}%  "
              f"(条带={strip_a:.4f}m², 唯一覆盖={unique_a:.4f}m²)")
        print(f"  轨迹长度:     {recorder._trajectory_len:.2f} m")
    print(f"{'='*50}")


def _save_temp_results(recorder: LiveCoverageRecorder, elapsed: float,
                       total_frames: int, video_path: str):
    """按 f 完成任务时保存临时测试结果（不覆盖覆盖率实验.csv）。

    生成 temp_result_MMDDHHMM.csv，用户手动确认后自行写入覆盖率实验.csv。
    """
    ts = datetime.now().strftime("%m%d%H%M")
    temp_path = f"./temp_result_{ts}.csv"

    work_sec = elapsed
    if recorder._work_start_time is not None:
        work_sec = elapsed - recorder._work_start_time

    N_total = 0
    rep_cov = 0.0
    strip_a = 0.0
    unique_a = 0.0
    if recorder._total_passable > 0:
        cfg = recorder.config
        res2 = cfg.resolution ** 2
        strip_w = 2.0 * cfg.coverage_radius
        N_total = int(np.sum((recorder._covered_count > 0) & recorder._passable_mask))
        unique_a = N_total * res2
        strip_a = recorder._trajectory_len * strip_w
        rep_cov = max(0.0, (strip_a - unique_a) / strip_a) if strip_a > 1e-9 else 0.0

    with open(temp_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(["指标", "值", "备注"])
        w.writerow(["视频文件", video_path, ""])
        w.writerow(["总录制时长 (s)", f"{elapsed:.1f}", ""])
        w.writerow(["有效工作时长 (s)", f"{work_sec:.1f}", ""])
        w.writerow(["总帧数", total_frames, ""])
        w.writerow(["有效轨迹点", recorder._trajectory_points, ""])
        w.writerow(["区域覆盖率", f"{N_total/recorder._total_passable*100:.1f}%",
                     f"已覆盖{N_total}/{recorder._total_passable}格"])
        w.writerow(["重复覆盖率", f"{rep_cov*100:.1f}%",
                     f"条带={strip_a:.4f}m², 唯一覆盖={unique_a:.4f}m²"])
        w.writerow(["轨迹长度 (m)", f"{recorder._trajectory_len:.2f}", ""])
        w.writerow(["割草宽度 (m)", f"{2.0*recorder.config.coverage_radius:.3f}", ""])
        w.writerow(["覆盖半径 (m)", f"{recorder.config.coverage_radius:.3f}", ""])
        w.writerow(["网格分辨率 (m)", f"{recorder.config.resolution:.4f}", ""])

    print(f"✓ 临时结果已保存: {temp_path}")
    print(f"  请确认后手动写入 覆盖率实验.csv")


# ═══════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════

def _resize_to_screen(img: np.ndarray,
                      max_w: int = SCREEN_W,
                      max_h: int = SCREEN_H) -> np.ndarray:
    """按比例缩放图像以适应屏幕。"""
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h) * 0.85
    if scale < 1.0:
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


# ═══════════════════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="俯视摄像头录屏 — 支持先录制再处理 / 边录制边显示覆盖率")

    # 模式选择（二选一）
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--record", action="store_true",
                            help="先录制视频再离线处理")
    mode_group.add_argument("--live", action="store_true",
                            help="边录制边实时显示覆盖率")

    # 通用参数
    parser.add_argument("--output", default="./camera_record.mp4",
                        help="录制输出视频路径 (默认: ./camera_record.mp4)")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX,
                        help=f"摄像头索引 (默认: {CAMERA_INDEX})")
    parser.add_argument("--mask", default=None,
                        help="蒙版 PNG 路径（白色=可通行）。不传则用 --color-* 或全区域")

    # 颜色蒙版参数（替代 --mask，用颜色提取地板）
    parser.add_argument("--color-lower", default=None,
                        help="HSV 下界，逗号分隔，如 '35,50,50'（绿色地板）")
    parser.add_argument("--color-upper", default=None,
                        help="HSV 上界，逗号分隔，如 '85,255,255'")

    # 覆盖参数
    parser.add_argument("--coverage-radius", type=float, default=0.085,
                        help="覆盖半径 m (默认: 0.085)")
    parser.add_argument("--resolution", type=float, default=0.005,
                        help="网格分辨率 m (默认: 0.005)")
    parser.add_argument("--robot-id", type=int, default=4,
                        help="车顶 ArUco ID (默认: 4)")
    parser.add_argument("--calib-frames", type=int, default=3,
                        help="标定稳定帧数，越大越抗抖动但需等越久 (默认: 3)")
    parser.add_argument("--interpolate", action="store_true",
                        help="ArUco 短暂丢失时线性插值连接轨迹")
    parser.add_argument("--min-movement", type=float, default=0.01,
                        help="最小移动距离阈值 m，低于此值视为静止抖动 (默认: 0.01)")

    # 离线分析参数（--record 模式下可选）
    parser.add_argument("--analyze", action="store_true",
                        help="录制完成后自动运行离线覆盖分析")
    parser.add_argument("--visualize", action="store_true",
                        help="生成可视化图像（需要 --analyze）")
    parser.add_argument("--results-dir", default="./coverage_results",
                        help="分析结果输出目录 (默认: ./coverage_results)")

    args = parser.parse_args()

    # 解析颜色参数
    color_lower = None
    color_upper = None
    if args.color_lower and args.color_upper:
        color_lower = tuple(int(x) for x in args.color_lower.split(","))
        color_upper = tuple(int(x) for x in args.color_upper.split(","))
        if len(color_lower) != 3 or len(color_upper) != 3:
            print("错误: --color-lower/--color-upper 需要 3 个逗号分隔的整数")
            sys.exit(1)

    # ── 模式一：先录制 ──
    if args.record:
        print("=" * 50)
        print("  模式一: 先录制再处理")
        print("=" * 50)

        if not record_only(args.output, args.camera):
            sys.exit(1)

        if args.analyze:
            if not args.mask:
                print("错误: --analyze 需要 --mask 参数")
                sys.exit(1)
            if not os.path.exists(args.mask):
                print(f"错误: 蒙版文件不存在: {args.mask}")
                sys.exit(1)

            print(f"\n开始离线覆盖分析: {args.output}")
            config = CameraCoverageConfig(
                coverage_radius=args.coverage_radius,
                resolution=args.resolution,
                robot_id=args.robot_id,
                calib_frames=args.calib_frames,
            )
            analyzer = CameraCoverageAnalyzer(config)
            try:
                analyzer.analyze(args.output, args.mask)
            except Exception as e:
                print(f"分析失败: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

            prefix = os.path.splitext(os.path.basename(args.output))[0]
            analyzer.save_results(args.results_dir, prefix)

            if args.visualize:
                print("生成可视化...")
                analyzer.generate_visualizations(args.results_dir, prefix)

    # ── 模式二：边录制边覆盖 ──
    elif args.live:
        if args.mask and not os.path.exists(args.mask):
            print(f"错误: 蒙版文件不存在: {args.mask}")
            sys.exit(1)

        print("=" * 50)
        print("  模式二: 边录制边显示覆盖率")
        print("=" * 50)

        config = CameraCoverageConfig(
            coverage_radius=args.coverage_radius,
            resolution=args.resolution,
            robot_id=args.robot_id,
            calib_frames=args.calib_frames,
        )
        live_coverage(
            config=config,
            camera_index=args.camera,
            output_video=args.output,
            mask_path=args.mask,
            color_lower=color_lower,
            color_upper=color_upper,
            interpolate=args.interpolate,
        )

    print("\n完成 ✓")


if __name__ == "__main__":
    main()
