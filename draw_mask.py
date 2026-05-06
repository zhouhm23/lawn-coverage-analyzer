#!/usr/bin/env python3
"""
draw_mask.py — 鼠标勾画多边形生成蒙版，替代 PS 手动制作。

用法:
    python draw_mask.py                          # 用摄像头首帧勾画
    python draw_mask.py --image scene.png        # 用图片勾画

操作:
    鼠标左键 — 添加多边形顶点
    鼠标右键 — 删除最后一个顶点
    r       — 清空所有顶点
    Enter   — 确认并保存蒙版
    q/ESC   — 退出

输出: mask.png (白色=可通行，黑色=障碍)
"""

import argparse
import sys
import cv2
import numpy as np

# 全局鼠标回调状态
_polygon_pts: list = []
_frame: np.ndarray = None
_display: np.ndarray = None
_window_name = "Draw Mask — 左键加点 | 右键删点 | Enter 保存 | r 清空 | q 退出"


def _mouse_callback(event, x, y, flags, param):
    """鼠标回调：左键加点，右键删点。"""
    global _polygon_pts, _display
    if event == cv2.EVENT_LBUTTONDOWN:
        _polygon_pts.append((x, y))
        _redraw()
    elif event == cv2.EVENT_RBUTTONDOWN:
        if _polygon_pts:
            _polygon_pts.pop()
            _redraw()


def _redraw():
    """重绘画面（底图 + 已有点 + 连线）。"""
    global _polygon_pts, _display, _frame
    _display = _frame.copy()
    n = len(_polygon_pts)
    for i, pt in enumerate(_polygon_pts):
        cv2.circle(_display, pt, 5, (0, 255, 0), -1)
        if i > 0:
            cv2.line(_display, _polygon_pts[i - 1], pt, (0, 255, 0), 2)
    if n >= 2:
        # 首尾连线暗示闭合
        cv2.line(_display, _polygon_pts[-1], _polygon_pts[0], (0, 255, 255), 1)
    # HUD
    cv2.putText(_display, f"Points: {n}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(_display, "L=add  R=del  r=reset  Enter=save  q=quit",
                (12, _display.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.imshow(_window_name, _display)


def _generate_mask(shape: tuple, pts: list) -> np.ndarray:
    """从多边形顶点生成二值蒙版 (H,W) uint8。"""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if len(pts) >= 3:
        poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [poly], 255)
    return mask


def _resize_to_screen(img, max_w=1920, max_h=1080):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h) * 0.85
    if scale < 1.0:
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def main():
    parser = argparse.ArgumentParser(
        description="鼠标勾画多边形，生成可通行区域蒙版")
    parser.add_argument("--image", default=None,
                        help="用图片勾画（不传则用摄像头首帧）")
    parser.add_argument("--output", default="mask.png",
                        help="输出蒙版路径 (默认: mask.png)")
    parser.add_argument("--camera", type=int, default=1,
                        help="摄像头索引 (默认: 1)")
    args = parser.parse_args()

    # ── 获取底图 ──
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"无法读取图片: {args.image}")
            sys.exit(1)
        print(f"底图: {args.image} ({frame.shape[1]}×{frame.shape[0]})")
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"无法打开摄像头 (index={args.camera})")
            sys.exit(1)
        # 实时预览直到用户按 Enter 截图
        print("摄像头预览中，按 Enter 截图开始勾画，按 q 退出...")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            disp = _resize_to_screen(frame)
            cv2.putText(disp, "Press Enter to capture, q to quit",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Camera Preview", disp)
            key = cv2.waitKey(30) & 0xFF
            if key == 13 or key == ord('\r'):  # Enter
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)
        cap.release()
        cv2.destroyWindow("Camera Preview")
        print(f"截图: {frame.shape[1]}×{frame.shape[0]}")

    # ── 多边形勾画 ──
    global _polygon_pts, _frame
    _polygon_pts = []
    _frame = _resize_to_screen(frame)

    cv2.namedWindow(_window_name)
    cv2.setMouseCallback(_window_name, _mouse_callback)
    _redraw()

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13 or key == ord('\r'):  # Enter → 保存
            if len(_polygon_pts) < 3:
                print("⚠ 需要至少 3 个点才能构成多边形")
                continue
            mask = _generate_mask(_frame.shape, _polygon_pts)
            cv2.imwrite(args.output, mask)
            print(f"✓ 蒙版已保存: {args.output} ({mask.shape[1]}×{mask.shape[0]})")
            break
        elif key == ord('r'):
            _polygon_pts = []
            _redraw()
            print("  已清空所有顶点")
        elif key == ord('q') or key == 27:  # ESC
            print("  已取消")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
