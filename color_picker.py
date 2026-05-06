#!/usr/bin/env python3
"""
color_picker.py — HSV 颜色范围交互式标定工具。

用法:
    python color_picker.py

操作:
    H_low/H_high/S_low/S_high/V_low/V_high — 6 个滑块调 HSV 范围
    画面中实时显示提取结果（白色=选中区域）
    调好后记下终端的 HSV 值，传给 record_camera.py --color-lower/--color-upper
"""

import cv2
import numpy as np

WINDOW = "HSV Color Picker — 调好后按 q 退出，终端会打印 HSV 值"


def _on_trackbar(_val):
    pass  # 回调仅用于 slider 生效


def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("无法打开摄像头 (index=1)")
        return

    cv2.namedWindow(WINDOW)
    cv2.createTrackbar("H_low", WINDOW, 0, 179, _on_trackbar)
    cv2.createTrackbar("H_high", WINDOW, 179, 179, _on_trackbar)
    cv2.createTrackbar("S_low", WINDOW, 0, 255, _on_trackbar)
    cv2.createTrackbar("S_high", WINDOW, 255, 255, _on_trackbar)
    cv2.createTrackbar("V_low", WINDOW, 0, 255, _on_trackbar)
    cv2.createTrackbar("V_high", WINDOW, 255, 255, _on_trackbar)

    print("拖动滑块调整 HSV 范围，按 q 退出并打印最终值...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h_low = cv2.getTrackbarPos("H_low", WINDOW)
        h_high = cv2.getTrackbarPos("H_high", WINDOW)
        s_low = cv2.getTrackbarPos("S_low", WINDOW)
        s_high = cv2.getTrackbarPos("S_high", WINDOW)
        v_low = cv2.getTrackbarPos("V_low", WINDOW)
        v_high = cv2.getTrackbarPos("V_high", WINDOW)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
        upper = np.array([h_high, s_high, v_high], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # 显示：原图 + mask 叠加
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        display = np.hstack([frame, mask_color])

        # 放缩
        h, w = display.shape[:2]
        scale = min(1920 / w, 1080 / h) * 0.85
        if scale < 1.0:
            display = cv2.resize(display, (int(w * scale), int(h * scale)))

        cv2.imshow(WINDOW, display)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n--color-lower {h_low},{s_low},{v_low} --color-upper {h_high},{s_high},{v_high}")
    print("把上面这行加到 record_camera.py --live 后面即可")


if __name__ == "__main__":
    main()
