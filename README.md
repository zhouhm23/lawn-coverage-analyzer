# 割草机导航 · 相机覆盖分析

## 快速开始

### 1. 先画蒙版（三选一）

```bash
# A) 鼠标勾画多边形 → 最直观，推荐先用这个
python draw_mask.py

# B) 提取纯色地板（如绿色地面）
#    先用 color_picker.py 标定 HSV：
python color_picker.py
#    记下输出的 --color-lower/--color-upper 值，然后：
python record_camera.py --live --color-lower 35,50,50 --color-upper 85,255,255

# C) PS 制作 mask.png（老方法，白色=可通行，黑色=障碍）
```

### 2. 实时覆盖（推荐日常工作）

```bash
# 全区域可通行
python record_camera.py --live

# 用蒙版文件
python record_camera.py --live --mask mask.png

# 用颜色提取（绿色地板示例）
python record_camera.py --live --color-lower 35,50,50 --color-upper 85,255,255
```

操作：摆好摄像头 → 角点 ArUco 自动标定 → 启动割草机 → 右下角实时看覆盖率 → 按 q 退出。

> 如果角点检测闪烁，画面会显示 `CALIBRATING [N/30]` 进度条。标定需要连续 30 帧稳定检测到 4 角。
> 可用 `--calib-frames 15` 减少等待，或用 `--calib-frames 60` 更稳定。

### 3. 离线分析（先录后算）

```bash
python record_camera.py --record --output demo.mp4
python run_camera_coverage.py --video demo.mp4 --mask mask.png --visualize
```

### 4. 单张图片测试

```bash
python test_aruco.py --image test.png
python test_aruco.py --camera
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `test_aruco.py` | 单张图片 ArUco 检测 |
| `draw_mask.py` | 鼠标勾画多边形生成蒙版 |
| `color_picker.py` | HSV 颜色范围交互式标定 |
| `record_camera.py` | 录屏 + 实时覆盖 / 离线记录 |
| `run_camera_coverage.py` | 离线视频覆盖分析 |
| `camera_coverage.py` | 核心算法（单应矩阵、覆盖计算） |

## 蒙版优先级

`文件蒙版 > 颜色蒙版 > 全区域可通行`（不传任何蒙版参数时默认全区域）
