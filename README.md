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

# 用蒙版文件 + ArUco 插值（推荐，减少轨迹断点）
python record_camera.py --live --mask mask.png --calib-frames 2 --interpolate

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

## 测试
### 1
```bash
(base) D:\python\Python\割草机导航\相机处理>python record_camera.py --live --mask mask.png --calib-frames 2
==================================================
  模式二: 边录制边显示覆盖率
==================================================
[1/3] 加载蒙版文件: mask.png
[2/3] 打开摄像头 (index=1)...
      分辨率: 640×480
      输出视频: ./camera_record.mp4
[3/3] 实时覆盖模式启动...
等待 ArUco 角点稳定标定 (ID=0,1,2,3, 需连续 2 帧)... 按 'q' 退出
  ✓ 角点 ID=0: avg=(154.0,109.4)  std=(0.00,0.12)px
  ✓ 角点 ID=1: avg=(141.2,362.2)  std=(0.00,0.00)px
  ✓ 角点 ID=2: avg=(502.2,402.6)  std=(0.00,0.12)px
  ✓ 角点 ID=3: avg=(515.5,109.8)  std=(0.00,0.00)px
✓ 单应矩阵标定成功 (基于 2 帧平均)
✓ 覆盖网格初始化: 360×480, 可通行 169468 格 (≈ 4.237 m²)
开始实时覆盖分析...

==================================================
  实时覆盖录制汇总
==================================================
  录制时长:     866.6 s
  总帧数:       22784
  有效轨迹点:   9148
  区域覆盖率:   90.4%
  重复覆盖率:   90.0%
  轨迹长度:     37.44 m
==================================================

完成 ✓
```
问题：
1.界面中文显示有问题，应该是没设置中文字体
2.重复覆盖率太高了，可能定义和我用的不一样：
重复覆盖率 R 用于评估覆盖规划效率，即算法在掉头或细胞跨越时，对已清扫区域造成的过度覆盖比例，高重复覆盖率意味着无效的能源消耗与作业耗时增加。在计算累计覆盖区域时，额外计算每一个像素的覆盖状态变化的次数，在任务终结后，统计内部重叠计数值大于2 的正覆盖像素总和，计算得到路径重复率R：R = \frac{N_{re}}{N_{total}} \times 100\%
在往复式覆盖路径策略下，最优工作带应保持细微的接缝重叠，因此理想的路径值并非绝对为 0，而是根据车体特性及导航偏离，稳定控制在较低的最佳工程冗余水准内。
3.最好是车开始移动时开始计时工作时间，停止时间靠我手动停止即可
4.其实车标记识别不是很连续，不过我看覆盖率还挺高，不知道是不是你自动连接线段了，如果是的话那就没啥问题

## 已修复 (v1.1)

### 1. 中文字体
集成 PIL/Pillow，自动检测系统中文字体。需 `pip install Pillow`。

### 2. 重复覆盖率公式修正
- 旧: `sum(count>=2) / total_passable`
- 新: `sum(count>2) / sum(count>0)` — N_re/N_total，分母为已覆盖区域

### 3. 工作时间计时
从首次检测到机器人 ArUco 开始计时。HUD 显示"工作时间"。

### 4. ArUco 不连续
覆盖率依靠每个点 12cm 半径圆形覆盖填充间隙。新增 `--interpolate` 可在丢失 ≤5 帧时续接。