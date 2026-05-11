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

# 用蒙版文件 + ArUco 插值（减少轨迹断点，识别差时再用）
python record_camera.py --live --mask mask.png --calib-frames 2

# 用颜色提取（绿色地板示例）
python record_camera.py --live --color-lower 35,50,50 --color-upper 85,255,255
```

操作：摆好摄像头 → 角点 ArUco 自动标定 → 启动割草机 → 右下角实时看覆盖率。
- **按 `q`**：直接退出（不重命名视频）
- **按 `f`**：完成任务 → 视频加时间后缀（如 `camera_record05101600.mp4`）+ 保存 `temp_result_xxx.csv` 临时结果（需手动确认后写入 覆盖率实验.csv）

录制视频为**原始画面**（无标注叠加），可随时离线重新分析。

> 如果角点检测闪烁，画面会显示 `CALIBRATING [N/3]` 进度条。标定需要连续 3 帧稳定检测到 4 角。
> 可用 `--calib-frames 1` 更快，或用 `--calib-frames 10` 更稳定。

### 3. 离线分析（先录后算）

```bash
python record_camera.py --record --output demo.mp4

# 离线分析（默认每秒处理1帧，车速0.26m/s下安全；如需逐帧可用 --frame-skip 1）
python run_camera_coverage.py --video demo.mp4 --mask mask.png --visualize

# 导出论文插图（实物照 + 红色条带覆盖 + 绿色轨迹中心线）
python run_camera_coverage.py --video camera_record05102100.mp4 --mask mask.png --export-overlay
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

## 测试1 (v1.0 原始输出)
```bash
(base) D:\python\Python\割草机导航\相机处理>python record_camera.py --live --mask mask.png --calib-frames 2
...
==================================================
  实时覆盖录制汇总
==================================================
  录制时长:     866.6 s（含标定等待 3min35s + 静止 7s）
  总帧数:       22784
  有效轨迹点:   9148
  区域覆盖率:   90.4%
  轨迹长度:     37.44 m
==================================================
```
> v1.3 条带面积法修正后：
> 工作时长 = 866.6 − 215 − 7 = **644.6s**，
> N_total ≈ 169468×0.904 = 153199，唯一覆盖 = 3.830m²，条带 = 37.44×0.24 = 8.986m²，
> 重复覆盖率 = (8.986−3.830)/8.986 ≈ **57.4%**
问题：
1.界面中文显示有问题，应该是没设置中文字体
2.重复覆盖率太高了，可能定义和我用的不一样：
重复覆盖率 R 用于评估覆盖规划效率，即算法在掉头或细胞跨越时，对已清扫区域造成的过度覆盖比例，高重复覆盖率意味着无效的能源消耗与作业耗时增加。在计算累计覆盖区域时，额外计算每一个像素的覆盖状态变化的次数，在任务终结后，统计内部重叠计数值大于2 的正覆盖像素总和，计算得到路径重复率R：R = \frac{N_{re}}{N_{total}} \times 100\%
在往复式覆盖路径策略下，最优工作带应保持细微的接缝重叠，因此理想的路径值并非绝对为 0，而是根据车体特性及导航偏离，稳定控制在较低的最佳工程冗余水准内。
3.最好是车开始移动时开始计时工作时间，停止时间靠我手动停止即可
4.其实车标记识别不是很连续，不过我看覆盖率还挺高，不知道是不是你自动连接线段了，如果是的话那就没啥问题

### 1. 中文字体
集成 PIL/Pillow，自动检测系统中文字体。需 `pip install Pillow`。

### 2. 重复覆盖率公式 (v2.1 条带面积法 + 空间重采样滤波器)

**网格自适应阈值法（v2.0，已废弃）**: 依赖 `hits_per_pass` 动态阈值，但对非均匀采样场景不稳定。

**条带面积法（v2.1 回退）**:
$$\text{repeat} = \frac{S_{\text{strip}} - S_{\text{unique}}}{S_{\text{strip}}} \times 100\%, \quad S_{\text{strip}} = L \cdot 2R, \quad S_{\text{unique}} = N_{\text{covered}} \cdot \Delta x^2$$

- $S_{\text{strip}}$: 理论条带总面积 = 轨迹长度 × 割草宽度
- $S_{\text{unique}}$: 网格唯一覆盖面积（count > 0 的格面积总和）
- $L$: 空间重采样后的轨迹长度
- $2R$: 割草条带宽度（= 2 × coverage_radius）

**空间重采样滤波器（消除采样频率误差）**:
将原始轨迹按固定空间间隔（默认 0.02m）均匀重采样，使轨迹长度独立于帧率。
1Hz 和 30Hz 采样得到相近的轨迹长度 → 条带面积法结果可跨实验对比。

```bash
# 调节空间重采样间隔（0=不重采样，使用原始轨迹）
python run_camera_coverage.py --video demo.mp4 --mask mask.png --spatial-interval 0.02
```

### 3. 工作时间计时
从首次检测到机器人 ArUco 开始计时。HUD 显示"工作时间"。

### 4. ArUco 不连续
覆盖率依靠每个点 12cm 半径圆形覆盖填充间隙。新增 `--interpolate` 可在丢失 ≤5 帧时续接。

### 5. 静止抖动过滤 (v1.2)
ArUco 检测存在亚像素抖动，静止时轨迹长度会虚假增长。新增 `--min-movement` 阈值（默认 0.01m = 1cm），低于此值的移动不计入轨迹长度。参考车速 0.26m/s，每帧位移约 8.7mm。
```bash
# 调整阈值（如 ArUco 抖动较大可适当提高）
python record_camera.py --live --mask mask.png --min-movement 0.015
```

## 测试2 (v1.2 原始输出 — 条带面积法修复前)
```
总录制时长:   632.6 s（含静止 24s）
有效工作时长: 631.5 s（代码计时）
总帧数:       18676
有效轨迹点:   10976
区域覆盖率:   93.0%
轨迹长度:     17.72 m
```
> v1.3 条带面积法修正后：
> 工作时长 = 632.6 − 24 = **608.6s**，
> N_total = 158004，唯一覆盖 = 3.950m²，条带 = 17.72×0.24 = 4.253m²，
> 重复覆盖率 = (4.253−3.950)/4.253 ≈ **7.1%** ✓
>
### 测试3
```text
视觉+改进ccpp：
总录制时长:   575.3 s - 25s -26s=524.3
总帧数:       15954
有效轨迹点:   8999
区域覆盖率:   86.7%
重复覆盖率:   15.1%  (条带=4.366m², 唯一覆盖=3.705m²)
轨迹长度:     18.19 m

雷达+原始ccpp：
总录制时长:   928.0 s - 67s - 3s=858
总帧数:       22837
有效轨迹点:   17977
区域覆盖率:   66.1%
重复覆盖率:   0.0%  (条带=2.653m², 唯一覆盖=2.810m²)
轨迹长度:     15.61 m

python run_camera_coverage.py --video camera_record05102100.mp4 --mask mask2100.png --export-overlay 
区域覆盖率:      50.31 %
重复覆盖率:      23.88 %
覆盖效率:       0.0158 m⁻¹
轨迹总长度:      31.90 m
可通行总面积:    4.252 m²
有效/总帧:      604/762
⚠ ArUco 丢失率 20.7% > 5%，请检查视频质量！
```

## 测试4 (v2.1 条带面积法 + 空间重采样)
```bash
python run_camera_coverage.py --video camera_record05102100.mp4 --mask mask2100.png --spatial-interval 0.02
```
```
区域覆盖率:      66.05 %
重复覆盖率:      48.15 %  (条带=5.4158m², 唯一覆盖=2.8083m²)
覆盖效率:       0.0207 m⁻¹
轨迹总长度:      31.86 m（空间重采样后，原始=31.90m）
可通行总面积:    4.252 m²
有效/总帧:      604/762
⚠ ArUco 丢失率 20.7% > 5%，请检查视频质量！

camera_record05111712.mp4测试结果：
1.在线分析，重复率明显错误
总录制时长:   606.9 s
有效工作时长: 600.9 s
总帧数:       15515
有效轨迹点:   13306
区域覆盖率:   76.6%
重复覆盖率:   0.0%  (条带=1.8595m², 唯一覆盖=3.2545m²)
轨迹长度:     10.94 m

2.纠正最小阈值后：
python run_camera_coverage.py --video camera_record05111712.mp4 --mask mask2100.png --spatial-interval 0.02 --frame-skip 30 2>&1 --export-overlay
  区域覆盖率:      76.02 %
  重复覆盖率:      36.70 %
    (条带=4.9314m², 唯一覆盖=3.1217m²)
  覆盖效率:       0.0262 m⁻¹
  轨迹总长度:      29.01 m
  可通行总面积:    4.106 m²
  有效/总帧:      451/518
  ⚠ ArUco 丢失率 12.9% > 5%，请检查视频质量！

  结果合理了
```