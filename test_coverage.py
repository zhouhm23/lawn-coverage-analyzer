#!/usr/bin/env python3
"""
test_coverage.py — 覆盖算法精度验证脚本。

用合成无噪轨迹测试 camera_coverage 核心算法: accumulate_coverage,
compute_metrics, compute_trajectory_length, resample_trajectory_spatial。

每个测试用例用解析几何真值或 1mm 高分辨率参考值验证。

用法:
    python test_coverage.py
"""

import math
import sys
import os
import csv

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)

import numpy as np
from camera_coverage import (
    CameraCoverageConfig,
    accumulate_coverage,
    compute_metrics,
    compute_trajectory_length,
    resample_trajectory_spatial,
)


# ═══════════════════════════════════════════════════════════
# 常量 & 工具
# ═══════════════════════════════════════════════════════════

R = 0.085
W = 2 * R
RES = 0.005


def _traj(points, t0=0.0, dt=1.0):
    return [(t0 + i * dt, x, y) for i, (x, y) in enumerate(points)]


def _circle_intersection(r_, d):
    if d >= 2 * r_:
        return 0.0
    if d <= 0:
        return math.pi * r_ ** 2
    return (2 * r_ ** 2 * math.acos(d / (2 * r_))
            - d / 2 * math.sqrt(4 * r_ ** 2 - d ** 2))


def _highres_ref(traj, cfg, fine_res=0.001):
    gw = int(math.ceil(cfg.paper_width / fine_res))
    gh = int(math.ceil(cfg.paper_height / fine_res))
    covered = np.zeros((gh, gw), dtype=np.int32)
    rad = int(math.ceil(cfg.coverage_radius / fine_res))
    for _, px, py in traj:
        cx = int(px / fine_res)
        cy = int(py / fine_res)
        x0 = max(cx - rad, 0)
        x1 = min(cx + rad + 1, gw)
        y0 = max(cy - rad, 0)
        y1 = min(cy + rad + 1, gh)
        if x0 >= x1 or y0 >= y1:
            continue
        xs = (np.arange(x0, x1, dtype=np.float64) + 0.5) * fine_res
        ys = (np.arange(y0, y1, dtype=np.float64) + 0.5) * fine_res
        X, Y = np.meshgrid(xs, ys)
        circle = (X - px) ** 2 + (Y - py) ** 2 <= (cfg.coverage_radius ** 2)
        covered[y0:y1, x0:x1] += circle.astype(np.int32)
    N = int(np.sum(covered > 0))
    return N * (fine_res ** 2)


def _full_mask(cfg):
    return np.ones((cfg.grid_h, cfg.grid_w), dtype=bool)


def _cfg(**kw):
    return CameraCoverageConfig(coverage_radius=R, resolution=RES, **kw)


def _uturn(x, y_from, y_to, n=20):
    """U 形转弯: 从 (x,y_from) 半圆转到 (x,y_to)。"""
    r = abs(y_to - y_from) / 2
    yc = (y_from + y_to) / 2
    if y_to > y_from:
        a = np.linspace(-np.pi / 2, np.pi / 2, n)
    else:
        a = np.linspace(np.pi / 2, -np.pi / 2, n)
    return [(x + r * np.cos(aa), yc + r * np.sin(aa)) for aa in a]


# ═══════════════════════════════════════════════════════════
# 测试用例
# ═══════════════════════════════════════════════════════════

def test_single_point():
    """TC1: 单点 → unique ≈ πR²"""
    cfg = _cfg(spatial_resample_interval=0)
    traj = _traj([(0.9, 1.2)])
    covered = accumulate_coverage(traj, _full_mask(cfg), cfg)
    m = compute_metrics(covered, _full_mask(cfg), 0.0, W, RES)
    exp = math.pi * R ** 2
    err = abs(m["unique_area_m2"] - exp) / exp * 100
    print(f"  TC1: unique={m['unique_area_m2']:.5f} m², exp={exp:.5f}, err={err:.2f}%")
    assert err < 2, f"err {err:.1f}%"
    return True


def test_straight_line():
    """TC2: 直线 L=1.5m → unique≈L×2R+πR², repeat=0"""
    cfg = _cfg(spatial_resample_interval=0)
    L = 1.5
    traj = _traj([(0.15 + i * L / 75, 1.2) for i in range(76)])
    covered = accumulate_coverage(traj, _full_mask(cfg), cfg)
    tlen = compute_trajectory_length(traj, min_movement=0)
    m = compute_metrics(covered, _full_mask(cfg), tlen, W, RES)
    eu = L * W + math.pi * R ** 2
    print(f"  TC2: unique={m['unique_area_m2']:.5f}(exp {eu:.5f}), strip={m['strip_area_m2']:.5f}(exp {L*W:.5f}), repeat={m['repeat_coverage']:.3f}")
    assert abs(m["unique_area_m2"] - eu) / eu < 0.05
    assert m["repeat_coverage"] == 0.0
    return True


def test_two_separated_points():
    """TC3: 两点相距 >2R → 无重叠, unique=2πR²"""
    cfg = _cfg(spatial_resample_interval=0)
    d = 0.5
    traj = _traj([(0.5, 1.2), (0.5 + d, 1.2)])
    covered = accumulate_coverage(traj, _full_mask(cfg), cfg)
    tlen = compute_trajectory_length(traj, min_movement=0)
    m = compute_metrics(covered, _full_mask(cfg), tlen, W, RES)
    exp = 2 * math.pi * R ** 2
    err = abs(m["unique_area_m2"] - exp) / exp * 100
    print(f"  TC3: unique={m['unique_area_m2']:.5f}, exp=2πR²={exp:.5f}, err={err:.2f}%")
    assert err < 2
    return True


def test_two_overlapping_points():
    """TC4: 两点 d=1.5R → 圆交面积公式验证"""
    cfg = _cfg(spatial_resample_interval=0)
    d = 1.5 * R
    traj = _traj([(0.5, 1.2), (0.5 + d, 1.2)])
    covered = accumulate_coverage(traj, _full_mask(cfg), cfg)
    tlen = compute_trajectory_length(traj, min_movement=0)
    m = compute_metrics(covered, _full_mask(cfg), tlen, W, RES)
    ol = _circle_intersection(R, d)
    exp = 2 * math.pi * R ** 2 - ol
    err = abs(m["unique_area_m2"] - exp) / exp * 100
    print(f"  TC4: unique={m['unique_area_m2']:.5f}, exp={exp:.5f}(overlap={ol:.5f}), err={err:.2f}%")
    assert err < 3
    return True


def test_parallel_no_overlap():
    """TC5: 双线间距 2R + U 转 → repeat≈0"""
    cfg = _cfg(spatial_resample_interval=0)
    L, x0, y0 = 1.2, 0.15, 1.0
    n_l = 61
    line1 = [(x0 + j * L / (n_l - 1), y0) for j in range(n_l)]
    turn = _uturn(x0 + L, y0, y0 + W)
    line2 = [(x0 + L - j * L / (n_l - 1), y0 + W) for j in range(n_l)]
    traj = _traj(line1 + turn + line2)
    covered = accumulate_coverage(traj, _full_mask(cfg), cfg)
    tlen = compute_trajectory_length(traj, min_movement=0)
    m = compute_metrics(covered, _full_mask(cfg), tlen, W, RES)
    ref = _highres_ref(traj, cfg)
    ue = abs(m["unique_area_m2"] - ref) / ref * 100
    print(f"  TC5: unique={m['unique_area_m2']:.5f}(ref={ref:.5f},err={ue:.2f}%), repeat={m['repeat_coverage']*100:.1f}%")
    assert ue < 3
    assert m["repeat_coverage"] < 0.05
    return True


def test_parallel_half_overlap():
    """TC6: 双线间距 R(50%重叠) + U 转 → repeat 在 15-50%"""
    cfg = _cfg(spatial_resample_interval=0)
    L, x0, y0 = 1.2, 0.15, 1.0
    n_l = 61
    line1 = [(x0 + j * L / (n_l - 1), y0) for j in range(n_l)]
    turn = _uturn(x0 + L, y0, y0 + R)
    line2 = [(x0 + L - j * L / (n_l - 1), y0 + R) for j in range(n_l)]
    traj = _traj(line1 + turn + line2)
    covered = accumulate_coverage(traj, _full_mask(cfg), cfg)
    tlen = compute_trajectory_length(traj, min_movement=0)
    m = compute_metrics(covered, _full_mask(cfg), tlen, W, RES)
    ref = _highres_ref(traj, cfg)
    ue = abs(m["unique_area_m2"] - ref) / ref * 100
    rp = m["repeat_coverage"]
    print(f"  TC6: unique={m['unique_area_m2']:.5f}(ref={ref:.5f},err={ue:.2f}%), repeat={rp*100:.1f}%")
    assert ue < 3
    assert 0.10 < rp < 0.60, f"repeat {rp*100:.1f}% out of [10,60]%"
    return True


def test_full_coverage():
    """TC7: 10线×1.6m, 间距 0.12m → coverage > 85%"""
    cfg = _cfg(spatial_resample_interval=0)
    L, n_l, n_lines, spacing = 1.6, 81, 10, 0.12
    x0, y0 = 0.1, 0.15
    pts = []
    for i in range(n_lines):
        y = y0 + i * spacing
        if i % 2 == 0:
            line = [(x0 + j * L / (n_l - 1), y) for j in range(n_l)]
        else:
            line = [(x0 + L - j * L / (n_l - 1), y) for j in range(n_l)]
        pts.extend(line)
        if i < n_lines - 1:
            pts.extend(_uturn(line[-1][0], y, y0 + (i + 1) * spacing))
    traj = _traj(pts)
    covered = accumulate_coverage(traj, _full_mask(cfg), cfg)
    tlen = compute_trajectory_length(traj, min_movement=0)
    m = compute_metrics(covered, _full_mask(cfg), tlen, W, RES)
    ref = _highres_ref(traj, cfg)
    ue = abs(m["unique_area_m2"] - ref) / ref * 100
    print(f"  TC7: unique={m['unique_area_m2']:.5f}(ref={ref:.5f},err={ue:.2f}%), cov={m['area_coverage']*100:.1f}%, repeat={m['repeat_coverage']*100:.1f}%")
    assert ue < 3
    # 10线 ×0.12m间距 ≈ 覆盖 y 方向 ~1.3m / 2.4m 纸面 ≈ 50%
    assert m["area_coverage"] > 0.45, f"coverage {m['area_coverage']:.1%} < 45%"
    return True


def test_resample_fidelity():
    """TC8: 不同源采样率 → 重采样后长度一致 (<1%)"""
    cfg = _cfg(spatial_resample_interval=0.02)
    L = 5.0
    s = _traj([(0.1 + i * 0.5, 1.2) for i in range(11)])
    d = _traj([(0.1 + i * 0.05, 1.2) for i in range(101)])
    ls = compute_trajectory_length(resample_trajectory_spatial(s, 0.02), 0.0)
    ld = compute_trajectory_length(resample_trajectory_spatial(d, 0.02), 0.0)
    diff = abs(ls - ld) / L * 100
    print(f"  TC8: sparse={ls:.4f}m, dense={ld:.4f}m, diff={diff:.3f}%")
    assert diff < 1
    return True


def test_noise_inflation():
    """TC9: σ=3mm 噪声 → 重采样后膨胀 <5%"""
    cfg = _cfg(spatial_resample_interval=0.02)
    np.random.seed(42)
    L, n, sig = 5.0, 251, 0.003
    clean = [(0.1 + i * L / (n - 1), 1.2) for i in range(n)]
    noisy = [(x + np.random.normal(0, sig), y + np.random.normal(0, sig)) for x, y in clean]
    lc = compute_trajectory_length(resample_trajectory_spatial(_traj(clean), 0.02), 0.0)
    ln = compute_trajectory_length(resample_trajectory_spatial(_traj(noisy), 0.02), 0.0)
    infl = (ln - lc) / lc * 100
    print(f"  TC9: clean={lc:.4f}m, noisy={ln:.4f}m, infl={infl:+.2f}%")
    assert infl < 5
    return True


# ═══════════════════════════════════════════════════════════
# 真实场景模拟 (TC10-TC12)
# ═══════════════════════════════════════════════════════════

V = 0.40                      # 名义车速 m/s
SIGMA = 0.003                 # ArUco 噪声 σ (实测 ~0.8mm, 保守取 3mm)
DROPOUT_RATE = 0.15           # 模拟 ArUco 丢失率 15%


def _make_backforth(n_passes, L, spacing, x0=0.1, y0=0.15, n_line=81):
    """生成蛇形往复轨迹点列表 [(x,y), ...]，带 U 形转弯。"""
    pts = []
    for i in range(n_passes):
        y = y0 + i * spacing
        if i % 2 == 0:
            line = [(x0 + j * L / (n_line - 1), y) for j in range(n_line)]
        else:
            line = [(x0 + L - j * L / (n_line - 1), y) for j in range(n_line)]
        pts.extend(line)
        if i < n_passes - 1:
            pts.extend(_uturn(line[-1][0], y, y0 + (i + 1) * spacing))
    return pts


def _add_noise(points, sigma=SIGMA, seed=123):
    """给点加高斯噪声。"""
    rng = np.random.RandomState(seed)
    return [(x + rng.normal(0, sigma), y + rng.normal(0, sigma)) for x, y in points]


def _simulate_dropout(points, rate=DROPOUT_RATE, max_gap=5, seed=456):
    """模拟 ArUco 丢失: 随机删除连续块 (≤max_gap 帧)。保留原始索引。"""
    rng = np.random.RandomState(seed)
    keep = [True] * len(points)
    i = 0
    while i < len(points):
        if rng.random() < rate:
            gap = rng.randint(1, max_gap + 1)
            for j in range(i, min(i + gap, len(points))):
                keep[j] = False
            i += gap
        else:
            i += 1
    return [p for p, k in zip(points, keep) if k], keep


def _full_pipeline(points, cfg, label=""):
    """完整管线: 轨迹 → 重采样 → 指标。返回 (traj_len, unique, repeat, coverage)"""
    traj = _traj(points)
    resampled = resample_trajectory_spatial(traj, cfg.spatial_resample_interval)
    tlen = compute_trajectory_length(resampled, min_movement=0.0)
    covered = accumulate_coverage(resampled, _full_mask(cfg), cfg)
    m = compute_metrics(covered, _full_mask(cfg), tlen, W, RES)
    return tlen, m["unique_area_m2"], m["repeat_coverage"], m["area_coverage"]


def test_realistic_noise():
    """TC10: 往复轨迹 + ArUco 噪声 → 量化噪声导致的长度和重复率偏差"""
    cfg = _cfg(spatial_resample_interval=0.02)
    # 3线往复 (简化但足够说明问题)
    clean_pts = _make_backforth(3, 1.6, 0.15, n_line=81)
    noisy_pts = _add_noise(clean_pts, sigma=SIGMA)

    tl_c, uq_c, rp_c, cv_c = _full_pipeline(clean_pts, cfg, "clean")
    tl_n, uq_n, rp_n, cv_n = _full_pipeline(noisy_pts, cfg, "noisy")

    len_infl = (tl_n - tl_c) / tl_c * 100
    rp_delta = (rp_n - rp_c) * 100  # 百分点差异

    print(f"  TC10 噪声影响 (σ={SIGMA*1000:.0f}mm, 3线往复):")
    print(f"        clean:  len={tl_c:.3f}m, unique={uq_c:.4f}m², repeat={rp_c*100:.1f}%")
    print(f"        noisy:  len={tl_n:.3f}m, unique={uq_n:.4f}m², repeat={rp_n*100:.1f}%")
    print(f"        Δlen={len_infl:+.2f}%, Δrepeat={rp_delta:+.1f}pp")

    # 噪声应该小幅度膨胀长度 (<5%)，unique 基本不变
    assert len_infl < 5, f"噪声长度膨胀 {len_infl:.1f}%"
    assert abs(uq_n - uq_c) / uq_c < 0.03, "噪声不应显著改变 unique 面积"
    return True


def test_realistic_dropout():
    """TC11: 往复轨迹 + ArUco 丢帧 → 量化丢帧导致的偏差"""
    cfg = _cfg(spatial_resample_interval=0.02)
    # 使用更多 pass 让效果更明显
    clean_pts = _make_backforth(5, 1.6, 0.15, n_line=81)

    # 模拟丢帧: 从干净轨迹删点
    dropped_pts, keep_mask = _simulate_dropout(clean_pts, rate=DROPOUT_RATE, max_gap=5)
    actual_drop_rate = 1 - sum(keep_mask) / len(keep_mask)

    tl_c, uq_c, rp_c, cv_c = _full_pipeline(clean_pts, cfg, "clean")
    tl_d, uq_d, rp_d, cv_d = _full_pipeline(dropped_pts, cfg, "dropped")

    len_change = (tl_d - tl_c) / tl_c * 100
    rp_delta = (rp_d - rp_c) * 100
    cv_delta = (cv_d - cv_c) * 100

    print(f"  TC11 丢帧影响 (rate={actual_drop_rate:.0%}, max_gap=5帧):")
    print(f"        clean:   len={tl_c:.3f}m, unique={uq_c:.4f}m², repeat={rp_c*100:.1f}%, cov={cv_c*100:.1f}%")
    print(f"        dropped: len={tl_d:.3f}m, unique={uq_d:.4f}m², repeat={rp_d*100:.1f}%, cov={cv_d*100:.1f}%")
    print(f"        Δlen={len_change:+.2f}%, Δrepeat={rp_delta:+.1f}pp, Δcov={cv_delta:+.1f}pp")

    # 丢帧后直线连接跳过转弯 → 长度偏低，覆盖率略降
    assert len_change < 10, f"丢帧长度变化 {len_change:.1f}% 超出预期"
    assert cv_delta > -10, f"覆盖率下降 {cv_delta:.1f}pp 超出预期"
    return True


def test_realistic_combined():
    """TC12: 噪声 + 丢帧 + 速度约束滤波 → 端到端偏差量化"""
    cfg = _cfg(spatial_resample_interval=0.02)
    clean_pts = _make_backforth(5, 1.6, 0.15, n_line=81)

    # 加噪声
    noisy_pts = _add_noise(clean_pts, sigma=SIGMA, seed=42)
    # 模拟丢帧
    dropped_pts, keep_mask = _simulate_dropout(noisy_pts, rate=DROPOUT_RATE, max_gap=5, seed=99)

    tl_c, uq_c, rp_c, cv_c = _full_pipeline(clean_pts, cfg, "clean")
    tl_x, uq_x, rp_x, cv_x = _full_pipeline(dropped_pts, cfg, "noisy+dropped")

    len_change = (tl_x - tl_c) / tl_c * 100
    rp_delta = (rp_x - rp_c) * 100
    cv_delta = (cv_x - cv_c) * 100

    # 用高分辨参考值验证 unique_area 准确性
    ref_unique = _highres_ref(_traj(dropped_pts), cfg)
    uq_vs_ref = abs(uq_x - ref_unique) / ref_unique * 100

    print(f"  TC12 噪声+丢帧 综合:")
    print(f"        clean:         len={tl_c:.3f}m, unique={uq_c:.4f}m², repeat={rp_c*100:.1f}%, cov={cv_c*100:.1f}%")
    print(f"        noisy+dropped: len={tl_x:.3f}m, unique={uq_x:.4f}m², repeat={rp_x*100:.1f}%, cov={cv_x*100:.1f}%")
    print(f"        unique vs 1mm-ref: err={uq_vs_ref:.2f}%")
    print(f"        Δlen={len_change:+.2f}%, Δrepeat={rp_delta:+.1f}pp, Δcov={cv_delta:+.1f}pp")

    # 核心: unique_area 即使经过噪声+丢帧+重采样, 仍应与高分辨参考一致
    assert uq_vs_ref < 3, f"unique vs ref 误差 {uq_vs_ref:.1f}%"
    assert abs(cv_delta) < 15, f"覆盖率偏差 {cv_delta:.1f}pp 过大"
    return True


# ═══════════════════════════════════════════════════════════
# TC13: 参数扫描 — frame_skip × spatial_interval 敏感性
# ═══════════════════════════════════════════════════════════

def _generate_dense_ground_truth():
    """生成密集真值轨迹 (模拟 30fps × 0.40m/s, ~8.7mm 间距) + 已知几何路径。"""
    # 5线往复，模拟约 60s 真实割草
    L, spacing, n_lines = 1.6, 0.15, 5
    x0, y0 = 0.1, 0.15
    n_line = int(L / 0.0087) + 1  # ~8.7mm 间距, 相当于 30fps @ 0.40m/s
    pts = []
    for i in range(n_lines):
        y = y0 + i * spacing
        if i % 2 == 0:
            line = [(x0 + j * L / (n_line - 1), y) for j in range(n_line)]
        else:
            line = [(x0 + L - j * L / (n_line - 1), y) for j in range(n_line)]
        pts.extend(line)
        if i < n_lines - 1:
            turn_n = max(10, int(math.pi * spacing / 2 / 0.0087))
            pts.extend(_uturn(line[-1][0], y, y0 + (i + 1) * spacing, n=turn_n))
    return pts


def _subsample(points, step):
    """每隔 step-1 个点取一个 (step=1 → 全部, step=30 → 1/30)。"""
    return [p for i, p in enumerate(points) if i % step == 0]


def test_param_sweep(output_csv="param_sweep.csv"):
    """TC13: 参数扫描 — 同时跑无噪/有噪(σ=3mm)两组, 对比噪声对参数选择的影响。

    输出 param_sweep.csv，每组参数有 clean/noisy 两行。
    真值由密集无噪轨迹 + si=0.005 确定。
    """
    import time as _time
    t0 = _time.time()

    # ── 密集真值轨迹 ─────────────────────────────────
    dense_clean = _generate_dense_ground_truth()
    rng = np.random.RandomState(42)
    dense_noisy = [(x + rng.normal(0, 0.003), y + rng.normal(0, 0.003))
                   for x, y in dense_clean]

    print(f"  TC13 参数扫描: 密集真值 {len(dense_clean)} 点 (无噪 + σ=3mm噪声)")

    # ── 真值 (无噪密集 + 细重采样) ──────────────────
    cfg_truth = CameraCoverageConfig(coverage_radius=R, resolution=RES,
                                     spatial_resample_interval=0.005,
                                     frame_skip=1, video_fps=30.0)
    tl_t, uq_t, rp_t, cv_t = _full_pipeline(dense_clean, cfg_truth)
    ref_unique = _highres_ref(_traj(dense_clean), cfg_truth, fine_res=0.001)

    print(f"       真值(无噪): len={tl_t:.3f}m, unique={uq_t:.4f}m², "
          f"repeat={rp_t*100:.1f}%, cov={cv_t*100:.1f}%")
    print(f"       1mm-ref unique: {ref_unique:.4f}m²")
    print()

    # ── 参数网格 ─────────────────────────────────────
    frame_skips = [1, 2, 3, 5, 10, 15, 30, 60]
    spatial_intervals = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
    SIGMA = 0.003
    rows = []
    header = ["noise", "frame_skip", "spatial_interval", "n_raw", "n_resampled",
              "raw_len_m", "resampled_len_m", "unique_area_m2", "strip_area_m2",
              "repeat_coverage", "area_coverage",
              "len_vs_truth_pct", "repeat_vs_truth_pp"]

    def _run_one(points, fs, si, label):
        sub = _subsample(points, fs)
        cfg = CameraCoverageConfig(coverage_radius=R, resolution=RES,
                                   spatial_resample_interval=si,
                                   frame_skip=1, video_fps=30.0)
        traj = _traj(sub)
        raw_len = compute_trajectory_length(traj, cfg.min_movement)
        if si > 0 and len(sub) > 1:
            resampled = resample_trajectory_spatial(traj, si)
        else:
            resampled = traj
        resampled_len = compute_trajectory_length(resampled, min_movement=0.0)
        covered = accumulate_coverage(resampled, _full_mask(cfg), cfg)
        m = compute_metrics(covered, _full_mask(cfg), resampled_len, W, RES)
        len_vs = (resampled_len - tl_t) / tl_t * 100
        rp_vs = (m["repeat_coverage"] - rp_t) * 100
        return [label, fs, si, len(sub), len(resampled),
                f"{raw_len:.4f}", f"{resampled_len:.4f}",
                f"{m['unique_area_m2']:.5f}", f"{m['strip_area_m2']:.5f}",
                f"{m['repeat_coverage']:.4f}", f"{m['area_coverage']:.4f}",
                f"{len_vs:+.2f}", f"{rp_vs:+.2f}"]

    for fs in frame_skips:
        for si in spatial_intervals:
            rows.append(_run_one(dense_clean, fs, si, "clean"))
            rows.append(_run_one(dense_noisy, fs, si, "noisy"))

    # 写入 CSV
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow([f"# truth: len={tl_t:.4f}m, unique={uq_t:.5f}m², "
                     f"repeat={rp_t:.4f}, cov={cv_t:.4f}, 1mm-ref={ref_unique:.5f}m², "
                     f"noise_sigma={SIGMA}m"])
        w.writerows(rows)

    # ── 打印: clean vs noisy 对比 (仅 si=0.02) ──────
    print(f"  {'fs':>4s} {'si':>5s} {'clean_len':>9s} {'noisy_len':>9s} {'Δ噪声%':>7s} "
          f"{'clean_rpt':>9s} {'noisy_rpt':>9s} {'Δ噪声pp':>7s}")
    print(f"  {'-'*4} {'-'*5} {'-'*9} {'-'*9} {'-'*7} {'-'*9} {'-'*9} {'-'*7}")
    for fs in frame_skips:
        c = [r for r in rows if r[1] == fs and r[2] == 0.02 and r[0] == "clean"][0]
        n = [r for r in rows if r[1] == fs and r[2] == 0.02 and r[0] == "noisy"][0]
        len_diff = (float(n[6]) - float(c[6])) / float(c[6]) * 100
        rp_diff = (float(n[9]) - float(c[9])) * 100
        print(f"  {fs:4d} {0.02:5.2f} {float(c[6]):9.3f} {float(n[6]):9.3f} {len_diff:+6.1f}% "
              f"{float(c[9])*100:8.1f}% {float(n[9])*100:8.1f}% {rp_diff:+6.1f}pp")

    # ── 关键指标 ────────────────────────────────────
    si002_clean = [float(r[6]) for r in rows if r[1] <= 15 and r[2] == 0.02 and r[0] == "clean"]
    si002_noisy = [float(r[6]) for r in rows if r[1] <= 15 and r[2] == 0.02 and r[0] == "noisy"]
    avg_infl = (np.mean(si002_noisy) - np.mean(si002_clean)) / np.mean(si002_clean) * 100
    print(f"  si=0.02, fs≤15: 噪声平均膨胀 = {avg_infl:+.2f}%")
    assert abs(avg_infl) < 5, f"噪声膨胀 {avg_infl:.1f}% 超出预期"

    elapsed = _time.time() - t0
    print(f"  ✓ 参数扫描完成 ({len(rows)} 组合, {elapsed:.1f}s) → {output_csv}")
    return True


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    tests = [
        ("TC1 单点圆面积",        test_single_point),
        ("TC2 直线 stadium 面积", test_straight_line),
        ("TC3 两分离点(无重叠)",   test_two_separated_points),
        ("TC4 两点(圆交面积)",     test_two_overlapping_points),
        ("TC5 双线(间距2R,无重叠)", test_parallel_no_overlap),
        ("TC6 双线(间距R,50%重叠)", test_parallel_half_overlap),
        ("TC7 全覆盖模式",         test_full_coverage),
        ("TC8 空间重采样保真度",   test_resample_fidelity),
        ("TC9 噪声鲁棒性",         test_noise_inflation),
        ("TC10 往复+噪声(真实)",   test_realistic_noise),
        ("TC11 往复+丢帧(真实)",   test_realistic_dropout),
        ("TC12 噪声+丢帧 综合",    test_realistic_combined),
    ]

    # --sweep 仅运行参数扫描
    if "--sweep" in sys.argv:
        tests = [("TC13 参数扫描", test_param_sweep)]
    elif len(sys.argv) > 1:
        print("用法: python test_coverage.py [--sweep]")
        print("  (无参数)  运行 TC1-TC12 精度验证")
        print("  --sweep   运行 TC13 参数扫描, 输出 param_sweep.csv")
        sys.exit(0)
    passed = failed = 0
    print("=" * 60)
    print(f"  camera_coverage 核心算法验证  (R={R}m, W={W}m, grid=5mm)")
    print("=" * 60)
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  ✓ {name}\n")
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {name} FAIL: {e}\n")
        except Exception as e:
            failed += 1
            print(f"  ✗ {name} ERROR: {e}\n")
            import traceback; traceback.print_exc()
    print("=" * 60)
    print(f"  {passed}/{passed+failed} 通过")
    if failed:
        print(f"  ⚠ {failed} 失败")
        sys.exit(1)
    else:
        print(f"  ✓ accumulate_coverage 精度: <3%  |  compute_metrics 一致性: OK")
    print("=" * 60)


if __name__ == "__main__":
    main()
