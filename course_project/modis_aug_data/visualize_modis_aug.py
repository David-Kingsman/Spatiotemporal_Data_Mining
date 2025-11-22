#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" data visualizer for MODIS_Aug.mat
Functions:
1. Read the MODIS 2020 Aug LST data (training_tensor / test_tensor)
2. Combine the 1-31 days of 100x200 temperature fields into a large figure (5x7 subplots)
3. Draw:
   - Daily global mean temperature time series
   - Daily missing ratio time series
   - Monthly average temperature spatial distribution

Usage:
python3 modis_aug_data/visualize_modis_aug.py --mat_path modis_aug_data/MODIS_Aug.mat --save_dir modis_aug_data/data_visualization
"""

import os
import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 150

# Region range according to the readme
LAT_MIN, LAT_MAX = 35.0, 40.0
LON_MIN, LON_MAX = -115.0, -105.0


def load_modis_aug(mat_path):
    """Load MODIS_Aug.mat and return training_tensor, test_tensor (if exists)"""
    data = sio.loadmat(mat_path)
    training = data.get("training_tensor", None)
    test = data.get("test_tensor", None)

    if training is None:
        raise ValueError("'training_tensor' key not found in .mat file, please check the file content.")
    # ensure float32
    training = training.astype(np.float32)
    if test is not None:
        test = test.astype(np.float32)
    return training, test

def kelvin_to_celsius(arr):
    """Kelvin -> Celsius (keep missing values np.nan unchanged)"""
    return arr - 273.15

def mask_missing(arr, missing_value=0.0):
    """Replace 0 with np.nan for visualization and statistics"""
    arr = arr.copy()
    arr[arr == missing_value] = np.nan
    return arr

def get_lat_lon_grids(lat_size=100, lon_size=200,
                      lat_min=35.0, lat_max=40.0,
                      lon_min=-115.0, lon_max=-105.0):
    """
    Construct latitude/longitude grid (according to the readme: 35~40, -115~-105)
    """
    lats = np.linspace(lat_min, lat_max, lat_size)
    lons = np.linspace(lon_min, lon_max, lon_size)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return lat_grid, lon_grid


def plot_all_days_grid(tensor, title_prefix="Training", save_path=None,
                       cmap="viridis", vmin=None, vmax=None,
                       to_celsius=True):
    """
    将 31 天数据拼成一张 8x4 的 grid 图（8行4列）。
    tensor: (H, W, T) = (100, 200, 31)
    """
    H, W, T = tensor.shape
    assert T == 31, "当前时间维度为 {}，本脚本假定为 31 天。".format(T)

    data = mask_missing(tensor, missing_value=0.0)
    if to_celsius:
        data = kelvin_to_celsius(data)

    # 如果未指定 vmin/vmax，就根据所有天的有效值统一设定色标范围
    if vmin is None or vmax is None:
        vmin = np.nanpercentile(data, 5)
        vmax = np.nanpercentile(data, 95)

    # 8行4列布局（8×4=32，最后一行只有3个图）
    fig, axes = plt.subplots(8, 4, figsize=(16, 20),
                             constrained_layout=True)
    axes = axes.ravel()  # 展平成一维 array

    for day in range(T):
        ax = axes[day]
        im = ax.imshow(data[:, :, day],
                       origin="lower",
                       cmap=cmap,
                       vmin=vmin,
                       vmax=vmax,
                       extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],  # 设置坐标范围
                       aspect='auto')
        ax.set_title("Day {}".format(day + 1), fontsize=10)
        
        # 设置坐标轴标签和刻度
        if day >= 28:  # 最后一行显示x轴标签（28, 29, 30）
            ax.set_xlabel("Longitude", fontsize=8)
            ax.set_xticks([LON_MIN, (LON_MIN + LON_MAX) / 2, LON_MAX])
            ax.set_xticklabels(["{:.0f}°".format(LON_MIN), 
                               "{:.0f}°".format((LON_MIN + LON_MAX) / 2),
                               "{:.0f}°".format(LON_MAX)], fontsize=7)
        else:
            ax.set_xticks([])
        
        if day % 4 == 0:  # 第一列显示y轴标签（0, 4, 8, 12, 16, 20, 24, 28）
            ax.set_ylabel("Latitude", fontsize=8)
            ax.set_yticks([LAT_MIN, (LAT_MIN + LAT_MAX) / 2, LAT_MAX])
            ax.set_yticklabels(["{:.0f}°".format(LAT_MIN),
                               "{:.0f}°".format((LAT_MIN + LAT_MAX) / 2),
                               "{:.0f}°".format(LAT_MAX)], fontsize=7)
        else:
            ax.set_yticks([])
        
        ax.tick_params(labelsize=7)

    # 隐藏最后一个多余的子图（第32个）
    axes[31].axis("off")

    # 添加颜色条
    cbar = fig.colorbar(im, ax=list(axes[:31]), shrink=0.8)
    cbar.set_label("Temperature (°C)" if to_celsius else "Temperature (K)", fontsize=12)

    fig.suptitle("{}: LST 1–31 Days (H={}, W={})".format(title_prefix, H, W), 
                 fontsize=16, fontweight='bold')

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print("[保存] {}".format(save_path))

    return fig


def plot_time_series_stats(tensor, title_prefix="Training", save_dir=None):
    """
    画：
    1. 每天 global mean 温度（只在非缺失点统计）
    2. 每天 missing ratio
    """
    data = tensor.copy()
    missing_mask = (data == 0.0)
    data = mask_missing(data, missing_value=0.0)
    data_c = kelvin_to_celsius(data)

    H, W, T = data.shape
    days = np.arange(1, T + 1)

    # global mean per day
    daily_mean = np.nanmean(data_c, axis=(0, 1))
    # missing ratio per day
    daily_missing_ratio = missing_mask.mean(axis=(0, 1))

    # 1) mean temperature
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(days, daily_mean, marker="o")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Mean LST (°C)")
    ax1.set_title("{}: Daily Mean Land Surface Temperature".format(title_prefix))
    ax1.grid(True)

    if save_dir is not None:
        path1 = os.path.join(save_dir, "{}_daily_mean_temp.png".format(title_prefix.lower()))
        fig1.savefig(path1, bbox_inches="tight")
        print("[保存] {}".format(path1))

    # 2) missing ratio
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(days, daily_missing_ratio, marker="o")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Missing ratio")
    ax2.set_title("{}: Daily Missing Ratio (0 as missing)".format(title_prefix))
    ax2.grid(True)
    ax2.set_ylim(0.0, 1.0)

    if save_dir is not None:
        path2 = os.path.join(save_dir, "{}_daily_missing_ratio.png".format(title_prefix.lower()))
        fig2.savefig(path2, bbox_inches="tight")
        print("[保存] {}".format(path2))

    return (fig1, fig2)


def plot_monthly_mean_map(tensor, title_prefix="Training",
                          save_path=None, cmap="viridis",
                          to_celsius=True):
    """
    全月平均温度的空间分布：对时间维度求 mean。
    """
    data = mask_missing(tensor, missing_value=0.0)
    if to_celsius:
        data = kelvin_to_celsius(data)

    monthly_mean = np.nanmean(data, axis=2)  # (H, W)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(monthly_mean,
                   origin="lower",
                   cmap=cmap,
                   extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],  # 设置坐标范围
                   aspect='auto')
    ax.set_title("{}: Monthly Mean LST".format(title_prefix), fontsize=12, fontweight='bold')
    
    # 设置坐标轴标签和刻度
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_xticks(np.linspace(LON_MIN, LON_MAX, 5))
    ax.set_xticklabels(["{:.1f}°".format(x) for x in np.linspace(LON_MIN, LON_MAX, 5)], fontsize=9)
    ax.set_yticks(np.linspace(LAT_MIN, LAT_MAX, 5))
    ax.set_yticklabels(["{:.1f}°".format(y) for y in np.linspace(LAT_MIN, LAT_MAX, 5)], fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Temperature (°C)" if to_celsius else "Temperature (K)", fontsize=11)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print("[保存] {}".format(save_path))

    return fig




def main():
    parser = argparse.ArgumentParser(description="MODIS_Aug Data Visualizer")
    parser.add_argument("--mat_path", type=str, required=True,
                        help="Path to MODIS_Aug.mat (包含 training_tensor / test_tensor)")
    parser.add_argument("--save_dir", type=str, default="modis_aug_data/figs",
                        help="图片保存目录（默认 figs）")
    parser.add_argument("--show", action="store_true",
                        help="是否在生成后显示图像（默认不显示，只保存）")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    training, test = load_modis_aug(args.mat_path)
    print("[INFO] training_tensor shape = {}".format(training.shape))
    if test is not None:
        print("[INFO] test_tensor shape = {}".format(test.shape))

    # 1. 1-31 天 grid 图（training）
    grid_path = os.path.join(args.save_dir, "training_all_days_grid.png")
    plot_all_days_grid(training, title_prefix="Training", save_path=grid_path)

    # 2. time series stats（training）
    plot_time_series_stats(training, title_prefix="Training", save_dir=args.save_dir)

    # 3. monthly mean map（training）
    monthly_path = os.path.join(args.save_dir, "training_monthly_mean.png")
    plot_monthly_mean_map(training, title_prefix="Training", save_path=monthly_path)

    # 4. Temperature distribution of all days (单独保存，便于报告使用，8行4列布局)
    temp_all_path = os.path.join(args.save_dir, "temperature_distribution_all_days.png")
    plot_all_days_grid(training, title_prefix="Temperature Distribution", save_path=temp_all_path)

    # 如果你也想对 test_tensor 做同样的可视化（而且 test 不是全 0），可以取消下面的注释
    if test is not None:
        test_grid_path = os.path.join(args.save_dir, "test_all_days_grid.png")
        plot_all_days_grid(test, title_prefix="Test", save_path=test_grid_path)

        plot_time_series_stats(test, title_prefix="Test", save_dir=args.save_dir)

        test_monthly_path = os.path.join(args.save_dir, "test_monthly_mean.png")
        plot_monthly_mean_map(test, title_prefix="Test", save_path=test_monthly_path)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
