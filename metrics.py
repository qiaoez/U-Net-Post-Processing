import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from datetime import datetime, timedelta
import pandas as pd
import xarray as xr
import numpy as np
import os
import glob
import re
from tqdm import tqdm
import warnings
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
import pickle
from models import UNet
def calculate_tcc_rmse_comparison(dataset, test_years=range(2017, 2021)):
    """计算 ECMWF 与 U-Net 的 TCC 和 RMSE（返回区域平均和网格数据）"""

    print("计算TCC和RMSE...")

    # 初始化结果存储 - 区域平均和网格数据
    tcc_results = {'ECMWF': {}, 'U-Net': {}}
    tcc_grid_results = {'ECMWF': {}, 'U-Net': {}}
    rmse_results = {'ECMWF': {}, 'U-Net': {}}
    rmse_grid_results = {'ECMWF': {}, 'U-Net': {}}

    cn_valid_mask = dataset.get_cn_valid_mask()

    for lead_time in range(1, 5):
        print(f"\n处理提前期 {lead_time} 周...")

        # 获取测试数据
        ec_test, obs_test = dataset.get_training_data_by_lead(lead_time, test_years)

        if len(ec_test) == 0:
            print(f"提前期 {lead_time} 周没有测试数据，跳过")
            continue

        # 1. ECMWF原始预报
        print("计算ECMWF的TCC和RMSE...")
        ec_tcc, ec_rmse, ec_tcc_grid, ec_rmse_grid = calculate_method_tcc_rmse_grid(ec_test, obs_test, cn_valid_mask,
                                                                                    'ECMWF')
        tcc_results['ECMWF'][lead_time] = ec_tcc
        tcc_grid_results['ECMWF'][lead_time] = ec_tcc_grid  # 保存网格数据
        rmse_results['ECMWF'][lead_time] = ec_rmse
        rmse_grid_results['ECMWF'][lead_time] = ec_rmse_grid  # 保存网格数据

        # 2. U-Net预报
        print("计算U-Net的TCC和RMSE...")
        unet_forecasts = calculate_unet_forecasts(dataset, lead_time, test_years)
        unet_tcc, unet_rmse, unet_tcc_grid, unet_rmse_grid = calculate_method_tcc_rmse_grid(unet_forecasts, obs_test,
                                                                                            cn_valid_mask, 'U-Net')
        tcc_results['U-Net'][lead_time] = unet_tcc
        tcc_grid_results['U-Net'][lead_time] = unet_tcc_grid  # 保存网格数据
        rmse_results['U-Net'][lead_time] = unet_rmse
        rmse_grid_results['U-Net'][lead_time] = unet_rmse_grid  # 保存网格数据

    # 返回区域平均结果和网格数据
    return tcc_results, rmse_results, tcc_grid_results, rmse_grid_results


# RMSE 与 TCC 的最低有效样本数：
# RMSE 是逐格点的均方根误差，2 个有效样本即可计算；TCC 是时间相关系数，
# 样本太少会得到 ±1 的伪相关，因此仍要求至少 5 个样本（与改动前一致）。
MIN_RMSE_SAMPLES = 2
MIN_TCC_SAMPLES = 5


def calculate_method_tcc_rmse_grid(forecasts, observations, valid_mask, method_name):
    """计算单个方法的TCC和RMSE（返回区域平均和网格数据）"""

    if len(forecasts) == 0 or len(observations) == 0:
        print(f"{method_name}: 没有数据")
        return np.nan, np.nan, np.nan, np.nan

    # 初始化存储每个网格点的TCC和RMSE
    grid_tcc = np.full(forecasts[0].shape, np.nan)  # 初始化为NaN
    grid_rmse = np.full(forecasts[0].shape, np.nan)  # 初始化为NaN
    grid_valid = np.zeros(forecasts[0].shape, dtype=bool)

    # 对每个网格点计算
    for i in range(forecasts[0].shape[0]):
        for j in range(forecasts[0].shape[1]):
            # 只处理有效网格点
            if valid_mask[i, j]:
                # 提取该网格点的时间序列
                pred_series = []
                obs_series = []

                for t in range(len(forecasts)):
                    pred_val = forecasts[t][i, j]
                    obs_val = observations[t][i, j]

                    # 只使用有效数据点
                    if not np.isnan(pred_val) and not np.isnan(obs_val):
                        pred_series.append(pred_val)
                        obs_series.append(obs_val)

                # RMSE 只要求至少 2 个有效样本（不依赖相关系数的样本量要求）
                if len(pred_series) >= MIN_RMSE_SAMPLES:
                    pred_series = np.array(pred_series)
                    obs_series = np.array(obs_series)

                    # 计算RMSE
                    rmse_val = np.sqrt(np.mean((pred_series - obs_series) ** 2))
                    grid_rmse[i, j] = rmse_val

                    # 计算TCC（时间相关系数）：仍要求至少 5 个样本
                    if len(pred_series) >= MIN_TCC_SAMPLES:
                        if np.std(pred_series) > 0 and np.std(obs_series) > 0:
                            tcc_val = np.corrcoef(pred_series, obs_series)[0, 1]
                            grid_tcc[i, j] = tcc_val

                    grid_valid[i, j] = True

    # 计算区域平均
    if np.any(grid_valid):
        valid_tcc = grid_tcc[grid_valid]
        # 有效格点可能都凑不到 5 个样本（例如只有 4 年的验证期），此时 TCC 记为 NaN
        mean_tcc = float(np.nanmean(valid_tcc)) if np.any(~np.isnan(valid_tcc)) else np.nan
        mean_rmse = float(np.nanmean(grid_rmse[grid_valid]))
        print(f"{method_name}: TCC={mean_tcc:.3f}, RMSE={mean_rmse:.3f}")
        return mean_tcc, mean_rmse, grid_tcc, grid_rmse
    else:
        print(f"{method_name}: 没有有效网格点")
        return np.nan, np.nan, grid_tcc, grid_rmse




def calculate_unet_forecasts(dataset, lead_time, test_years):
    """计算U-Net预报"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model_path = f'/3240101047/qiaoez/unet_lead_{lead_time}_best.pth'
    if not os.path.exists(model_path):
        print(f"U-Net模型文件不存在: {model_path}")
        return []

    # 所有提前期都使用U-Net模型
    model = UNet(in_channels=20, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()

    # 获取测试数据
    test_indices = [i for i, (_, _, _, lt, year) in enumerate(dataset.data)
                    if lt == lead_time and year in test_years]

    if not test_indices:
        return []

    test_dataset = Subset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    all_predictions = []

    with torch.no_grad():
        for inputs, _, _, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_predictions.append(outputs.cpu().numpy())

    if all_predictions:
        return np.concatenate(all_predictions, axis=0).squeeze()
    else:
        return []








# ============================================================================
# 新增图1：预报方差验证图（针对审稿人意见 R1-Major3）
# 说明：后处理模型通常会显著降低预报方差，需验证 U-Net 是否过度抑制变率。
#       本函数计算 ECMWF、U-Net、观测在 Week 1-4 的预报方差空间分布，
#       并绘制 4 行 3 列对比图（ECMWF方差 | U-Net方差 | U-Net minus ECMWF差值）。
#       仅需缓存数据 + U-Net 推理（不重训模型）。
# ============================================================================

def calculate_variance_comparison(dataset, test_years=range(2017, 2021), device=None):
    """计算 OBS、ECMWF、U-Net 在 Week 1-4 的预报方差空间分布及偏差

    参数:
        dataset: WeeklyEC_S2S_Dataset 实例（从缓存加载）
        test_years: 测试年份范围
        device: torch device（用于 U-Net 推理）

    返回:
        variance_results: dict, 结构为
            {
                'OBS':        {1: (H, W) ndarray, 2: ..., 3: ..., 4: ...},
                'ECMWF':      {1: (H, W) ndarray, 2: ..., 3: ..., 4: ...},
                'U-Net':      {1: (H, W) ndarray, 2: ..., 3: ..., 4: ...},
                'ECMWF_DIFF': {1: (H, W) ndarray, 2: ..., 3: ..., 4: ...},  # ECMWF - OBS
                'UNET_DIFF':  {1: (H, W) ndarray, 2: ..., 3: ..., 4: ...},  # U-Net - OBS
            }
    """
    print("=" * 60)
    print("计算预报方差对比（OBS vs ECMWF vs U-Net）...")
    print("=" * 60)

    cn_valid_mask = dataset.get_cn_valid_mask()

    variance_results = {
        'OBS': {}, 'ECMWF': {}, 'U-Net': {},
        'ECMWF_DIFF': {}, 'UNET_DIFF': {}
    }

    for lead_time in range(1, 5):
        print(f"\n处理提前期 {lead_time} 周...")

        # 获取测试数据：ec_test 形状 (T, H, W)，obs_test 形状 (T, H, W)
        ec_test, obs_test = dataset.get_training_data_by_lead(lead_time, test_years)

        if len(ec_test) == 0:
            print(f"  提前期 {lead_time} 周没有测试数据，跳过")
            continue

        # ---- 观测方差 ----
        obs_var_grid = np.nanvar(obs_test, axis=0, ddof=1)  # 样本方差
        # ---- ECMWF 方差（逐网格点的时间方差）----
        ec_var_grid = np.nanvar(ec_test, axis=0, ddof=1)

        # ---- U-Net 方差：需要加载模型推理 ----
        print(f"  加载 U-Net 模型并推理（提前期 {lead_time} 周）...")
        unet_forecasts = calculate_unet_forecasts(dataset, lead_time, test_years)
        # unet_forecasts 形状: (T, H, W)
        unet_var_grid = np.nanvar(unet_forecasts, axis=0, ddof=1)

        # 应用有效掩膜
        obs_var_grid = np.where(cn_valid_mask, obs_var_grid, np.nan)
        ec_var_grid = np.where(cn_valid_mask, ec_var_grid, np.nan)
        unet_var_grid = np.where(cn_valid_mask, unet_var_grid, np.nan)

        # 偏差：相对于 OBS
        ec_diff_grid = ec_var_grid - obs_var_grid   # ECMWF - OBS
        unet_diff_grid = unet_var_grid - obs_var_grid  # U-Net - OBS

        variance_results['OBS'][lead_time] = obs_var_grid
        variance_results['ECMWF'][lead_time] = ec_var_grid
        variance_results['U-Net'][lead_time] = unet_var_grid
        variance_results['ECMWF_DIFF'][lead_time] = ec_diff_grid
        variance_results['UNET_DIFF'][lead_time] = unet_diff_grid

        # 打印区域平均统计
        obs_mean = np.nanmean(obs_var_grid)
        ec_mean = np.nanmean(ec_var_grid)
        unet_mean = np.nanmean(unet_var_grid)
        ec_diff_mean = np.nanmean(ec_diff_grid)
        unet_diff_mean = np.nanmean(unet_diff_grid)
        ec_bias_pct = (ec_mean - obs_mean) / obs_mean * 100 if obs_mean > 0 else np.nan
        unet_bias_pct = (unet_mean - obs_mean) / obs_mean * 100 if obs_mean > 0 else np.nan
        print(f"  OBS   方差均值: {obs_mean:.4f}")
        print(f"  ECMWF 方差均值: {ec_mean:.4f} (相对OBS偏差: {ec_bias_pct:+.1f}%)")
        print(f"  U-Net 方差均值: {unet_mean:.4f} (相对OBS偏差: {unet_bias_pct:+.1f}%)")
        print(f"  ECMWF - OBS 均值: {ec_diff_mean:.4f}")
        print(f"  U-Net - OBS 均值: {unet_diff_mean:.4f}")

    print("\n预报方差对比计算完成。")
    return variance_results




# ============================================================================
# 新增图2：ACC 分布箱线图（针对审稿人意见 R1-Minor3）
# 说明：原稿仅给出 ACC 均值（0.74-0.77），审稿人希望展示 ACC 的空间分布
#       而非仅均值。本函数计算每个时间样本的 ACC（预报与观测的空间相关系数），
#       绘制 Week 1-4 的 ACC 箱线图，ECMWF vs U-Net 并排对比。
#       仅需缓存数据 + U-Net 推理（不重训模型）。
# ============================================================================

def calculate_acc_per_sample(dataset, test_years=range(2017, 2021), device=None):
    """计算每个时间样本的 ACC（预报与观测的空间相关系数）

    对每个时间步 t，计算 forecast[t] 与 observation[t] 在 TP 有效网格点上的
    皮尔逊相关系数，得到长度为 T 的 ACC 序列。

    参数:
        dataset: WeeklyEC_S2S_Dataset 实例
        test_years: 测试年份范围
        device: torch device（用于 U-Net 推理）

    返回:
        acc_results: dict, 结构为
            {
                'ECMWF': {1: ndarray (T,), 2: ..., 3: ..., 4: ...},
                'U-Net': {1: ndarray (T,), 2: ..., 3: ..., 4: ...},
            }
    """
    print("=" * 60)
    print("计算逐样本 ACC（ECMWF vs U-Net）...")
    print("=" * 60)

    cn_valid_mask = dataset.get_cn_valid_mask()

    acc_results = {'ECMWF': {}, 'U-Net': {}}

    for lead_time in range(1, 5):
        print(f"\n处理提前期 {lead_time} 周...")

        # 获取测试数据
        ec_test, obs_test = dataset.get_training_data_by_lead(lead_time, test_years)

        if len(ec_test) == 0:
            print(f"  提前期 {lead_time} 周没有测试数据，跳过")
            continue

        T = len(ec_test)

        # ---- ECMWF 逐样本 ACC ----
        ec_acc_list = []
        for t in range(T):
            pred = ec_test[t]
            obs = obs_test[t]
            # 应用有效掩膜，剔除 NaN
            pred_flat = pred[cn_valid_mask]
            obs_flat = obs[cn_valid_mask]
            # 进一步剔除 pred 或 obs 中的 NaN
            valid = ~(np.isnan(pred_flat) | np.isnan(obs_flat))
            pred_flat = pred_flat[valid]
            obs_flat = obs_flat[valid]
            if len(pred_flat) < 2:
                ec_acc_list.append(np.nan)
                continue
            # 计算皮尔逊相关系数
            pred_mean = pred_flat.mean()
            obs_mean = obs_flat.mean()
            pred_std = pred_flat.std()
            obs_std = obs_flat.std()
            if pred_std < 1e-10 or obs_std < 1e-10:
                ec_acc_list.append(np.nan)
                continue
            acc = np.mean((pred_flat - pred_mean) * (obs_flat - obs_mean)) / (pred_std * obs_std)
            ec_acc_list.append(acc)
        ec_acc = np.array(ec_acc_list)

        # ---- U-Net 逐样本 ACC ----
        print(f"  加载 U-Net 模型并推理（提前期 {lead_time} 周）...")
        unet_forecasts = calculate_unet_forecasts(dataset, lead_time, test_years)
        # unet_forecasts 形状: (T, H, W)
        unet_acc_list = []
        for t in range(T):
            pred = unet_forecasts[t]
            obs = obs_test[t]
            pred_flat = pred[cn_valid_mask]
            obs_flat = obs[cn_valid_mask]
            valid = ~(np.isnan(pred_flat) | np.isnan(obs_flat))
            pred_flat = pred_flat[valid]
            obs_flat = obs_flat[valid]
            if len(pred_flat) < 2:
                unet_acc_list.append(np.nan)
                continue
            pred_mean = pred_flat.mean()
            obs_mean = obs_flat.mean()
            pred_std = pred_flat.std()
            obs_std = obs_flat.std()
            if pred_std < 1e-10 or obs_std < 1e-10:
                unet_acc_list.append(np.nan)
                continue
            acc = np.mean((pred_flat - pred_mean) * (obs_flat - obs_mean)) / (pred_std * obs_std)
            unet_acc_list.append(acc)
        unet_acc = np.array(unet_acc_list)

        acc_results['ECMWF'][lead_time] = ec_acc
        acc_results['U-Net'][lead_time] = unet_acc

        # 打印统计
        ec_mean = np.nanmean(ec_acc)
        ec_std = np.nanstd(ec_acc)
        unet_mean = np.nanmean(unet_acc)
        unet_std = np.nanstd(unet_acc)
        print(f"  ECMWF ACC: 均值={ec_mean:.4f}, 标准差={ec_std:.4f}, "
              f"最小={np.nanmin(ec_acc):.4f}, 最大={np.nanmax(ec_acc):.4f}, "
              f"样本数={np.sum(~np.isnan(ec_acc))}")
        print(f"  U-Net ACC: 均值={unet_mean:.4f}, 标准差={unet_std:.4f}, "
              f"最小={np.nanmin(unet_acc):.4f}, 最大={np.nanmax(unet_acc):.4f}, "
              f"样本数={np.sum(~np.isnan(unet_acc))}")

    print("\n逐样本 ACC 计算完成。")
    return acc_results


