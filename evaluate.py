"""
评估模块
包含各种评估函数
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from scipy.interpolate import interp1d
from tqdm import tqdm
import os

from models import UNet
from config import *




def calculate_grid_metrics_all_points(pred, obs, threshold):
    """计算网格指标"""
    pred_events = pred > threshold
    obs_events = obs > threshold

    hits = np.logical_and(pred_events, obs_events).astype(float)
    false_alarms = np.logical_and(pred_events, ~obs_events).astype(float)
    misses = np.logical_and(~pred_events, obs_events).astype(float)
    correct_negatives = np.logical_and(~pred_events, ~obs_events).astype(float)

    hits = np.nan_to_num(hits, nan=0.0)
    false_alarms = np.nan_to_num(false_alarms, nan=0.0)
    misses = np.nan_to_num(misses, nan=0.0)
    correct_negatives = np.nan_to_num(correct_negatives, nan=0.0)

    return hits, false_alarms, misses, correct_negatives


def calculate_grid_metrics_all_points_v2(pred, obs, threshold, valid_mask):
    """改进的网格指标计算，只对有效点计算指标"""
    hits = np.zeros_like(pred, dtype=float)
    false_alarms = np.zeros_like(pred, dtype=float)
    misses = np.zeros_like(pred, dtype=float)
    correct_negatives = np.zeros_like(pred, dtype=float)

    valid_indices = np.where(valid_mask)

    for i, j in zip(valid_indices[0], valid_indices[1]):
        if not np.isnan(pred[i, j]) and not np.isnan(obs[i, j]) and not np.isnan(threshold[i, j]):
            pred_event = pred[i, j] > threshold[i, j]
            obs_event = obs[i, j] > threshold[i, j]

            if pred_event and obs_event:
                hits[i, j] = 1
            elif pred_event and not obs_event:
                false_alarms[i, j] = 1
            elif not pred_event and obs_event:
                misses[i, j] = 1
            else:
                correct_negatives[i, j] = 1

    return hits, false_alarms, misses, correct_negatives


def calculate_metrics_per_grid(hits, false_alarms, misses, correct_negatives):
    """计算每个网格点的技能评分 - 修改：返回TS评分"""
    # 计算TS评分
    denominator = hits + false_alarms + misses
    ts = np.zeros_like(hits, dtype=float)
    valid_mask = denominator > 0
    ts[valid_mask] = hits[valid_mask] / denominator[valid_mask]

    # 计算HSS评分
    hss_numerator = (hits * correct_negatives - false_alarms * misses) * 2
    hss_denominator = (hits + misses) * (misses + correct_negatives) + (hits + false_alarms) * (
                false_alarms + correct_negatives)
    hss = np.zeros_like(hits, dtype=float)
    valid_mask = hss_denominator > 0
    hss[valid_mask] = hss_numerator[valid_mask] / hss_denominator[valid_mask]

    # 计算POD评分
    pod = np.zeros_like(hits, dtype=float)
    denominator = hits + misses
    valid_mask = denominator > 0
    pod[valid_mask] = hits[valid_mask] / denominator[valid_mask]

    # 计算POFD评分（如果需要保留）
    pofd = np.zeros_like(hits, dtype=float)
    denominator = false_alarms + correct_negatives
    valid_mask = denominator > 0
    pofd[valid_mask] = false_alarms[valid_mask] / denominator[valid_mask]

    # 计算FAR评分
    far = np.zeros_like(hits, dtype=float)
    denominator = hits + false_alarms
    valid_mask = denominator > 0
    far[valid_mask] = false_alarms[valid_mask] / denominator[valid_mask]

    return hss, pod, ts, far, pofd  # 修改：返回TS评分




def evaluate_model(model, dataset, test_years=TEST_YEARS, batch_size=BATCH_SIZE):
    """评估模型性能 - 修改：添加TS评分"""
    device = torch.device('cuda' if torch.cuda.is_available() and USE_CUDA else 'cpu')
    model.eval()

    test_indices = [i for i, (_, _, _, _, year) in enumerate(dataset.data) if year in test_years]
    test_dataset = Subset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    thresholds = dataset.get_thresholds()
    cn_valid_mask = dataset.get_cn_valid_mask()

    lead_counts = {
        lead: {
            'hits': np.zeros(dataset.grid_size),
            'false_alarms': np.zeros(dataset.grid_size),
            'misses': np.zeros(dataset.grid_size),
            'correct_negatives': np.zeros(dataset.grid_size),
            'total_samples': np.zeros(dataset.grid_size)
        } for lead in range(1, 5)
    }

    with torch.no_grad():
        for inputs, targets, masks, lead_times in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)

            preds_np = outputs.cpu().numpy()
            targets_np = targets.numpy()
            masks_np = masks.numpy()
            lead_times_np = lead_times.numpy()

            for b in range(preds_np.shape[0]):
                lead_time = int(lead_times_np[b])
                pred_map = preds_np[b].squeeze()
                target_map = targets_np[b]
                mask_map = masks_np[b]

                hits, false_alarms, misses, correct_negatives = calculate_grid_metrics_all_points_v2(
                    pred_map, target_map, thresholds, cn_valid_mask
                )

                lead_counts[lead_time]['hits'] += hits
                lead_counts[lead_time]['false_alarms'] += false_alarms
                lead_counts[lead_time]['misses'] += misses
                lead_counts[lead_time]['correct_negatives'] += correct_negatives
                lead_counts[lead_time]['total_samples'] += (cn_valid_mask & ~np.isnan(target_map))

    results = {}
    for lead in range(1, 5):
        counts = lead_counts[lead]
        hits = counts['hits']
        false_alarms = counts['false_alarms']
        misses = counts['misses']
        correct_negatives = counts['correct_negatives']
        total_samples = counts['total_samples']

        # 修改：接收TS评分
        hss, pod, ts, far, pofd = calculate_metrics_per_grid(hits, false_alarms, misses, correct_negatives)

        valid_mask = cn_valid_mask & (total_samples > 0)

        if np.any(valid_mask):
            results[lead] = {
                'hss': np.nanmean(hss[valid_mask]),
                'pod': np.nanmean(pod[valid_mask]),
                'ts': np.nanmean(ts[valid_mask]),  # 修改：添加TS评分
                'far': np.nanmean(far[valid_mask]),
                'hss_grid': hss,
                'pod_grid': pod,
                'ts_grid': ts,  # 修改：添加TS网格数据
                'far_grid': far,
                'sample_count': total_samples
            }
        else:
            results[lead] = {
                'hss': np.nan, 'pod': np.nan, 'ts': np.nan, 'far': np.nan,  # 修改：添加TS
                'hss_grid': np.full(dataset.grid_size, np.nan),
                'pod_grid': np.full(dataset.grid_size, np.nan),
                'ts_grid': np.full(dataset.grid_size, np.nan),  # 修改：添加TS网格
                'far_grid': np.full(dataset.grid_size, np.nan),
                'sample_count': total_samples
            }

    return results


def evaluate_unet_0115(dataset, model_paths=None, train_years=TRAIN_YEARS):
    """评估2001-2015年U-Net模型的性能 - 修改：添加TS评分"""
    thresholds = dataset.get_thresholds()
    cn_valid_mask = dataset.get_cn_valid_mask()

    unet_results = {}

    device = torch.device('cuda' if torch.cuda.is_available() and USE_CUDA else 'cpu')

    for lead_time in range(1, 5):
        print(f"\n评估提前期 {lead_time} 周的U-Net模型在2001-2015年的性能...")

        if model_paths and lead_time in model_paths:
            model_path = model_paths[lead_time]
        else:
            model_path = f'unet_lead_{lead_time}_best.pth'

        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}，跳过")
            continue

        model = UNet(in_channels=20, out_channels=1).to(device)
        model.load_state_dict(torch.load(model_path))

        model.eval()

        indices = [i for i, (_, _, _, lt, year) in enumerate(dataset.data)
                   if lt == lead_time and year in train_years]

        if len(indices) == 0:
            print(f"提前期 {lead_time} 周在2001-2015年没有数据，跳过")
            continue

        test_dataset = Subset(dataset, indices)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

        unet_hits = np.zeros(dataset.grid_size)
        unet_false_alarms = np.zeros(dataset.grid_size)
        unet_misses = np.zeros(dataset.grid_size)
        unet_correct_negatives = np.zeros(dataset.grid_size)

        with torch.no_grad():
            for inputs, targets, masks, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)

                preds_np = outputs.cpu().numpy()
                targets_np = targets.numpy()
                masks_np = masks.numpy()

                for b in range(preds_np.shape[0]):
                    pred_map = preds_np[b].squeeze()
                    target_map = targets_np[b]
                    mask_map = masks_np[b]

                    hits, false_alarms, misses, correct_negatives = calculate_grid_metrics_all_points_v2(
                        pred_map, target_map, thresholds, cn_valid_mask
                    )

                    unet_hits += hits
                    unet_false_alarms += false_alarms
                    unet_misses += misses
                    unet_correct_negatives += correct_negatives

        unet_hits = np.where(cn_valid_mask, unet_hits, np.nan)
        unet_false_alarms = np.where(cn_valid_mask, unet_false_alarms, np.nan)
        unet_misses = np.where(cn_valid_mask, unet_misses, np.nan)
        unet_correct_negatives = np.where(cn_valid_mask, unet_correct_negatives, np.nan)

        # 修改：接收TS评分
        unet_hss, unet_pod, unet_ts, unet_far, unet_pofd = calculate_metrics_per_grid(
            unet_hits, unet_false_alarms, unet_misses, unet_correct_negatives
        )

        valid_mask = cn_valid_mask & (unet_hits + unet_false_alarms + unet_misses + unet_correct_negatives > 0)

        unet_results[lead_time] = {
            'hss': np.nanmean(unet_hss[valid_mask]) if np.any(valid_mask) else np.nan,
            'pod': np.nanmean(unet_pod[valid_mask]) if np.any(valid_mask) else np.nan,
            'ts': np.nanmean(unet_ts[valid_mask]) if np.any(valid_mask) else np.nan,  # 修改：添加TS评分
            'far': np.nanmean(unet_far[valid_mask]) if np.any(valid_mask) else np.nan,
            'hss_grid': unet_hss,
            'pod_grid': unet_pod,
            'ts_grid': unet_ts,  # 修改：添加TS网格数据
            'far_grid': unet_far
        }

        print(f"提前期 {lead_time} 周 - U-Net(2001-2015): HSS={unet_results[lead_time]['hss']:.3f}, TS={unet_results[lead_time]['ts']:.3f}")

    return unet_results


def evaluate_ec_period(dataset, years):
    """评估ECMWF在指定时期的性能 - 修改：添加TS评分"""
    thresholds = dataset.get_thresholds()
    cn_valid_mask = dataset.get_cn_valid_mask()

    ec_results = {}

    for lead_time in range(1, 5):
        print(f"评估ECMWF提前期 {lead_time} 周...")

        ec_data, obs_data = dataset.get_training_data_by_lead(lead_time, years)

        if len(ec_data) == 0:
            print(f"提前期 {lead_time} 周没有数据，跳过")
            continue

        hits = np.zeros(dataset.grid_size)
        false_alarms = np.zeros(dataset.grid_size)
        misses = np.zeros(dataset.grid_size)
        correct_negatives = np.zeros(dataset.grid_size)

        for t in range(len(ec_data)):
            ec_pred = ec_data[t]
            obs = obs_data[t]

            for i in range(dataset.grid_size[0]):
                for j in range(dataset.grid_size[1]):
                    if cn_valid_mask[i, j] and not np.isnan(obs[i, j]) and not np.isnan(ec_pred[i, j]):
                        pred_event = ec_pred[i, j] > thresholds[i, j]
                        obs_event = obs[i, j] > thresholds[i, j]

                        if pred_event and obs_event:
                            hits[i, j] += 1
                        elif pred_event and not obs_event:
                            false_alarms[i, j] += 1
                        elif not pred_event and obs_event:
                            misses[i, j] += 1
                        else:
                            correct_negatives[i, j] += 1

        # 修改：接收TS评分
        hss, pod, ts, far, pofd = calculate_metrics_per_grid(
            hits, false_alarms, misses, correct_negatives
        )

        valid_mask = cn_valid_mask & (hits + false_alarms + misses + correct_negatives > 0)

        ec_results[lead_time] = {
            'hss': np.nanmean(hss[valid_mask]) if np.any(valid_mask) else np.nan,
            'pod': np.nanmean(pod[valid_mask]) if np.any(valid_mask) else np.nan,
            'ts': np.nanmean(ts[valid_mask]) if np.any(valid_mask) else np.nan,  # 修改：添加TS评分
            'far': np.nanmean(far[valid_mask]) if np.any(valid_mask) else np.nan,
            'hss_grid': hss,
            'pod_grid': pod,
            'ts_grid': ts,  # 修改：添加TS网格数据
            'far_grid': far
        }

        print(f"提前期 {lead_time} 周 - ECMWF: HSS={ec_results[lead_time]['hss']:.3f}, TS={ec_results[lead_time]['ts']:.3f}")

    return ec_results