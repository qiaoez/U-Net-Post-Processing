"""
模型可解释性分析模块：置换特征重要性（PFI）
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import os
import random
from scipy.interpolate import griddata

from config import *
from models import UNet
def calculate_pfi_for_model(model, dataset, test_years, batch_size=32, metric='hss', device=None):
    """计算U-Net模型的置换因子特征重要性"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() and USE_CUDA else 'cpu')
    
    model.eval()
    
    test_indices = [i for i, (_, _, _, _, year) in enumerate(dataset.data) 
                   if year in test_years]
    
    if not test_indices:
        print("警告: 测试集为空")
        return [0] * 20
    
    test_dataset = Subset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    thresholds = dataset.get_thresholds()
    cn_valid_mask = dataset.get_cn_valid_mask()
    
    print("计算基准性能...")
    baseline_performance = evaluate_model_performance(
        model, test_loader, thresholds, cn_valid_mask, metric, device
    )
    
    print(f"基准性能 ({metric}): {baseline_performance:.6f}")
    
    feature_importance = []
    
    for channel_idx in range(20):
        permuted_loader = create_permuted_loader(
            dataset, test_indices, channel_idx, batch_size
        )
        
        permuted_performance = evaluate_model_performance(
            model, permuted_loader, thresholds, cn_valid_mask, metric, device
        )
        
        if baseline_performance > 0:
            importance = (baseline_performance - permuted_performance) / baseline_performance
        else:
            importance = 0
        
        feature_importance.append(importance)
    
    return feature_importance

def evaluate_model_performance(model, data_loader, thresholds, valid_mask, metric='hss', device=None):
    """评估模型性能"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() and USE_CUDA else 'cpu')
    
    model.eval()
    
    if metric == 'mse':
        total_mse = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets, masks, _ in data_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                masks = masks.to(device)
                
                outputs = model(inputs)
                valid_pixels = (masks < 0.5).float()
                num_valid = valid_pixels.sum().item()
                
                if num_valid > 0:
                    mse = F.mse_loss(outputs * valid_pixels, targets.unsqueeze(1) * valid_pixels, reduction='sum')
                    total_mse += mse.item()
                    total_samples += num_valid
        
        if total_samples > 0:
            return total_mse / total_samples
        else:
            return 0.0
    
    elif metric == 'hss':
        hits = np.zeros(valid_mask.shape)
        false_alarms = np.zeros(valid_mask.shape)
        misses = np.zeros(valid_mask.shape)
        correct_negatives = np.zeros(valid_mask.shape)
        
        with torch.no_grad():
            for inputs, targets, masks, _ in data_loader:
                inputs = inputs.to(device)
                targets_np = targets.numpy()
                masks_np = masks.numpy()
                
                outputs = model(inputs)
                preds_np = outputs.cpu().numpy().squeeze()
                
                for b in range(preds_np.shape[0]):
                    pred_map = preds_np[b] if preds_np.ndim > 2 else preds_np
                    target_map = targets_np[b]
                    mask_map = masks_np[b]
                    
                    for i in range(valid_mask.shape[0]):
                        for j in range(valid_mask.shape[1]):
                            if (valid_mask[i, j] and mask_map[i, j] < 0.5 and 
                                not np.isnan(target_map[i, j]) and not np.isnan(pred_map[i, j])):
                                
                                pred_event = pred_map[i, j] > thresholds[i, j]
                                obs_event = target_map[i, j] > thresholds[i, j]
                                
                                if pred_event and obs_event:
                                    hits[i, j] += 1
                                elif pred_event and not obs_event:
                                    false_alarms[i, j] += 1
                                elif not pred_event and obs_event:
                                    misses[i, j] += 1
                                else:
                                    correct_negatives[i, j] += 1
        
        # 导入calculate_metrics_per_grid函数
        from evaluate import calculate_metrics_per_grid
        hss, _, _, _, _ = calculate_metrics_per_grid(hits, false_alarms, misses, correct_negatives)
        
        valid_hss = hss[valid_mask & (hits + false_alarms + misses + correct_negatives > 0)]
        if len(valid_hss) > 0:
            return np.nanmean(valid_hss)
        else:
            return 0.0
    
    else:
        raise ValueError(f"不支持的评估指标: {metric}")
'''
class PermutedDataset(Dataset):
    """置换数据集"""
    def __init__(self, base_dataset, base_indices, channel_idx):
        self.base_dataset = base_dataset
        self.base_indices = base_indices
        self.channel_idx = channel_idx
    
    def __len__(self):
        return len(self.base_indices)
    
    def __getitem__(self, idx):
        original_idx = self.base_indices[idx]
        inputs, target, mask, lead_time = self.base_dataset[original_idx]
        
        permuted_inputs = inputs.clone()
        permuted_inputs[self.channel_idx] = 0
        
        return permuted_inputs, target, mask, lead_time
'''
def create_permuted_loader(dataset, indices, channel_idx, batch_size=32):
    """
    创建置换数据加载器，对指定通道的特征值进行随机打乱（沿batch维度）。
    参数：
        dataset: 原始数据集
        indices: 测试样本索引列表
        channel_idx: 要置换的通道索引
        batch_size: 批大小
    返回：
        DataLoader，每次迭代返回打乱指定通道后的批次
    """
    base_dataset = Subset(dataset, indices)

    def collate_fn(batch):
        # batch: list of (inputs, target, mask, lead_time)
        inputs_list = []
        targets_list = []
        masks_list = []
        lead_times_list = []

        for item in batch:
            inputs, target, mask, lead_time = item
            inputs_list.append(inputs.unsqueeze(0))      # [1, C, H, W]
            targets_list.append(target.unsqueeze(0))     # [1, H, W]
            masks_list.append(mask.unsqueeze(0))         # [1, H, W]
            lead_times_list.append(lead_time)

        # 拼接为批次
        inputs = torch.cat(inputs_list, dim=0)           # [B, C, H, W]
        targets = torch.cat(targets_list, dim=0)         # [B, H, W]
        masks = torch.cat(masks_list, dim=0)             # [B, H, W]
        lead_times = torch.tensor(lead_times_list)       # [B]

        # 对指定通道进行随机置换（沿batch维度打乱）
        if inputs.size(1) > channel_idx:
            # 提取该通道数据 [B, H, W]
            channel_data = inputs[:, channel_idx, :, :].clone()
            perm_indices = torch.randperm(channel_data.size(0))  # 随机排列索引
            shuffled_channel = channel_data[perm_indices]        # 打乱batch顺序
            inputs[:, channel_idx, :, :] = shuffled_channel      # 替换原通道
        else:
            print(f"警告: 通道索引 {channel_idx} 超出范围")

        return inputs, targets, masks, lead_times

    loader = DataLoader(
        base_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    return loader








