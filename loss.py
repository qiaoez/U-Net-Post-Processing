"""
损失函数模块
包含WeightedLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedLoss(nn.Module):
    def __init__(self, b=1, k=2):
        super(WeightedLoss, self).__init__()
        self.b = b
        self.k = k
        self.mse = nn.MSELoss(reduction='none')
        self.eps = 1e-6
        self.step = 0

    def forward(self, pred, target, threshold):
        self.step += 1

        if target.dim() == 3:
            target = target.unsqueeze(1)

        mask = ~torch.isnan(target)
        valid_pixels = mask.sum()

        if valid_pixels == 0:
            return torch.tensor(0.0, requires_grad=True).to(pred.device)

        target_valid = torch.where(mask, target, torch.zeros_like(target))
        target_mean = target_valid.sum() / (valid_pixels + self.eps)
        target_variance = torch.sum((target_valid - target_mean) ** 2) / (valid_pixels + self.eps)

        pred_valid = torch.where(mask, pred, torch.zeros_like(pred))
        pred_mean = pred_valid.sum() / (valid_pixels + self.eps)
        pred_variance = torch.sum((pred_valid - pred_mean) ** 2) / (valid_pixels + self.eps)

        if target_variance < self.eps:
            target_variance = torch.tensor(1.0).to(pred.device)
        if pred_variance < self.eps:
            pred_variance = torch.tensor(1.0).to(pred.device)

        mse_loss = self.mse(pred, target)
        mse_loss = torch.where(mask, mse_loss, torch.zeros_like(mse_loss))
        mse_loss_sum = mse_loss.sum()

        normalized_mse = mse_loss_sum / (pred_variance * valid_pixels + self.eps)

        obs_events = torch.sigmoid(self.k * (target - threshold))
        pred_events = torch.sigmoid(self.k * (pred - threshold))

        obs_events = torch.where(mask, obs_events, torch.zeros_like(obs_events))
        pred_events = torch.where(mask, pred_events, torch.zeros_like(pred_events))

        hits = torch.sum(pred_events * obs_events)
        false_alarms = torch.sum(pred_events * (1 - obs_events))
        misses = torch.sum((1 - pred_events) * obs_events)

        denominator = hits + false_alarms + misses
        if denominator > 0:
            ts = hits / denominator
        else:
            ts = torch.tensor(0.0).to(pred.device)

        one_minus_ts = 1 - ts

        total_loss = self.b * one_minus_ts + normalized_mse

        if self.training and self.step % 500 == 0:
            norm_mse_val = normalized_mse.detach().item()
            one_minus_ts_val = one_minus_ts.detach().item()
            ratio = norm_mse_val / (one_minus_ts_val + self.eps)

            print(f"Step {self.step}: norm_MSE={norm_mse_val:.3f}, 1-TS={one_minus_ts_val:.3f}, ratio={ratio:.3f}")

        return total_loss