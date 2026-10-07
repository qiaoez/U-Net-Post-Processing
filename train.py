"""
训练模块
包含模型训练函数
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import os

from models import UNet
from loss import WeightedLoss
from config import *

import os
import sys

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
def train_unet_model(dataset, lead_time, train_years=TRAIN_YEARS, val_years=VAL_YEARS,
                     batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS,
                     learning_rate=LEARNING_RATE, b=LOSS_B, k=LOSS_K):
    """训练U-Net模型"""
    indices = [i for i, (_, _, _, lt, year) in enumerate(dataset.data)
               if lt == lead_time and year in list(train_years) + list(val_years)]
    subset = Subset(dataset, indices)

    train_indices = [i for i, idx in enumerate(indices) if dataset.years[idx] in train_years]
    val_indices = [i for i, idx in enumerate(indices) if dataset.years[idx] in val_years]

    train_dataset = Subset(subset, train_indices)
    val_dataset = Subset(subset, val_indices)

    device = torch.device('cuda' if torch.cuda.is_available() and USE_CUDA else 'cpu')

    print(f"提前期 {lead_time} 周：进行U-Net模型训练")
    print(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = UNet(in_channels=20, out_channels=1).to(device)
    criterion = WeightedLoss(b=b, k=k)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.6, patience=8, verbose=True
    )

    thresholds = torch.tensor(dataset.get_thresholds(), dtype=torch.float32).to(device)

    train_losses = []
    val_losses = []

    patience = 20
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0

        if epoch % 10 == 0:
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Train")
        else:
            train_iterator = train_loader

        for inputs, targets, masks, _ in train_iterator:
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets.unsqueeze(1), thresholds)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, masks, _ in val_loader:
                inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1), thresholds)

                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            weight_path = os.path.join(MODEL_SAVE_DIR, f'unet_lead_{lead_time}_best.pth')
            torch.save(model.state_dict(), weight_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发，在第 {epoch + 1} 个epoch停止训练")
                break

    # 加载最佳模型
    model.load_state_dict(torch.load(
        os.path.join(MODEL_SAVE_DIR, f'unet_lead_{lead_time}_best.pth')))

    return model, train_losses, val_losses


def analyze_training_progress(train_losses, val_losses):
    """分析训练进度并提供建议"""
    print("\n" + "=" * 50)
    print("训练效果分析")
    print("=" * 50)

    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    best_val_loss = min(val_losses)
    best_epoch = val_losses.index(best_val_loss) + 1

    overfitting_ratio = final_val_loss / final_train_loss if final_train_loss > 0 else float('inf')

    print(f"关键指标:")
    print(f"  - 最终训练损失: {final_train_loss:.6f}")
    print(f"  - 最终验证损失: {final_val_loss:.6f}")
    print(f"  - 最佳验证损失: {best_val_loss:.6f} (第 {best_epoch} 轮)")
    print(f"  - 过拟合比率: {overfitting_ratio:.2f}")

    if overfitting_ratio > 1.5:
        print("明显过拟合: 验证损失显著高于训练损失")
        print("   建议: 增加Dropout率、添加正则化、减少模型复杂度")
    elif overfitting_ratio > 1.2:
        print(" 轻微过拟合: 验证损失略高于训练损失")
        print("   建议: 适当增加Dropout或数据增强")
    else:
        print("拟合良好: 训练和验证损失接近")

    last_10_train = train_losses[-10:]
    last_10_val = val_losses[-10:]

    train_convergence = (max(last_10_train) - min(last_10_train)) / np.mean(last_10_train)
    val_convergence = (max(last_10_val) - min(last_10_val)) / np.mean(last_10_val)

    if train_convergence < 0.05 and val_convergence < 0.05:
        print("模型已收敛: 最后10轮损失变化很小")
    elif train_convergence < 0.1 and val_convergence < 0.1:
        print("基本收敛: 损失仍在缓慢下降")
    else:
        print("未完全收敛: 考虑增加训练轮数")

    train_std = np.std(train_losses[-20:]) if len(train_losses) >= 20 else np.std(train_losses)
    val_std = np.std(val_losses[-20:]) if len(val_losses) >= 20 else np.std(val_losses)

    print(f"  - 训练损失稳定性: {train_std:.6f}")
    print(f"  - 验证损失稳定性: {val_std:.6f}")

    return {
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'overfitting_ratio': overfitting_ratio,
        'converged': train_convergence < 0.1 and val_convergence < 0.1
    }
