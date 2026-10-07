"""
配置文件
包含所有超参数和路径配置
"""

import random
import numpy as np
import torch
import os

# ==================== 基础路径与参数 ====================
# 数据路径
DATA_PATHS = {
    'gh_files': '/3240101047/EC_S2S_AI/ec_s2s.*.gh.nc',
    'pf_files': '/3240101047/EC_S2S_AI/ec_s2s.*.pf.nc',
    'pf13012_files': '/3240101047/EC_S2S_AI/ec_s2s.*.pf13012.nc',
    'tp_files': '/3240101047/EC_S2S_AI/*.tp.nc',
    'cn05_file': '/3240101047/EC_S2S_AI/CN05.1_Pre_1961_2022_daily_15x15_1.nc',
    'dem_file': '/3240101047/EC_S2S_AI/China_dem_reEC.nc',
    'tibet_shapefile': '/3240101047/EC_S2S_AI/cache/Admin Boundaries poly Tibet/TPBoundary_new(2021).shp',
}

LAT_RANGE = slice(40.5, 24)
LON_RANGE = slice(70.5, 105)

# 训练参数
BATCH_SIZE = 16
NUM_EPOCHS = 500
LEARNING_RATE = 1e-4
LOSS_B = 0.5
LOSS_K = 5

# 数据时间范围
TRAIN_YEARS = range(2001, 2013)
VAL_YEARS = range(2013, 2017)
TEST_YEARS = range(2017, 2021)

# 缓存设置
CACHE_DIR = '/3240101047/qiaoez/cache_300'

# 模型保存路径（unet_lead_{lead}_best.pth 所在目录）
MODEL_SAVE_DIR = '/3240101047/qiaoez'

# 随机种子
SEED = 42

# 设备设置
USE_CUDA = True

# 大气变量配置（200/500/850 hPa）
ATMOSPHERIC_VARIABLES = {
    '200hpa': ['gh', 'w', 'q', 't', 'u', 'v'],
    '500hpa': ['gh', 'w', 'q', 't', 'u', 'v'],
    '850hpa': ['gh', 'w', 'q', 't', 'u', 'v']
}

# ==================== 确定性训练配置 ====================
# 是否启用完全确定性模式（True = 结果可复现，速度略慢；False = 速度优先）
DETERMINISTIC_MODE = True

def set_deterministic(seed=SEED, deterministic=DETERMINISTIC_MODE):
    """
    设置随机种子和PyTorch后端为确定性模式，最大限度消除不同GPU节点间的运算差异。
    
    参数:
        seed: 随机种子
        deterministic: 是否启用严格确定性模式
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 固定 Python 哈希种子（影响字符串、字典等迭代顺序）
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if deterministic:
        # cuDNN 确定性设置
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # CUBLAS 工作空间配置（部分 CUDA 版本必需）
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # PyTorch 1.8+ 的全局确定性算法开关
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            # 较旧版本无此函数，忽略即可
            pass
        except RuntimeError as e:
            print(f"警告：启用 torch.use_deterministic_algorithms 失败，可能包含非确定性算子。\n{e}")
    else:
        # 非确定性模式下仍建议关闭 benchmark 以保证一定稳定性（可选）
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

# 自动执行确定性设置（导入 config 时即生效）
set_deterministic(SEED, DETERMINISTIC_MODE)

# ==================== 打印配置信息（可选） ====================
if __name__ == "__main__":
    print(f"随机种子: {SEED}")
    print(f"确定性模式: {'开启' if DETERMINISTIC_MODE else '关闭'}")
    print(f"cuDNN deterministic: {torch.backends.cudnn.deterministic}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
