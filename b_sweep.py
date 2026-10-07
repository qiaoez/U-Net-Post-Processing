"""
b 超参数扫描实验：k 固定为 2，b 在 (0, 2] 区间以 0.1 为步长扫描

功能：
1. 对提前期 1-4 周分别训练 U-Net，扫描 b = 0.1, 0.2, ..., 2.0（k 固定为 2），
   其余超参与论文一致（2001-2012 训练、2013-2016 验证、2017-2020 测试、
   batch 16、500 epoch、lr 1e-4）；
2. 提前期 1-2 的 b=0.5、提前期 3-4 的 b=1.3 直接使用论文已有缓存权重
   （/3240101047/qiaoez/unet_lead_{lead}_best.pth）评估，不重新训练；
3. 汇总每个 (b, 提前期) 的 HSS / TS / POD / FAR / TCC / RMSE，
   结果写入 SWEEP_RESULT_PATH。

注意：
- train.py 的训练函数会把模型保存到固定的 /3240101047/qiaoez/unet_lead_{lead}_best.pth，
  因此本流程开跑前备份论文权重、结束时恢复；
- 每个 (b, 提前期) 的结果单独缓存，支持任务中断后续训。
"""
import sys
import os
import shutil
import pickle
import subprocess

# 将当前文件所在目录添加到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import torch
import numpy as np

from torch.utils.data import DataLoader, Subset

from data_processing import WeeklyEC_S2S_Dataset
from train import train_unet_model, analyze_training_progress
from evaluate import evaluate_model, evaluate_ec_period, evaluate_unet_0115
from metrics import calculate_method_tcc_rmse_grid
from models import UNet
from config import *

print(f"Hostname: {os.uname().nodename}")
print(f"GPU Model: {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.version.cuda}, cuDNN: {torch.backends.cudnn.version()}")
print(f"Driver: {subprocess.getoutput('nvidia-smi --query-gpu=driver_version --format=csv,noheader')}")

# ==================== b 扫描实验配置 ====================
LEADS = [1, 2, 3, 4]           # 提前期（周）
SWEEP_K = 2                    # 扫描时固定 k = 2
# b 取值 0.1, 0.2, ..., 2.0（用整数步长生成再除以 10，避免浮点误差）
B_VALUES = [round(i / 10.0, 1) for i in range(1, 21)]

# 论文已有缓存权重所在的格子：提前期 1-2 为 b=0.5，提前期 3-4 为 b=1.3
CACHE_CELLS = {(1, 0.5), (2, 0.5), (3, 1.3), (4, 1.3)}

# 论文权重保存目录（与 train.py / config.py 保持一致）
PAPER_MODEL_DIR = MODEL_SAVE_DIR

# 扫描实验输出目录（不改动论文已用的 plot/ 与根目录产物）
SWEEP_DIR = '/3240101047/qiaoez/b_sweep'
SWEEP_MODEL_DIR = os.path.join(SWEEP_DIR, 'models')
SWEEP_CACHE_DIR = os.path.join(SWEEP_DIR, 'cache')
SWEEP_PAPER_WEIGHT_DIR = os.path.join(SWEEP_DIR, 'paper_weights')
SWEEP_RESULT_PATH = os.path.join(SWEEP_DIR, 'sweep_results.pkl')
EC_BASELINE_PATH = os.path.join(SWEEP_DIR, 'ec_baseline.pkl')

# 评估时段：验证集 2013-2016（模型选择期，图注会标明属内部验证而非独立评估）
EVAL_YEARS = VAL_YEARS
EVAL_YEARS_TAG = '2013-2016'
PERIOD_LABEL = 'Validation 2013-2016 (model-selection period)'

# 不画 FAR 柱状（FAR 数值偏高，会盖住 HSS 的变化趋势），
# 改用“空心圆点”标记该 b 的 FAR 高于 ECMWF 基准（即未达标）
FAR_ABOVE_BASELINE_FACE = 'white'
HSS_LINE_COLOR = '#2ca02c'
EC_BASELINE_COLOR = '#1f77b4'
RMSE_BAR_COLOR = '#ff7f0e'

# 自检开关：True 时只跑 1 个 b、2 个 epoch，用于验证 IO/绘图链路
SWEEP_SMOKE_TEST = False
SMOKE_TEST_EPOCHS = 2


def save_results(results_dict, filename):
    """保存结果到文件"""
    dir_name = os.path.dirname(filename)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(results_dict, f)
    print(f"结果已保存到 {filename}")


def load_results(filename):
    """从文件加载结果"""
    with open(filename, 'rb') as f:
        return pickle.load(f)


# ==================== 路径辅助函数 ====================
def _b_tag(b):
    """b 的字符串标记（保留 1 位小数）"""
    return f"{b:.1f}"


def paper_model_path(lead_time):
    """论文权重路径（train.py 中写死的保存路径）"""
    return os.path.join(PAPER_MODEL_DIR, f'unet_lead_{lead_time}_best.pth')


def cell_cache_path(lead_time, b):
    """单个 (b, 提前期) 的结果缓存路径"""
    return os.path.join(SWEEP_CACHE_DIR, f'result_lead{lead_time}_b{_b_tag(b)}.pkl')


def sweep_model_path(lead_time, b):
    """扫描训练得到的模型另存路径"""
    return os.path.join(SWEEP_MODEL_DIR, f'unet_lead{lead_time}_b{_b_tag(b)}.pth')




def get_device():
    """获取计算设备"""
    return torch.device('cuda' if torch.cuda.is_available() and USE_CUDA else 'cpu')


# ==================== 论文权重备份 / 恢复 ====================
def ensure_paper_weight_backup():
    """
    把论文权重复制到 b_sweep/paper_weights/。

    只在备份文件不存在时才复制：这样即使上一次任务被强杀（finally 未执行、
    原始文件已被覆盖），再次运行时也不会用被污染的文件覆盖干净备份。

    返回 {lead: 备份路径}
    """
    os.makedirs(SWEEP_PAPER_WEIGHT_DIR, exist_ok=True)
    backups = {}
    for lead in LEADS:
        src = paper_model_path(lead)
        dst = os.path.join(SWEEP_PAPER_WEIGHT_DIR, f'unet_lead_{lead}_best.pth')
        if not os.path.exists(src):
            print(f"警告：论文权重不存在，无法备份: {src}")
            continue
        if os.path.exists(dst):
            print(f"论文权重备份已存在，直接复用: {dst}")
        else:
            shutil.copy2(src, dst)
            print(f"已备份论文权重: {src} -> {dst}")
        backups[lead] = dst
    return backups


def restore_paper_weights(backups):
    """把备份的论文权重恢复到原始路径"""
    for lead, backup_path in backups.items():
        dst = paper_model_path(lead)
        try:
            shutil.copy2(backup_path, dst)
            print(f"已恢复论文权重: {dst}")
        except Exception as e:
            print(f"警告：恢复论文权重失败 {dst}: {e}")


# ==================== 指标计算 ====================
def predict_unet_forecasts(model, dataset, lead_time, test_years=EVAL_YEARS,
                           device=None, batch_size=BATCH_SIZE):
    """
    对指定提前期的测试样本做 U-Net 推理，返回形状 (N, H, W) 的预报数组。

    样本顺序与 dataset.get_training_data_by_lead(lead_time, test_years) 完全一致，
    保证与观测序列配对正确（TCC/RMSE 计算依赖这一点）。
    """
    if device is None:
        device = get_device()

    test_indices = [i for i, (_, _, _, lt, year) in enumerate(dataset.data)
                    if lt == lead_time and year in test_years]
    if len(test_indices) == 0:
        return np.empty((0,))

    model.eval()
    test_dataset = Subset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_predictions = []
    with torch.no_grad():
        for inputs, _, _, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_predictions.append(outputs.cpu().numpy())

    if not all_predictions:
        return np.empty((0,))
    return np.concatenate(all_predictions, axis=0).squeeze()


def evaluate_cell(model, dataset, lead_time, device):
    """
    评估单个 (b, 提前期) 模型的全部指标。

    - hss/pod/ts/far：复用 evaluate_model（与论文口径一致），只取该提前期的结果；
    - tcc/rmse：复用 tcc_rmse.calculate_method_tcc_rmse_grid。

    评估时段为 EVAL_YEARS（验证集 2013-2016）。
    """
    results = evaluate_model(model, dataset, test_years=EVAL_YEARS)
    lead_res = results.get(lead_time, {})

    metrics = {
        'hss': float(lead_res.get('hss', np.nan)),
        'pod': float(lead_res.get('pod', np.nan)),
        'ts': float(lead_res.get('ts', np.nan)),
        'far': float(lead_res.get('far', np.nan)),
    }

    forecasts = predict_unet_forecasts(model, dataset, lead_time, EVAL_YEARS, device)
    _, obs_test = dataset.get_training_data_by_lead(lead_time, EVAL_YEARS)
    cn_valid_mask = dataset.get_cn_valid_mask()

    if len(forecasts) == 0 or len(obs_test) == 0:
        metrics['tcc'] = np.nan
        metrics['rmse'] = np.nan
    else:
        tcc, rmse, _, _ = calculate_method_tcc_rmse_grid(
            forecasts, obs_test, cn_valid_mask, f'U-Net(lead{lead_time})'
        )
        metrics['tcc'] = float(tcc)
        metrics['rmse'] = float(rmse)

    return metrics


def compute_ec_baseline(dataset):
    """
    计算 ECMWF 基准（一次算好后缓存到 b_sweep/ec_baseline.pkl）：
    - far：取自 evaluate_ec_period 的 EC 结果（评估时段 EVAL_YEARS）；
    - rmse：取自 calculate_method_tcc_rmse_grid 的 EC 结果（同口径）。

    缓存里带 eval_years 标记，评估时段变化时自动重算。
    """
    if os.path.exists(EC_BASELINE_PATH):
        cached = load_results(EC_BASELINE_PATH)
        if isinstance(cached, dict) and cached.get('eval_years') == EVAL_YEARS_TAG:
            print(f"从缓存加载 ECMWF 基准: {EC_BASELINE_PATH}")
            return cached['values']
        print(f"ECMWF 基准缓存的评估时段不是 {EVAL_YEARS_TAG}，重新计算...")

    print(f"\n计算 ECMWF 基准（FAR / RMSE，评估时段 {EVAL_YEARS_TAG}）...")
    ec_results = evaluate_ec_period(dataset, EVAL_YEARS)
    cn_valid_mask = dataset.get_cn_valid_mask()

    baseline = {}
    for lead in LEADS:
        far_val = float(ec_results.get(lead, {}).get('far', np.nan))

        ec_test, obs_test = dataset.get_training_data_by_lead(lead, EVAL_YEARS)
        if len(ec_test) == 0:
            rmse_val = np.nan
        else:
            _, rmse_val, _, _ = calculate_method_tcc_rmse_grid(
                ec_test, obs_test, cn_valid_mask, f'ECMWF(lead{lead})'
            )
            rmse_val = float(rmse_val)

        baseline[lead] = {'far': far_val, 'rmse': rmse_val}
        print(f"ECMWF 基准 提前期{lead}周: FAR={far_val:.3f}, RMSE={rmse_val:.3f}")

    save_results({'eval_years': EVAL_YEARS_TAG, 'values': baseline}, EC_BASELINE_PATH)
    return baseline


# ==================== 单个格子 ====================
def run_single_cell(dataset, b, lead_time, device):
    """
    跑单个 (b, 提前期) 格子：
    - 若该格子属于论文缓存格子且权重副本存在，则直接加载权重评估（不训练）；
    - 否则正常训练，并把模型另存到 b_sweep/models/。
    """
    use_cache = (lead_time, b) in CACHE_CELLS
    model = None
    num_epochs = None

    if use_cache:
        weight_path = os.path.join(SWEEP_PAPER_WEIGHT_DIR, f'unet_lead_{lead_time}_best.pth')
        if os.path.exists(weight_path):
            print(f"[缓存] b={b}, 提前期={lead_time} 周：加载论文缓存权重评估（不训练）")
            model = UNet(in_channels=20, out_channels=1).to(device)
            model.load_state_dict(torch.load(weight_path, map_location=device))
            model.eval()
        else:
            print(f"警告：论文缓存权重不存在（{weight_path}），回退为正常训练 b={b}, 提前期={lead_time} 周")
            use_cache = False

    if not use_cache:
        print(f"[训练] b={b}, 提前期={lead_time} 周：开始训练（k={SWEEP_K}）")
        num_epochs = SMOKE_TEST_EPOCHS if SWEEP_SMOKE_TEST else NUM_EPOCHS
        model, train_losses, val_losses = train_unet_model(
            dataset, lead_time,
            train_years=TRAIN_YEARS,
            val_years=VAL_YEARS,
            batch_size=BATCH_SIZE,
            num_epochs=num_epochs,
            learning_rate=LEARNING_RATE,
            b=b,
            k=SWEEP_K
        )
        torch.save(model.state_dict(), sweep_model_path(lead_time, b))
        print(f"扫描模型已保存到: {sweep_model_path(lead_time, b)}")
        analyze_training_progress(train_losses, val_losses)

    model.eval()
    metrics = evaluate_cell(model, dataset, lead_time, device)
    metrics['b'] = float(b)
    metrics['lead'] = int(lead_time)
    metrics['source'] = 'paper_cache' if use_cache else 'trained'
    # 记录训练配置，用于判断缓存条目能否在当前配置下复用
    metrics['num_epochs'] = num_epochs
    metrics['smoke_test'] = bool(SWEEP_SMOKE_TEST) if not use_cache else False
    # 记录评估时段，评估时段变化时旧条目不能复用
    metrics['eval_years'] = EVAL_YEARS_TAG
    metrics['period_label'] = PERIOD_LABEL

    print(f"b={b}, 提前期={lead_time} 周 -> HSS={metrics['hss']:.3f}, POD={metrics['pod']:.3f}, "
          f"TS={metrics['ts']:.3f}, FAR={metrics['far']:.3f}, "
          f"TCC={metrics['tcc']:.3f}, RMSE={metrics['rmse']:.3f} [{metrics['source']}]")
    return metrics


def expected_epochs():
    """当前配置下每个格子应当训练的 epoch 数"""
    return SMOKE_TEST_EPOCHS if SWEEP_SMOKE_TEST else NUM_EPOCHS


def _training_config_matches(entry, lead_time, b):
    """
    缓存条目记录的训练配置是否与当前配置一致。

    一致（且权重文件存在）时只需按新评估时段重新评估，不必重新训练。
    """
    if (lead_time, b) in CACHE_CELLS:
        # 缓存格子必须是“加载论文权重重算”的结果，训练出来的不算
        return entry.get('source') == 'paper_cache'
    if entry.get('source') == 'paper_cache':
        return False
    return (entry.get('num_epochs') == expected_epochs()
            and bool(entry.get('smoke_test', False)) == bool(SWEEP_SMOKE_TEST))


def cache_entry_is_reusable(entry, lead_time, b):
    """
    判断已有缓存条目能否在当前配置下直接复用（训练配置与评估时段都要一致）。

    避免 SWEEP_SMOKE_TEST=True 时只训练 2 个 epoch 的结果被正式运行当成已完成格子跳过；
    也避免旧的评估时段（如 2017-2020）结果被当成新时段（2013-2016）的结果复用。
    """
    if entry.get('eval_years') != EVAL_YEARS_TAG:
        return False
    return _training_config_matches(entry, lead_time, b)


def evaluate_from_saved_model(dataset, b, lead_time, device):
    """
    只重新评估、不训练：用已保存的权重按当前评估时段重算指标。

    权重取法：缓存格子取 b_sweep/paper_weights/ 下的论文权重，
    其余格子取 b_sweep/models/ 下该 (lead, b) 训练得到的权重。
    找不到权重时返回 None，由调用方回退为正常训练。
    """
    weight_path = None
    source = None

    if (lead_time, b) in CACHE_CELLS:
        candidate = os.path.join(SWEEP_PAPER_WEIGHT_DIR, f'unet_lead_{lead_time}_best.pth')
        if os.path.exists(candidate):
            weight_path, source = candidate, 'paper_cache'
    if weight_path is None:
        candidate = sweep_model_path(lead_time, b)
        if os.path.exists(candidate):
            weight_path, source = candidate, 'trained'

    if weight_path is None:
        return None

    print(f"[重算] b={b}, 提前期={lead_time} 周：加载已保存权重重新评估（不训练）: {weight_path}")
    model = UNet(in_channels=20, out_channels=1).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    metrics = evaluate_cell(model, dataset, lead_time, device)
    metrics['b'] = float(b)
    metrics['lead'] = int(lead_time)
    metrics['source'] = source
    metrics['num_epochs'] = None if source == 'paper_cache' else expected_epochs()
    metrics['smoke_test'] = False if source == 'paper_cache' else bool(SWEEP_SMOKE_TEST)
    metrics['eval_years'] = EVAL_YEARS_TAG
    metrics['period_label'] = PERIOD_LABEL

    print(f"b={b}, 提前期={lead_time} 周 -> HSS={metrics['hss']:.3f}, FAR={metrics['far']:.3f}, "
          f"RMSE={metrics['rmse']:.3f} [{metrics['source']}, 重算]")
    return metrics


def run_b_sweep(dataset, device):
    """
    遍历所有 (b, 提前期) 格子。已存在结果缓存的格子直接跳过（断点续训）。
    返回 (results, ec_baseline)，其中 results[(b, lead)] = 指标字典。
    """
    b_values = B_VALUES[:1] if SWEEP_SMOKE_TEST else B_VALUES
    ec_baseline = compute_ec_baseline(dataset)

    results = {}
    total = len(b_values) * len(LEADS)
    counter = 0

    for b in b_values:
        for lead in LEADS:
            counter += 1
            cache_path = cell_cache_path(lead, b)
            cached_entry = load_results(cache_path) if os.path.exists(cache_path) else None

            if cached_entry is not None and cache_entry_is_reusable(cached_entry, lead, b):
                print(f"[{counter}/{total}] 跳过已完成格子（读取缓存）: b={b}, 提前期={lead} 周")
                entry = cached_entry
            else:
                entry = None

                # 训练配置一致、只是评估时段变了 -> 用已保存权重重新评估，不重新训练
                if cached_entry is not None and _training_config_matches(cached_entry, lead, b):
                    print(f"[{counter}/{total}] 仅评估时段变化（{cached_entry.get('eval_years')} -> "
                          f"{EVAL_YEARS_TAG}），重新评估不训练: b={b}, 提前期={lead} 周")
                    entry = evaluate_from_saved_model(dataset, b, lead, device)
                    if entry is None:
                        print("          未找到已保存权重，回退为正常训练")

                if entry is None:
                    if cached_entry is not None:
                        print(f"[{counter}/{total}] 缓存与当前训练配置不一致（epochs/自检模式），"
                              f"重新训练: b={b}, 提前期={lead} 周")
                    else:
                        print(f"\n[{counter}/{total}] 处理格子: b={b}, 提前期={lead} 周")
                    entry = run_single_cell(dataset, b, lead, device)
                save_results(entry, cache_path)
            results[(b, lead)] = entry

    save_results(
        {
            'k': SWEEP_K,
            'b_values': list(b_values),
            'leads': list(LEADS),
            'cache_cells': sorted(CACHE_CELLS),
            'smoke_test': bool(SWEEP_SMOKE_TEST),
            'eval_years': EVAL_YEARS_TAG,
            'period_label': PERIOD_LABEL,
            'results': results,
            'ec_baseline': ec_baseline,
        },
        SWEEP_RESULT_PATH
    )
    return results, ec_baseline


# ==================== 绘图 ====================
def _metric_series(results, b_values, lead_time, key):
    """取出某个提前期、某个指标随 b 变化的序列"""
    return [results.get((b, lead_time), {}).get(key, np.nan) for b in b_values]








def print_sweep_summary(results, ec_baseline):
    """打印扫描结果的汇总表（评估时段为验证集 2013-2016）"""
    b_values = sorted({b for (b, _lead) in results.keys()})
    print("\n" + "=" * 96)
    print(f"b 扫描结果汇总（k={SWEEP_K}, 评估时段 {EVAL_YEARS_TAG}: {PERIOD_LABEL}）")
    print("=" * 96)
    header = (f"{'Lead':<6}{'b':<7}{'HSS':<9}{'POD':<9}{'TS':<9}{'FAR':<9}"
              f"{'TCC':<9}{'RMSE':<9}{'source':<12}")
    for lead in LEADS:
        base = ec_baseline.get(lead, {})
        print("-" * 96)
        print(f"ECMWF 基准（提前期{lead}周）: FAR={base.get('far', np.nan):.3f}, "
              f"RMSE={base.get('rmse', np.nan):.3f}")
        print(header)
        for b in b_values:
            entry = results.get((b, lead))
            if entry is None:
                continue
            print(f"{lead:<6}{b:<7.1f}{entry.get('hss', np.nan):<9.3f}"
                  f"{entry.get('pod', np.nan):<9.3f}{entry.get('ts', np.nan):<9.3f}"
                  f"{entry.get('far', np.nan):<9.3f}{entry.get('tcc', np.nan):<9.3f}"
                  f"{entry.get('rmse', np.nan):<9.3f}{entry.get('source', '-'):<12}")
    print("=" * 96)


# ==================== 主流程 ====================
def main_b_sweep():
    """b 超参数扫描主流程（k 固定为 2）"""
    print("=" * 60)
    print("开始 b 超参数扫描实验（k = %d）" % SWEEP_K)
    print(f"b 取值: {B_VALUES}")
    print(f"评估时段: {EVAL_YEARS_TAG}（{PERIOD_LABEL}）")
    print(f"缓存格子（不重训）: {sorted(CACHE_CELLS)}")
    print(f"输出目录: {SWEEP_DIR}")
    print("=" * 60)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    for directory in (SWEEP_DIR, SWEEP_MODEL_DIR, SWEEP_CACHE_DIR, SWEEP_PAPER_WEIGHT_DIR):
        os.makedirs(directory, exist_ok=True)

    # 备份论文权重（只备份一次，避免被覆盖后的文件污染备份）
    backups = ensure_paper_weight_backup()

    results = {}
    try:
        device = get_device()
        print(f"使用设备: {device}")

        print("\n加载数据集...")
        dataset = WeeklyEC_S2S_Dataset(start_year=2001, end_year=2020)
        stats = dataset.get_data_statistics()
        print(f"输入数据形状: {stats['input_shape']}")
        print(f"目标数据形状: {stats['target_shape']}")

        results, ec_baseline = run_b_sweep(dataset, device)

        print_sweep_summary(results, ec_baseline)
        print(f"\n扫描汇总结果: {SWEEP_RESULT_PATH}")
    finally:
        # 无论是否异常，都恢复论文权重
        restore_paper_weights(backups)

    print("\nb 扫描实验完成！")
    print("=" * 60)
    return results




if __name__ == "__main__":
    results = main_b_sweep()
