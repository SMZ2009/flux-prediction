# -*- coding: utf-8 -*-
"""
核心升级：支持指定事件+指定相位（上升/峰值/下降）的局部预测 + 相位边界平滑过渡
输入: DE1通量 → 输出: DE4/DE1比值 → 反推DE4通量
特点：
- 事件拆分为上升段/峰值段/下降段，可单独预测某一段
- 手动指定「目标事件ID + 目标相位」(如只预测事件42的峰值段)
- 时间轴优化：仅展示目标相位的时序曲线
- 保留原有的归一化、MLP模型、评估指标
- 新增：相位边界重叠区 + 联合预测模型 + 边界平滑处理
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from datetime import timedelta
from sklearn.metrics import r2_score
import torch.nn as nn
from scipy.interpolate import UnivariateSpline
from scipy.fft import fft, fftfreq

# ======================== 新增：傅里叶变换特征提取 ========================
def extract_fourier_features(time_series, de1_flux, n_harmonics=5):
    """
    提取时间序列的傅里叶特征
    :param time_series: 时间序列
    :param de1_flux: DE1通量数据
    :param n_harmonics: 谐波数量
    :return: 傅里叶特征向量
    """
    # 确保输入是numpy数组
    time_series = np.array(time_series)
    de1_flux = np.array(de1_flux)
    
    # 计算FFT
    n = len(de1_flux)
    if n < 2:
        return np.zeros(3 * n_harmonics)
    
    # 计算频率
    freq = fftfreq(n)
    # 计算FFT
    fft_vals = fft(de1_flux)
    # 取幅度
    fft_amps = np.abs(fft_vals) / n
    
    # 只取正频率
    positive_freqs = freq[freq > 0]
    positive_amps = fft_amps[freq > 0]
    
    # 排序并取前n_harmonics个频率分量
    if len(positive_amps) > 0:
        sorted_indices = np.argsort(positive_amps)[::-1]
        top_indices = sorted_indices[:n_harmonics]
        top_freqs = positive_freqs[top_indices]
        top_amps = positive_amps[top_indices]
        
        # 计算相位
        top_phases = np.angle(fft_vals[freq > 0][top_indices])
    else:
        top_freqs = np.zeros(n_harmonics)
        top_amps = np.zeros(n_harmonics)
        top_phases = np.zeros(n_harmonics)
    
    # 填充到固定长度
    freqs_padded = np.pad(top_freqs, (0, n_harmonics - len(top_freqs)), 'constant')
    amps_padded = np.pad(top_amps, (0, n_harmonics - len(top_amps)), 'constant')
    phases_padded = np.pad(top_phases, (0, n_harmonics - len(top_phases)), 'constant')
    
    # 组合特征
    features = np.concatenate([freqs_padded, amps_padded, phases_padded])
    return features

def extract_windowed_fourier_features(de1_flux, window_size=20, n_harmonics=5):
    """
    提取滑动窗口的傅里叶特征
    :param de1_flux: DE1通量数据
    :param window_size: 窗口大小
    :param n_harmonics: 谐波数量
    :return: 每个时间点的傅里叶特征
    """
    n = len(de1_flux)
    features = []
    
    for i in range(n):
        # 计算窗口范围
        start = max(0, i - window_size + 1)
        end = i + 1
        window = de1_flux[start:end]
        time_series = np.arange(len(window))
        
        # 提取傅里叶特征
        fourier_feat = extract_fourier_features(time_series, window, n_harmonics)
        features.append(fourier_feat)
    
    return np.array(features)

# ======================== 新增：注意力机制 ========================
class SelfAttention(nn.Module):
    """
    自注意力机制
    """
    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim if hidden_dim is not None else input_dim
        # 定义查询、键、值的线性变换
        self.query = nn.Linear(input_dim, self.hidden_dim)
        self.key = nn.Linear(input_dim, self.hidden_dim)
        self.value = nn.Linear(input_dim, self.hidden_dim)
        # 输出线性变换
        self.output = nn.Linear(self.hidden_dim, input_dim)
        # 缩放因子
        self.scale = torch.tensor(self.hidden_dim ** 0.5, dtype=torch.float32)
    
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        q = self.query(x)  # [batch_size, seq_len, hidden_dim]
        k = self.key(x)    # [batch_size, seq_len, hidden_dim]
        v = self.value(x)  # [batch_size, seq_len, hidden_dim]
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [batch_size, seq_len, seq_len]
        # 计算注意力权重
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)  # [batch_size, seq_len, seq_len]
        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, seq_len, hidden_dim]
        # 输出线性变换
        output = self.output(attn_output)  # [batch_size, seq_len, input_dim]
        # 残差连接
        return output + x

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制层
    """
    def __init__(self, input_dim, num_heads=4, hidden_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim if hidden_dim is not None else input_dim
        
        # 确保hidden_dim能被num_heads整除
        assert self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = self.hidden_dim // self.num_heads
        
        # 线性变换层
        self.query = nn.Linear(input_dim, self.hidden_dim)
        self.key = nn.Linear(input_dim, self.hidden_dim)
        self.value = nn.Linear(input_dim, self.hidden_dim)
        
        # 输出层
        self.output = nn.Linear(self.hidden_dim, input_dim)
        
        # 缩放因子
        self.scale = torch.tensor(self.head_dim ** 0.5, dtype=torch.float32)
    
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        
        # 计算Q, K, V
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # 重塑为多头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 转置为 [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        
        # 加权求和
        attn_output = torch.matmul(attn_weights, v)
        
        # 转置并重塑
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_dim)
        
        # 输出变换
        output = self.output(attn_output)
        
        # 残差连接
        return output + x
# ========================================================================

print("PyTorch version:", torch.__version__)

# ========================
# 1. 事件分割函数（保留时间戳）
# ========================
def split_events_by_time_gap(df, time_col='EPOCH_yyyy-mm-ddThh:mm:ss.sssZ', gap_threshold=timedelta(hours=1.5)):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce', utc=True)
    df = df.dropna(subset=[time_col])
    df = df.sort_values(by=time_col).reset_index(drop=True)
    df['time_diff_hours'] = df[time_col].diff().dt.total_seconds() / 3600
    df['event_split'] = (df['time_diff_hours'] > gap_threshold.total_seconds()/3600) | (df.index == 0)
    df['event_id'] = df['event_split'].cumsum()
    event_list = [group for _, group in df.groupby('event_id')]
    
    print(f"共识别到 {len(event_list)} 个独立事件")
    for i, event in enumerate(event_list):
        start = event[time_col].min().strftime('%Y-%m-%d %H:%M:%S UTC')
        end = event[time_col].max().strftime('%Y-%m-%d %H:%M:%S UTC')
        length = len(event)
        avg_interval = event['time_diff_hours'].mean() if len(event)>1 else 0
        print(f"事件 {i+1}: {start} ~ {end} | 数据点数: {length} | 平均间隔: {avg_interval:.2f}小时")
    return event_list, df['event_id'].values

# ========================
# 2. 事件相位拆分函数（基于数据自动调整，无重叠区）
# ========================
def split_event_into_phases(event_df, time_col, de1_col):
    """
    将单个事件拆分为「上升段/峰值段/下降段」（基于数据自动调整，无重叠区）
    :param event_df: 单个事件的DataFrame
    :param time_col: 时间列名
    :param de1_col: DE1通量列名
    :return: 各相位数据字典（head/peak/tail）和峰值位置信息
    """
    # 按时间排序（确保时序正确）
    event_df = event_df.sort_values(by=time_col).reset_index(drop=True)
    total_len = len(event_df)
    
    # 获取DE1通量数据
    de1_data = event_df[de1_col].values
    
    # 找到峰值点（最大DE1值的索引）
    peak_idx = np.argmax(de1_data)
    
    # 自动确定上升段和下降段的边界
    # 上升段：从开始到峰值
    # 峰值段：以峰值为中心的区域
    # 下降段：从峰值到结束
    
    # 计算各段长度
    head_len = peak_idx + 1
    tail_len = total_len - peak_idx
    
    # 拆分相位（无重叠区）
    head_phase = event_df.iloc[:peak_idx+1].copy()  # 上升段（从开始到峰值）
    # 峰值段：以峰值为中心的区域
    peak_start = max(0, peak_idx - total_len//10)
    peak_end = min(total_len, peak_idx + total_len//10)
    peak_phase = event_df.iloc[peak_start:peak_end+1].copy()  # 峰值段（中心区域）
    tail_phase = event_df.iloc[peak_idx:].copy()  # 下降段（从峰值到结束）
    
    # 输出相位信息
    print(f"  上升段：{len(head_phase)}个数据点 | 峰值段：{len(peak_phase)}个数据点 | 下降段：{len(tail_phase)}个数据点")
    print(f"  峰值位置: {peak_idx}")
    
    return {
        'head': head_phase,   # 上升段
        'peak': peak_phase,   # 峰值段（核心预测目标）
        'tail': tail_phase,   # 下降段
        'peak_idx': peak_idx  # 峰值位置
    }

# ========================
# 3. 时间轴处理函数（适配局部相位）
# ========================
def process_event_time_axes(time_series, event_ids):
    time_dt = pd.to_datetime(time_series, utc=True)
    unique_events = sorted(np.unique(event_ids))
    plot_pos = 0
    plot_positions = np.zeros(len(time_dt))
    event_ticks = []
    event_labels = []
    event_boundaries = []
    
    for event_id in unique_events:
        event_mask = (event_ids == event_id)
        event_time = time_dt[event_mask]
        if len(event_time) == 0:
            continue
        
        event_start = event_time.min()
        event_end = event_time.max()
        event_duration_hours = (event_end - event_start).total_seconds() / 3600
        # event_time is a DatetimeIndex; subtracting a Timestamp yields a TimedeltaIndex
        # TimedeltaIndex does not have .dt accessor, but provides .total_seconds() directly
        event_time_diff = (event_time - event_start).total_seconds() / 3600
        event_plot_pos = plot_pos + event_time_diff
        
        plot_positions[event_mask] = event_plot_pos
        event_ticks.append(plot_pos)
        event_labels.append(f'Event {int(event_id)}\n{event_start.strftime("%Y-%m-%d %H")}')
        event_boundaries.append(plot_pos + event_duration_hours)
        plot_pos = plot_pos + event_duration_hours + 2
    
    return plot_positions, event_ticks, event_labels, event_boundaries



# --- 核心配置（重点：指定目标事件+目标相位） ---
COLUMN_CONFIG = {
    "time_col": "EPOCH_yyyy-mm-ddThh:mm:ss.sssZ",
    "de1_col": "DE1_.038-.053MEV_E_1/(cm**2-s-sr-MeV)",
    "de4_col": "DE4_.175-.315MEV_E_1/(cm**2-s-sr-MeV)"
}
# 使用原始字符串避免路径中的转义问题
DATA_FILE_PATH = r"sample_data.csv"
SAVE_DIRNAME = 'model_enhanced_with_fourier_attention'

# ========== 关键配置：指定要预测的局部范围 ==========
TARGET_EVENT_ID = 100  # 要预测的目标事件ID（必填）
TARGET_PHASE = 'peak'  # 要预测的相位：head（上升段）/peak（峰值段）/tail（下降段）
# ==================================================

TEST_EVENT_IDS = [TARGET_EVENT_ID]  # 测试集仅包含目标事件
VAL_RATIO_IN_TRAIN = 0.1
EPS = 1e-8

os.makedirs(SAVE_DIRNAME, exist_ok=True)

# --- 加载数据 & 事件分割 ---
df = pd.read_csv(DATA_FILE_PATH, header=0)
missing_cols = [col for col in COLUMN_CONFIG.values() if col not in df.columns]
if missing_cols:
    raise ValueError(f"CSV中缺失以下列名：{missing_cols}！")

event_list, all_event_ids = split_events_by_time_gap(
    df=df,
    time_col=COLUMN_CONFIG["time_col"],
    gap_threshold=timedelta(hours=1.5)
)

# ========================
# 5. 数据清理 + 事件相位拆分（无重叠区）
# ========================
event_clean_data = []
event_id_mapping = []
target_event_phases = {}  # 存储目标事件的所有相位数据

for event_idx, event_df in enumerate(event_list):
    original_event_id = event_idx + 1  # 事件ID从1开始
    print(f"\n处理事件 {original_event_id}：")
    
    # 提取基础数据
    de1 = event_df[COLUMN_CONFIG["de1_col"]].values.astype('float32')
    de4 = event_df[COLUMN_CONFIG["de4_col"]].values.astype('float32')
    time_stamps = event_df[COLUMN_CONFIG["time_col"]].values
    # 使用整型事件ID，避免后续与字符串/浮点混用导致的问题
    event_id_array = np.full(len(de1), original_event_id, dtype=int)
    flux_cols = np.column_stack([de1, de4])
    
    # 清理无效值
    invalid_mask = (
        np.isnan(flux_cols).any(axis=1) |
        (flux_cols < -1e30).any(axis=1) |
        (flux_cols > 1e30).any(axis=1) |
        (flux_cols[:, 0] <= 0)
    )
    clean_flux = flux_cols[~invalid_mask]
    clean_time = time_stamps[~invalid_mask]
    clean_event_ids = event_id_array[~invalid_mask]

    if len(clean_flux) < 10:
        print(f"⚠️  事件 {original_event_id} 数据点过少，跳过")
        continue

    # 计算DE4/DE1比值
    de4_over_de1 = clean_flux[:, 1] / clean_flux[:, 0]
    valid_ratio_mask = (de4_over_de1 > 0) & (de4_over_de1 < 1e5)
    clean_de1 = clean_flux[valid_ratio_mask, 0]
    clean_ratio = de4_over_de1[valid_ratio_mask]
    clean_de4 = clean_flux[valid_ratio_mask, 1]
    clean_time = clean_time[valid_ratio_mask]
    clean_event_ids = clean_event_ids[valid_ratio_mask]

    if len(clean_de1) < 10:
        continue

    # 构建清理后的事件数据
    clean_event_df = pd.DataFrame({
        'de1': clean_de1,
        'de4_over_de1': clean_ratio,
        'de4': clean_de4,
        'time': clean_time,
        'event_id': clean_event_ids
    })
    
    # 关键：如果是目标事件，拆分所有相位
    if original_event_id == TARGET_EVENT_ID:
        print(f"📌 拆分目标事件 {TARGET_EVENT_ID} 的所有相位：")
        # 拆分相位（无重叠区）
        event_phases = split_event_into_phases(
            event_df=clean_event_df,
            time_col='time',
            de1_col='de1'
        )
        # 存储所有相位数据
        target_event_phases = event_phases
        print(f"✅ 提取目标事件 {TARGET_EVENT_ID} 的所有相位数据")
    
    # 保存所有有效事件数据（用于训练）
    event_clean_data.append({
        'de1': clean_de1,
        'de4_over_de1': clean_ratio,
        'de4': clean_de4,
        'time': clean_time,
        'event_id': clean_event_ids
    })
    event_id_mapping.append(original_event_id)

# 验证目标事件数据是否存在
if not target_event_phases:
    raise ValueError(f"❌ 目标事件 {TARGET_EVENT_ID} 无有效数据！")

print(f"\n有效事件数量: {len(event_clean_data)}")
print(f"有效事件的原始ID列表: {event_id_mapping}")

# ========================
# 6. 划分训练/验证/测试集
# ========================
np.random.seed(42)

# 验证测试事件ID有效性
invalid_test_ids = [tid for tid in TEST_EVENT_IDS if tid not in event_id_mapping]
if invalid_test_ids:
    raise ValueError(f"❌ 测试事件ID {invalid_test_ids} 不存在于有效事件列表中！")

# 划分测试集（仅目标事件）和训练+验证集
test_event_idx = [i for i, eid in enumerate(event_id_mapping) if eid in TEST_EVENT_IDS]
train_val_idx = [i for i, eid in enumerate(event_id_mapping) if eid not in TEST_EVENT_IDS]

if len(test_event_idx) == 0:
    raise ValueError(f"❌ 没有找到匹配的测试事件！")
if len(train_val_idx) == 0:
    raise ValueError(f"❌ 训练+验证集为空！请减少测试事件数量")

# 划分训练/验证集
val_num = max(1, int(len(train_val_idx) * VAL_RATIO_IN_TRAIN))
train_val_shuffle = np.random.permutation(train_val_idx)
val_event_idx = train_val_shuffle[:val_num]
train_event_idx = train_val_shuffle[val_num:]

print(f"\n📊 事件划分结果：")
print(f"目标测试事件ID: {TEST_EVENT_IDS} (预测所有相位)")
print(f"训练集事件索引: {train_event_idx} (对应原始ID: {[event_id_mapping[i] for i in train_event_idx]})")
print(f"验证集事件索引: {val_event_idx} (对应原始ID: {[event_id_mapping[i] for i in val_event_idx]})")

# 合并训练/验证数据（所有事件的全量数据）
def merge_events(events):
    merged = {
        'de1': [],
        'de4_over_de1': [],
        'de4': [],
        'time': [],
        'event_id': []
    }
    for e in events:
        merged['de1'].append(e['de1'])
        merged['de4_over_de1'].append(e['de4_over_de1'])
        merged['de4'].append(e['de4'])
        merged['time'].append(e['time'])
        merged['event_id'].append(e['event_id'])
    
    for key in merged:
        merged[key] = np.concatenate(merged[key], axis=0)
    return merged

train_data = merge_events([event_clean_data[i] for i in train_event_idx])
val_data = merge_events([event_clean_data[i] for i in val_event_idx])

# 测试集：使用目标事件的所有相位数据
test_data = {
    'de1': target_event_phases[TARGET_PHASE]['de1'].values,
    'de4_over_de1': target_event_phases[TARGET_PHASE]['de4_over_de1'].values,
    'de4': target_event_phases[TARGET_PHASE]['de4'].values,
    'time': target_event_phases[TARGET_PHASE]['time'].values,
    'event_id': target_event_phases[TARGET_PHASE]['event_id'].values
}

print(f"\n📈 数据规模（聚焦目标相位）：")
print(f"训练集: DE1样本数={len(train_data['de1'])}")
print(f"验证集: DE1样本数={len(val_data['de1'])}")
print(f"测试集（目标事件{TARGET_EVENT_ID}的{TARGET_PHASE}段）: DE1样本数={len(test_data['de1'])}")

# ========================
# 7. 数据归一化（复用原逻辑）
# ========================
train_de1_log = np.log10(train_data['de1'] + EPS)
de1_log_mean = np.mean(train_de1_log)
de1_log_std = np.std(train_de1_log)

train_ratio_log = np.log10(train_data['de4_over_de1'] + EPS)
ratio_log_mean = np.mean(train_ratio_log)
ratio_log_std = np.std(train_ratio_log)

def normalize_de1(de1):
    de1_log = np.log10(de1 + EPS)
    return (de1_log - de1_log_mean) / de1_log_std

def normalize_ratio(ratio):
    ratio_log = np.log10(ratio + EPS)
    return (ratio_log - ratio_log_mean) / ratio_log_std

def denormalize_ratio(ratio_norm):
    ratio_log = ratio_norm * ratio_log_std + ratio_log_mean
    return 10**ratio_log - EPS

# 提取傅里叶特征
print("\n提取傅里叶特征...")
train_fourier = extract_windowed_fourier_features(train_data['de1'], window_size=20, n_harmonics=5)
val_fourier = extract_windowed_fourier_features(val_data['de1'], window_size=20, n_harmonics=5)
test_fourier = extract_windowed_fourier_features(test_data['de1'], window_size=20, n_harmonics=5)

# 归一化傅里叶特征
train_fourier_mean = np.mean(train_fourier, axis=0)
train_fourier_std = np.std(train_fourier, axis=0) + EPS
train_fourier_norm = (train_fourier - train_fourier_mean) / train_fourier_std
val_fourier_norm = (val_fourier - train_fourier_mean) / train_fourier_std
test_fourier_norm = (test_fourier - train_fourier_mean) / train_fourier_std

# 归一化DE1和比值
train_de1_norm = normalize_de1(train_data['de1'])
train_ratio_norm = normalize_ratio(train_data['de4_over_de1'])

val_de1_norm = normalize_de1(val_data['de1'])
val_ratio_norm = normalize_ratio(val_data['de4_over_de1'])

test_de1_norm = normalize_de1(test_data['de1'])
test_ratio_norm = normalize_ratio(test_data['de4_over_de1'])

# ========================
# 8. 构建Dataset和DataLoader
# ========================
class EnhancedDE1ToRatioDataset(Dataset):
    def __init__(self, de1_norm, fourier_features, ratio_norm):
        self.de1_norm = np.array(de1_norm).reshape(-1, 1).astype(np.float32)
        self.fourier_features = np.array(fourier_features).astype(np.float32)
        self.ratio_norm = np.array(ratio_norm).reshape(-1, 1).astype(np.float32)
    
    def __len__(self):
        return len(self.de1_norm)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.de1_norm[idx], dtype=torch.float32),
            torch.tensor(self.fourier_features[idx], dtype=torch.float32),
            torch.tensor(self.ratio_norm[idx], dtype=torch.float32)
        )

BATCH_SIZE = 64
train_dataset = EnhancedDE1ToRatioDataset(train_de1_norm, train_fourier_norm, train_ratio_norm)
val_dataset = EnhancedDE1ToRatioDataset(val_de1_norm, val_fourier_norm, val_ratio_norm)
test_dataset = EnhancedDE1ToRatioDataset(test_de1_norm, test_fourier_norm, test_ratio_norm)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ========================
# 9. 定义增强型相位感知模型（含傅里叶特征和注意力机制）
# ========================
class EnhancedPhaseAwareModel(nn.Module):
    def __init__(self, input_dim=1, fourier_dim=15, phase_embed_dim=8, hidden_dim=64, output_dim=1, use_attention=True):
        super().__init__()
        # 相位嵌入层
        self.phase_embed = nn.Embedding(num_embeddings=3, embedding_dim=phase_embed_dim)
        
        # 特征融合层
        total_input_dim = input_dim + fourier_dim + phase_embed_dim
        self.feature_fusion = nn.Linear(total_input_dim, hidden_dim)
        
        # 注意力机制
        self.use_attention = use_attention
        if use_attention:
            # 自注意力层
            self.attention = SelfAttention(hidden_dim)
        
        # 主网络
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, de1, fourier_features, phase_id):
        # 相位嵌入
        phase_emb = self.phase_embed(phase_id)
        
        # 拼接特征
        x = torch.cat([de1, fourier_features, phase_emb], dim=1)
        
        # 特征融合
        x = self.feature_fusion(x)
        
        # 应用注意力机制
        if self.use_attention:
            # 添加序列维度以适应注意力机制
            x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            x = self.attention(x)
            x = x.squeeze(1)  # [batch_size, hidden_dim]
        
        # 主网络预测
        return self.net(x)

# 相位ID映射
PHASE_MAP = {'head': 0, 'peak': 1, 'tail': 2}

# ========================
# 9.1 定义多种模型变体用于消融实验
# ========================
class BaseModel(nn.Module):
    """基础模型（仅使用DE1特征）"""
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, de1, fourier_features=None, phase_id=None):
        return self.net(de1)

class ModelWithFourier(nn.Module):
    """基础模型 + 傅里叶特征"""
    def __init__(self, input_dim=1, fourier_dim=15, hidden_dim=64, output_dim=1):
        super().__init__()
        total_input_dim = input_dim + fourier_dim
        self.feature_fusion = nn.Linear(total_input_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, de1, fourier_features, phase_id=None):
        x = torch.cat([de1, fourier_features], dim=1)
        x = self.feature_fusion(x)
        return self.net(x)

class ModelWithPhase(nn.Module):
    """基础模型 + 相位嵌入"""
    def __init__(self, input_dim=1, phase_embed_dim=8, hidden_dim=64, output_dim=1):
        super().__init__()
        self.phase_embed = nn.Embedding(num_embeddings=3, embedding_dim=phase_embed_dim)
        total_input_dim = input_dim + phase_embed_dim
        self.feature_fusion = nn.Linear(total_input_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, de1, fourier_features=None, phase_id=None):
        phase_emb = self.phase_embed(phase_id)
        x = torch.cat([de1, phase_emb], dim=1)
        x = self.feature_fusion(x)
        return self.net(x)

class ModelWithAttention(nn.Module):
    """基础模型 + 注意力机制"""
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.attention = SelfAttention(hidden_dim)
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, de1, fourier_features=None, phase_id=None):
        x = self.input_layer(de1)
        x = x.unsqueeze(1)  # 添加序列维度
        x = self.attention(x)
        x = x.squeeze(1)  # 移除序列维度
        return self.net(x)

# 消融实验配置
ABLATION_MODELS = {
    'BaseModel': BaseModel,
    'ModelWithFourier': ModelWithFourier,
    'ModelWithPhase': ModelWithPhase,
    'ModelWithAttention': ModelWithAttention,
    'EnhancedPhaseAwareModel': EnhancedPhaseAwareModel
}

# 训练和评估单个模型的函数
def train_and_evaluate_model(model_name, model_class, train_loader, val_loader, test_loader, test_data, target_event_phases, TARGET_PHASE, SAVE_DIRNAME):
    """
    训练和评估单个模型变体
    """
    print(f"\n{'='*60}")
    print(f"训练模型: {model_name}")
    print(f"{'='*60}")
    
    # 初始化模型
    if model_name == 'BaseModel':
        model = model_class(input_dim=1, hidden_dim=64, output_dim=1)
    elif model_name == 'ModelWithFourier':
        model = model_class(input_dim=1, fourier_dim=15, hidden_dim=64, output_dim=1)
    elif model_name == 'ModelWithPhase':
        model = model_class(input_dim=1, phase_embed_dim=8, hidden_dim=64, output_dim=1)
    elif model_name == 'ModelWithAttention':
        model = model_class(input_dim=1, hidden_dim=64, output_dim=1)
    else:  # EnhancedPhaseAwareModel
        model = model_class(
            input_dim=1, 
            fourier_dim=15, 
            phase_embed_dim=8, 
            hidden_dim=64, 
            output_dim=1, 
            use_attention=True
        )
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 训练参数
    EPOCHS = 20
    PATIENCE = 5
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # 训练循环
    for epoch in range(EPOCHS):
        # 训练
        model.train()
        train_loss = 0.0
        for de1_batch, fourier_batch, ratio_batch in train_loader:
            optimizer.zero_grad()
            
            # 根据模型类型生成适当的输入
            if model_name == 'BaseModel':
                pred_ratio = model(de1_batch)
            elif model_name == 'ModelWithFourier':
                pred_ratio = model(de1_batch, fourier_batch)
            elif model_name == 'ModelWithPhase':
                phase_id = torch.zeros_like(de1_batch[:, 0], dtype=torch.long)
                pred_ratio = model(de1_batch, phase_id=phase_id)
            elif model_name == 'ModelWithAttention':
                pred_ratio = model(de1_batch)
            else:  # EnhancedPhaseAwareModel
                phase_id = torch.zeros_like(de1_batch[:, 0], dtype=torch.long)
                pred_ratio = model(de1_batch, fourier_batch, phase_id)
            
            loss = criterion(pred_ratio, ratio_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * de1_batch.shape[0]
        
        avg_train_loss = train_loss / len(train_dataset)
        train_losses.append(avg_train_loss)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for de1_batch, fourier_batch, ratio_batch in val_loader:
                # 根据模型类型生成适当的输入
                if model_name == 'BaseModel':
                    pred_ratio = model(de1_batch)
                elif model_name == 'ModelWithFourier':
                    pred_ratio = model(de1_batch, fourier_batch)
                elif model_name == 'ModelWithPhase':
                    phase_id = torch.zeros_like(de1_batch[:, 0], dtype=torch.long)
                    pred_ratio = model(de1_batch, phase_id=phase_id)
                elif model_name == 'ModelWithAttention':
                    pred_ratio = model(de1_batch)
                else:  # EnhancedPhaseAwareModel
                    phase_id = torch.zeros_like(de1_batch[:, 0], dtype=torch.long)
                    pred_ratio = model(de1_batch, fourier_batch, phase_id)
                
                loss = criterion(pred_ratio, ratio_batch)
                val_loss += loss.item() * de1_batch.shape[0]
        
        avg_val_loss = val_loss / len(val_dataset)
        val_losses.append(avg_val_loss)

        # 早停
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or patience_counter == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if patience_counter >= PATIENCE:
            print(f"🛑 早停触发！在Epoch {epoch+1} 停止训练")
            break
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    
    # 测试模型
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for de1_batch, fourier_batch, ratio_batch in test_loader:
            # 根据模型类型生成适当的输入
            if model_name == 'BaseModel':
                pred_ratio = model(de1_batch)
            elif model_name == 'ModelWithFourier':
                pred_ratio = model(de1_batch, fourier_batch)
            elif model_name == 'ModelWithPhase':
                phase_id = PHASE_MAP[TARGET_PHASE]
                phase_ids = torch.full((len(de1_batch),), phase_id, dtype=torch.long)
                pred_ratio = model(de1_batch, phase_id=phase_ids)
            elif model_name == 'ModelWithAttention':
                pred_ratio = model(de1_batch)
            else:  # EnhancedPhaseAwareModel
                phase_id = PHASE_MAP[TARGET_PHASE]
                phase_ids = torch.full((len(de1_batch),), phase_id, dtype=torch.long)
                pred_ratio = model(de1_batch, fourier_batch, phase_ids)
            
            test_predictions.append(pred_ratio.cpu().numpy())
            test_targets.append(ratio_batch.cpu().numpy())
    
    # 计算评估指标
    test_predictions = np.concatenate(test_predictions, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)
    
    # 确保是一维数组
    test_predictions = np.squeeze(test_predictions)
    test_targets = np.squeeze(test_targets)
    
    # 反归一化
    test_pred_ratio = denormalize_ratio(test_predictions)
    test_true_ratio = denormalize_ratio(test_targets)
    
    # 确保是一维数组
    test_pred_ratio = np.squeeze(test_pred_ratio)
    test_true_ratio = np.squeeze(test_true_ratio)
    
    # 计算DE4预测值
    test_pred_de4 = test_pred_ratio * test_data['de1']
    test_true_de4 = test_data['de4']
    
    # 确保长度匹配
    assert len(test_pred_de4) == len(test_true_de4), f"预测值和真实值长度不匹配: {len(test_pred_de4)} != {len(test_true_de4)}"
    
    # 计算评估指标
    mse = np.mean((test_pred_de4 - test_true_de4) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(test_pred_de4 - test_true_de4))
    r2 = r2_score(test_true_de4, test_pred_de4)
    
    # 计算平均绝对百分比误差
    mape = np.mean(np.abs((test_pred_de4 - test_true_de4) / (test_true_de4 + 1e-8))) * 100
    
    print(f"\n📊 模型 {model_name} 性能评估：")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²: {r2:.4f}")
    
    # 保存模型性能
    performance = {
        'model_name': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model': model
    }
    
    return performance

# ========================
# 10. 运行消融实验
# ========================
print("\n开始消融实验...")
ablations_results = {}

# 为每个模型变体运行训练和评估
for model_name, model_class in ABLATION_MODELS.items():
    performance = train_and_evaluate_model(
        model_name=model_name,
        model_class=model_class,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test_data=test_data,
        target_event_phases=target_event_phases,
        TARGET_PHASE=TARGET_PHASE,
        SAVE_DIRNAME=SAVE_DIRNAME
    )
    ablations_results[model_name] = performance

# 选择最佳模型
best_model_name = min(ablations_results, key=lambda x: ablations_results[x]['mse'])
best_model_performance = ablations_results[best_model_name]
best_model = ablations_results[best_model_name]['model']

print(f"\n{'='*80}")
print("消融实验总结")
print(f"{'='*80}")
print(f"最佳模型: {best_model_name}")
print(f"最佳性能 - MSE: {best_model_performance['mse']:.4f}, RMSE: {best_model_performance['rmse']:.4f}, R²: {best_model_performance['r2']:.4f}")
print(f"\n所有模型性能:")
for model_name, performance in ablations_results.items():
    print(f"{model_name}:")
    print(f"  - MSE: {performance['mse']:.4f}")
    print(f"  - RMSE: {performance['rmse']:.4f}")
    print(f"  - MAE: {performance['mae']:.4f}")
    print(f"  - MAPE: {performance['mape']:.2f}%")
    print(f"  - R²: {performance['r2']:.4f}")
print(f"{'='*80}")

# 绘制所有模型的损失曲线
plt.figure(figsize=(12, 8))
for model_name, performance in ablations_results.items():
    plt.plot(performance['train_losses'], label=f'{model_name} - Train', linestyle='-')
    plt.plot(performance['val_losses'], label=f'{model_name} - Val', linestyle='--')

plt.title('Training & Validation Loss for Different Model Variants')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss (Normalized)')
plt.legend()
plt.grid(True, ls='--')
plt.tight_layout()
plt.savefig(f'{SAVE_DIRNAME}/ablation_losses.png', dpi=150)
plt.show()

# 绘制消融实验性能对比图
metrics = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R²']

for metric in metrics:
    plt.figure(figsize=(10, 6))
    values = []
    model_names = []
    
    for model_name, performance in ablations_results.items():
        if metric == 'MSE':
            values.append(performance['mse'])
        elif metric == 'RMSE':
            values.append(performance['rmse'])
        elif metric == 'MAE':
            values.append(performance['mae'])
        elif metric == 'MAPE':
            values.append(performance['mape'])
        elif metric == 'R²':
            values.append(performance['r2'])
        model_names.append(model_name)
    
    # 绘制条形图
    bars = plt.bar(model_names, values, color=['blue', 'green', 'red', 'purple', 'orange'])
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', ha='center', va='bottom')
    
    plt.title(f'Ablation Study - {metric} Comparison')
    plt.xlabel('Model Variant')
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(SAVE_DIRNAME, f'ablation_{metric.lower()}_comparison.png')
    plt.savefig(save_path, dpi=150)
    plt.show()

# ========================
# 11. 测试模型（无边界平滑）
# ========================
# 为所有模型变体生成预测结果
all_models_predictions = {}

for model_name, model_info in ablations_results.items():
    model = model_info['model']
    model.eval()
    
    # 预测目标事件的所有相位
    phase_preds = {}
    phase_times = {}
    phase_positions = {}
    
    for phase_name in ['head', 'peak', 'tail']:
        phase_data = target_event_phases[phase_name]
        de1 = phase_data['de1'].values
        time = phase_data['time'].values
        event_id = phase_data['event_id'].values
        
        # 提取傅里叶特征
        fourier_features = extract_windowed_fourier_features(de1, window_size=20, n_harmonics=5)
        fourier_norm = (fourier_features - train_fourier_mean) / train_fourier_std
        
        # 归一化
        de1_norm = normalize_de1(de1)
        de1_tensor = torch.tensor(de1_norm.reshape(-1, 1), dtype=torch.float32)
        fourier_tensor = torch.tensor(fourier_norm, dtype=torch.float32)
        phase_id = PHASE_MAP[phase_name]
        phase_ids = torch.full((len(de1),), phase_id, dtype=torch.long)
        
        # 预测
        with torch.no_grad():
            if model_name == 'BaseModel':
                pred_norm = model(de1_tensor).numpy().flatten()
            elif model_name == 'ModelWithFourier':
                pred_norm = model(de1_tensor, fourier_tensor).numpy().flatten()
            elif model_name == 'ModelWithPhase':
                pred_norm = model(de1_tensor, phase_id=phase_ids).numpy().flatten()
            elif model_name == 'ModelWithAttention':
                pred_norm = model(de1_tensor).numpy().flatten()
            else:  # EnhancedPhaseAwareModel
                pred_norm = model(de1_tensor, fourier_tensor, phase_ids).numpy().flatten()
        
        # 反归一化
        pred_ratio = denormalize_ratio(pred_norm)
        pred_ratio = np.clip(pred_ratio, 0, None)
        pred_de4 = pred_ratio * de1
        pred_de4 = np.clip(pred_de4, 0, None)
        
        # 存储结果
        phase_preds[phase_name] = pred_de4
        phase_times[phase_name] = time
        
        # 计算时间位置
        plot_positions, _, _, _ = process_event_time_axes(time, event_id)
        phase_positions[phase_name] = plot_positions
    
    all_models_predictions[model_name] = {
        'phase_preds': phase_preds,
        'phase_times': phase_times,
        'phase_positions': phase_positions
    }

# 使用最佳模型的预测结果
best_model_predictions = all_models_predictions[best_model_name]
all_phase_preds = best_model_predictions['phase_preds']
all_phase_times = best_model_predictions['phase_times']
all_phase_positions = best_model_predictions['phase_positions']

# 直接使用峰值段的预测结果
test_pred_de4 = all_phase_preds['peak']

# 提取目标相位的真实值
test_true_de4 = target_event_phases[TARGET_PHASE]['de4'].values
test_true_ratio = target_event_phases[TARGET_PHASE]['de4_over_de1'].values
test_data_de1 = target_event_phases[TARGET_PHASE]['de1'].values

# 计算预测比值
test_pred_ratio = test_pred_de4 / test_data_de1
test_pred_ratio = np.clip(test_pred_ratio, 0, None)

# 评估指标
ratio_rmse = np.sqrt(np.mean((test_pred_ratio - test_true_ratio)**2))
ratio_mae = np.mean(np.abs(test_pred_ratio - test_true_ratio))
ratio_r2 = r2_score(test_true_ratio, test_pred_ratio)

de4_rmse = np.sqrt(np.mean((test_pred_de4 - test_true_de4)**2))
de4_mae = np.mean(np.abs(test_pred_de4 - test_true_de4))
de4_r2 = r2_score(test_true_de4, test_pred_de4)

print(f"\n==================== 目标相位预测结果（事件{TARGET_EVENT_ID} - {TARGET_PHASE}段） ====================")
print(f"DE4/DE1 比值预测 - RMSE: {ratio_rmse:.6f} | MAE: {ratio_mae:.6f} | R²: {ratio_r2:.4f}")
print(f"DE4 通量预测     - RMSE: {de4_rmse:.6f} | MAE: {de4_mae:.6f} | R²: {de4_r2:.4f}")

# ========================
# 12. 可视化（仅展示目标相位）
# ========================
# 12.1 目标相位：DE1 vs 比值散点图
plt.figure(figsize=(12, 6))
sample_idx = np.random.choice(len(test_data_de1), size=min(500, len(test_data_de1)), replace=False)
plt.scatter(test_data_de1[sample_idx], test_true_ratio[sample_idx], s=8, alpha=0.7, label='True Ratio', color='blue')
plt.scatter(test_data_de1[sample_idx], test_pred_ratio[sample_idx], s=8, alpha=0.7, label='Pred Ratio', color='red')
plt.xlabel('DE1 Flux (Event {0} - {1} phase)'.format(TARGET_EVENT_ID, TARGET_PHASE))
plt.ylabel('DE4/DE1 Ratio')
plt.xscale('log')
plt.yscale('log')
plt.title(f'DE1 vs DE4/DE1 Ratio (Event {TARGET_EVENT_ID} - {TARGET_PHASE} phase)')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{SAVE_DIRNAME}/target_phase_de1_vs_ratio.png', dpi=150)
plt.show()

# 12.2 目标相位：DE4通量时序对比（核心可视化）
plt.figure(figsize=(16, 8))
# 处理时间轴
plot_time = target_event_phases[TARGET_PHASE]['time'].values
plot_event_ids = target_event_phases[TARGET_PHASE]['event_id'].values
plot_positions, event_ticks, event_labels, event_boundaries = process_event_time_axes(plot_time, plot_event_ids)

# 绘制目标相位的通量曲线
plt.plot(plot_positions, test_true_de4, color='blue', linewidth=1.5, label='True DE4 Flux')
plt.plot(plot_positions, test_pred_de4, color='green', linewidth=1.5, alpha=0.8, label='Pred DE4 Flux (Smoothed)')

# 添加事件分隔线
for boundary in event_boundaries[:-1]:
    plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)

# 样式设置
ax = plt.gca()
ax.xaxis.set_major_locator(FixedLocator(event_ticks))
def _event_label_formatter(x, pos):
    """Formatter for x-axis event labels. Handles cases where ``pos`` may be None
    (e.g., when called from interactive events like mouse move). If pos is None,
    find the nearest tick to x and return its label; otherwise use pos as index.
    """
    try:
        if pos is None:
            # nearest tick index
            ticks = np.array(event_ticks)
            if ticks.size == 0:
                return ''
            idx = int(np.argmin(np.abs(ticks - x)))
            return event_labels[idx] if idx < len(event_labels) else ''
        pos_int = int(pos)
        return event_labels[pos_int] if 0 <= pos_int < len(event_labels) else ''
    except Exception:
        return ''

ax.xaxis.set_major_formatter(plt.FuncFormatter(_event_label_formatter))
plt.setp(ax.xaxis.get_majorticklabels(), fontsize=10, ha='center')
ax.set_xlabel(f'Event {TARGET_EVENT_ID} - {TARGET_PHASE} phase Time (UTC)', fontsize=12)
ax.set_ylabel('DE4 Flux (1/(cm²·s·sr·MeV))', fontsize=12)
ax.set_yscale('log')
ax.set_title(f'True vs Pred DE4 Flux (Event {TARGET_EVENT_ID} - {TARGET_PHASE} phase) - Smoothed', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, which='major', ls='-', alpha=0.7)
ax.grid(True, which='minor', ls='--', alpha=0.4)
ax.set_xlim(plot_positions.min() - 0.5, plot_positions.max() + 0.5)
plt.tight_layout()
plt.savefig(f'{SAVE_DIRNAME}/target_phase_de4_timeseries.png', dpi=150)
plt.show()

# 12.3 目标相位：通量预测散点图（y=x参考线）
plt.figure(figsize=(12, 6))
plt.scatter(test_true_de4, test_pred_de4, s=8, alpha=0.7, color='green')
# 添加y=x参考线
min_de4 = min(test_true_de4.min(), test_pred_de4.min())
max_de4 = max(test_true_de4.max(), test_pred_de4.max())
plt.plot([min_de4, max_de4], [min_de4, max_de4], 'k--', linewidth=1.5, label='Perfect Prediction (y=x)')
plt.xlabel(f'True DE4 Flux (Event {TARGET_EVENT_ID} - {TARGET_PHASE} phase)')
plt.ylabel(f'Pred DE4 Flux (Event {TARGET_EVENT_ID} - {TARGET_PHASE} phase)')
plt.xscale('log')
plt.yscale('log')
plt.title(f'True vs Pred DE4 Flux (Event {TARGET_EVENT_ID} - {TARGET_PHASE} phase) - Smoothed', fontsize=14)
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{SAVE_DIRNAME}/target_phase_de4_scatter.png', dpi=150)
plt.show()

# 12.4 整个事件：实际DE4和仅在峰值段显示预测DE4
plt.figure(figsize=(16, 8))

# 获取整个事件的时间和实际DE4数据
# 修复：使用pd.concat替代已废弃的append方法
full_event_data = pd.concat([
    target_event_phases['head'],
    target_event_phases['peak'],
    target_event_phases['tail']
], ignore_index=True)

full_time = full_event_data['time'].values
full_true_de4 = full_event_data['de4'].values

# 处理时间轴
plot_event_ids = full_event_data['event_id'].values
plot_positions, event_ticks, event_labels, event_boundaries = process_event_time_axes(full_time, plot_event_ids)

# 绘制整个事件的实际DE4
plt.plot(plot_positions, full_true_de4, color='blue', linewidth=1.5, label='True DE4 Flux (Whole Event)')

# 绘制仅在峰值段的预测DE4
peak_time = target_event_phases['peak']['time'].values
peak_true_de4 = target_event_phases['peak']['de4'].values
peak_plot_positions, _, _, _ = process_event_time_axes(peak_time, plot_event_ids[:len(peak_time)])

# 调整峰值段预测值的位置以匹配整个事件的时间轴
# 找到峰值段在整个事件中的起始位置
peak_start_in_full = len(target_event_phases['head'])
peak_end_in_full = peak_start_in_full + len(target_event_phases['peak'])
peak_positions_in_full = plot_positions[peak_start_in_full:peak_end_in_full]

# 绘制峰值段的预测DE4
plt.plot(peak_positions_in_full, test_pred_de4, color='green', linewidth=1.5, alpha=0.8, label='Pred DE4 Flux (Peak Phase Only)')

# 添加峰值标记
peak_idx = target_event_phases.get('peak_idx', peak_start_in_full + len(target_event_phases['peak'])//2)
if peak_idx < len(plot_positions):
    plt.axvline(x=plot_positions[peak_idx], color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Peak Position')

# 添加事件分隔线
for boundary in event_boundaries[:-1]:
    plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)

# 样式设置
ax = plt.gca()
ax.xaxis.set_major_locator(FixedLocator(event_ticks))
def _event_label_formatter(x, pos):
    """Formatter for x-axis event labels. Handles cases where ``pos`` may be None
    (e.g., when called from interactive events like mouse move). If pos is None,
    find the nearest tick to x and return its label; otherwise use pos as index.
    """
    try:
        if pos is None:
            # nearest tick index
            ticks = np.array(event_ticks)
            if ticks.size == 0:
                return ''
            idx = int(np.argmin(np.abs(ticks - x)))
            return event_labels[idx] if idx < len(event_labels) else ''
        pos_int = int(pos)
        return event_labels[pos_int] if 0 <= pos_int < len(event_labels) else ''
    except Exception:
        return ''

ax.xaxis.set_major_formatter(plt.FuncFormatter(_event_label_formatter))
plt.setp(ax.xaxis.get_majorticklabels(), fontsize=10, ha='center')
ax.set_xlabel(f'Event {TARGET_EVENT_ID} - Full Time (UTC)', fontsize=12)
ax.set_ylabel('DE4 Flux (1/(cm²·s·sr·MeV))', fontsize=12)
ax.set_yscale('log')
ax.set_title(f'True DE4 Flux (Whole Event) vs Pred DE4 Flux (Peak Phase Only) - Event {TARGET_EVENT_ID}', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, which='major', ls='-', alpha=0.7)
ax.grid(True, which='minor', ls='--', alpha=0.4)
ax.set_xlim(plot_positions.min() - 0.5, plot_positions.max() + 0.5)
plt.tight_layout()
plt.savefig(f'{SAVE_DIRNAME}/whole_event_de4_with_peak_prediction.png', dpi=150)
plt.show()

# ========================
# 13. 单样本预测（目标相位内的任意点）
# ========================
def predict_de4_ratio_and_flux(de1_input, phase_name=TARGET_PHASE):
    # 构建单样本的窗口（使用相同值填充窗口）
    window_de1 = np.full(20, de1_input)
    
    # 提取傅里叶特征
    fourier_features = extract_windowed_fourier_features(window_de1, window_size=20, n_harmonics=5)
    fourier_norm = (fourier_features[-1] - train_fourier_mean) / train_fourier_std
    
    # 归一化DE1
    de1_norm = normalize_de1(np.array([de1_input]))
    
    # 转换为张量
    de1_tensor = torch.tensor(de1_norm.reshape(1, 1), dtype=torch.float32)
    fourier_tensor = torch.tensor(fourier_norm.reshape(1, -1), dtype=torch.float32)
    phase_id = torch.tensor([PHASE_MAP[phase_name]], dtype=torch.long)
    
    best_model.eval()
    with torch.no_grad():
        if best_model_name == 'BaseModel':
            pred_norm = best_model(de1_tensor).numpy()[0][0]
        elif best_model_name == 'ModelWithFourier':
            pred_norm = best_model(de1_tensor, fourier_tensor).numpy()[0][0]
        elif best_model_name == 'ModelWithAttention':
            pred_norm = best_model(de1_tensor).numpy()[0][0]
        elif best_model_name == 'ModelWithPhase':
            pred_norm = best_model(de1_tensor, phase_id=phase_id).numpy()[0][0]
        else:  # EnhancedPhaseAwareModel
            pred_norm = best_model(de1_tensor, fourier_tensor, phase_id).numpy()[0][0]
    pred_ratio = denormalize_ratio(pred_norm)
    pred_ratio = max(pred_ratio, 0)
    pred_de4 = pred_ratio * de1_input
    pred_de4 = max(pred_de4, 0)
    return pred_ratio, pred_de4

# 从目标相位中随机选几个点演示
demo_size = min(5, len(test_data_de1))
demo_idx = np.random.choice(len(test_data_de1), size=demo_size, replace=False)
demo_de1_values = test_data_de1[demo_idx]
print(f"\n==================== 目标相位内单样本预测演示（事件{TARGET_EVENT_ID} - {TARGET_PHASE}段） ====================")
for i, de1_val in enumerate(demo_de1_values):
    true_ratio = test_true_ratio[demo_idx[i]]
    true_de4 = test_true_de4[demo_idx[i]]
    pred_ratio, pred_de4 = predict_de4_ratio_and_flux(de1_val)
    print(f"样本{i+1} - 输入DE1: {de1_val:.2f} → 真实比值: {true_ratio:.4f} | 预测比值: {pred_ratio:.4f} → 真实DE4: {true_de4:.2f} | 预测DE4: {pred_de4:.2f}")

# ========================
# 14. 消融实验结果可视化
# ========================
print("\n生成消融实验结果可视化...")

# 14.1 生成消融实验性能对比图
metrics = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R²']

for metric in metrics:
    plt.figure(figsize=(10, 6))
    values = []
    model_names = []
    
    for model_name, performance in ablations_results.items():
        if metric == 'MSE':
            values.append(performance['mse'])
        elif metric == 'RMSE':
            values.append(performance['rmse'])
        elif metric == 'MAE':
            values.append(performance['mae'])
        elif metric == 'MAPE':
            values.append(performance['mape'])
        elif metric == 'R²':
            values.append(performance['r2'])
        model_names.append(model_name)
    
    # 绘制条形图
    bars = plt.bar(model_names, values, color=['blue', 'green', 'red', 'purple', 'orange'])
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', ha='center', va='bottom')
    
    plt.title(f'Ablation Study - {metric} Comparison', fontsize=14)
    plt.xlabel('Model Variant', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(SAVE_DIRNAME, f'ablation_{metric.lower()}_comparison.png')
    plt.savefig(save_path, dpi=150)
    print(f"✅ 消融实验 {metric} 对比图已保存至: {save_path}")
    plt.show()

# 14.2 生成所有模型变体的预测对比图（仅峰值段）
plt.figure(figsize=(16, 8))

# 获取整个事件的时间和实际DE4数据
full_event_data = pd.concat([
    target_event_phases['head'],
    target_event_phases['peak'],
    target_event_phases['tail']
], ignore_index=True)

full_time = full_event_data['time'].values
full_true_de4 = full_event_data['de4'].values
plot_event_ids = full_event_data['event_id'].values
plot_positions, event_ticks, event_labels, event_boundaries = process_event_time_axes(full_time, plot_event_ids)

# 绘制整个事件的实际DE4
plt.plot(plot_positions, full_true_de4, color='blue', linewidth=1.5, label='True DE4 Flux (Whole Event)')

# 标记峰值位置
peak_idx = target_event_phases.get('peak_idx', len(target_event_phases['head']) + len(target_event_phases['peak'])//2)
if peak_idx < len(plot_positions):
    plt.axvline(x=plot_positions[peak_idx], color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Peak Position')

# 标记峰值段范围
peak_start_in_full = len(target_event_phases['head'])
peak_end_in_full = peak_start_in_full + len(target_event_phases['peak'])
peak_positions_in_full = plot_positions[peak_start_in_full:peak_end_in_full]
plt.axvspan(peak_positions_in_full[0], peak_positions_in_full[-1], alpha=0.1, color='yellow', label='Peak Phase Range')

# 为每个模型绘制预测结果
colors = ['green', 'red', 'purple', 'orange', 'brown']
for i, (model_name, predictions) in enumerate(all_models_predictions.items()):
    model_pred_de4 = predictions['phase_preds']['peak']
    plt.plot(peak_positions_in_full, model_pred_de4, linestyle='--', label=f'{model_name} Prediction', linewidth=1.5, color=colors[i % len(colors)])

# 样式设置
ax = plt.gca()
ax.xaxis.set_major_locator(FixedLocator(event_ticks))
def _event_label_formatter(x, pos):
    """Formatter for x-axis event labels. Handles cases where ``pos`` may be None
    (e.g., when called from interactive events like mouse move). If pos is None,
    find the nearest tick to x and return its label; otherwise use pos as index.
    """
    try:
        if pos is None:
            # nearest tick index
            ticks = np.array(event_ticks)
            if ticks.size == 0:
                return ''
            idx = int(np.argmin(np.abs(ticks - x)))
            return event_labels[idx] if idx < len(event_labels) else ''
        pos_int = int(pos)
        return event_labels[pos_int] if 0 <= pos_int < len(event_labels) else ''
    except Exception:
        return ''

ax.xaxis.set_major_formatter(plt.FuncFormatter(_event_label_formatter))
plt.setp(ax.xaxis.get_majorticklabels(), fontsize=10, ha='center')
ax.set_xlabel(f'Event {TARGET_EVENT_ID} - Full Time (UTC)', fontsize=12)
ax.set_ylabel('DE4 Flux (1/(cm²·s·sr·MeV))', fontsize=12)
ax.set_yscale('log')
ax.set_title(f'True DE4 Flux (Whole Event) vs Pred DE4 Flux (Peak Phase Only) - All Model Variants', fontsize=14)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, which='major', ls='-', alpha=0.7)
ax.grid(True, which='minor', ls='--', alpha=0.4)
ax.set_xlim(plot_positions.min() - 0.5, plot_positions.max() + 0.5)
plt.tight_layout()

# 保存图像
save_path = os.path.join(SAVE_DIRNAME, f'ablation_all_models_prediction_comparison.png')
plt.savefig(save_path, dpi=150)
print(f"✅ 所有模型预测对比图像已保存至: {save_path}")
plt.show()

# 14.3 生成消融实验总结
print(f"\n{'='*80}")
print("消融实验总结")
print(f"{'='*80}")
print(f"最佳模型: {best_model_name}")
print(f"最佳性能:")
print(f"- MSE: {best_model_performance['mse']:.4f}")
print(f"- RMSE: {best_model_performance['rmse']:.4f}")
print(f"- MAE: {best_model_performance['mae']:.4f}")
print(f"- MAPE: {best_model_performance['mape']:.2f}%")
print(f"- R²: {best_model_performance['r2']:.4f}")
print(f"\n所有模型性能:")
for model_name, performance in ablations_results.items():
    print(f"{model_name}:")
    print(f"  - MSE: {performance['mse']:.4f}")
    print(f"  - RMSE: {performance['rmse']:.4f}")
    print(f"  - MAE: {performance['mae']:.4f}")
    print(f"  - MAPE: {performance['mape']:.2f}%")
    print(f"  - R²: {performance['r2']:.4f}")
print(f"{'='*80}")

print("\n消融实验完成！")
print(f"所有结果和图像已保存至: {SAVE_DIRNAME}")