#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用工具函数"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Matplotlib 中文显示配置 ---
def setup_chinese_font():
    """配置Matplotlib中文显示"""
    try:
        from matplotlib import font_manager
        _chinese_fonts = ['SimHei', 'Microsoft YaHei', 'STSong', 'SimSun', 'KaiTi', 'FangSong']
        _available_names = {f.name for f in font_manager.fontManager.ttflist}
        _chosen = None
        for _name in _chinese_fonts:
            if _name in _available_names:
                _chosen = _name
                break
        if _chosen is None:
            for f in font_manager.fontManager.ttflist:
                lname = f.name.lower()
                if any(k in lname for k in ['hei', 'yahei', 'song', 'sim', 'kai', 'fang']):
                    _chosen = f.name
                    break
        if _chosen:
            plt.rcParams['font.sans-serif'] = [_chosen]
    except Exception:
        pass
    plt.rcParams['axes.unicode_minus'] = False

# --- 评估指标函数 ---
def calculate_metrics(y_true, y_pred):
    """计算回归评估指标"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "mse": mse,
        "mae": mae,
        "rmse": np.sqrt(mse),
        "r2": r2
    }

# --- 随机种子设置 ---
def set_random_seed(seed=42):
    """设置随机种子保证可复现"""
    torch.manual_seed(seed)
    np.random.seed(seed)

# --- 模型保存/加载 ---
def save_model(model, path):
    """保存模型权重"""
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    """加载模型权重"""
    model.load_state_dict(torch.load(path, map_location=device))
    return model