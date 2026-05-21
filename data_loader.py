#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CSV数据加载与预处理"""
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy import interpolate

class DataLoaderCSV:
    """CSV数据加载器：加载/预处理空间粒子通量CSV数据"""
    def __init__(self, csv_path=None):
        self.csv_path = csv_path
        self.data = None
        self.time_series = None
        self.de1_flux = None
        self.de4_flux = None

    def load_data(self, csv_path=None):
        """加载CSV数据，自动识别时间/DE1/DE4列"""
        if csv_path is not None:
            self.csv_path = csv_path
        if self.csv_path is None:
            raise ValueError("请提供CSV文件路径")

        # 检查文件是否存在
        import os
        if not os.path.exists(self.csv_path):
            raise ValueError(f"CSV文件不存在: {self.csv_path}\n请将CSV文件放在正确的位置，或在main.py中修改csv_path变量")

        print(f"正在加载数据: {self.csv_path}")
        # 多编码尝试读取
        df = self._read_csv_with_multiple_encodings()
        print(f"数据形状: {df.shape} | 列名: {df.columns.tolist()}")

        # 自动识别列
        time_col = self._auto_detect_column(df, ['epoch', 'time', 'date'])
        de1_col = self._auto_detect_column(df, ['de1', '.038'])
        de4_col = self._auto_detect_column(df, ['de4', '.175'])

        # 兜底逻辑
        if de1_col is None and len(df.columns) >= 2:
            de1_col = df.columns[0] if 'epoch' not in df.columns[0].lower() else df.columns[1]
        if de4_col is None and len(df.columns) >= 3:
            cols = [c for c in df.columns if c != time_col and c != de1_col]
            de4_col = cols[0] if cols else None

        if de1_col is None or de4_col is None:
            raise ValueError("无法识别DE1和DE4数据列，请检查CSV文件格式")

        # 解析时间序列
        self.time_series = self._parse_time_series(df, time_col)
        # 提取通量数据
        self.de1_flux = df[de1_col].values.astype(np.float32)
        self.de4_flux = df[de4_col].values.astype(np.float32)

        print(f"加载完成: {len(self.time_series)} 个数据点")
        print(f"DE1通量范围: [{self.de1_flux.min():.2f}, {self.de1_flux.max():.2f}]")
        print(f"DE4通量范围: [{self.de4_flux.min():.2f}, {self.de4_flux.max():.2f}]")
        return self.time_series, self.de1_flux, self.de4_flux

    def _read_csv_with_multiple_encodings(self):
        """多编码尝试读取CSV"""
        encodings = ['utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(self.csv_path, encoding=enc)
            except Exception:
                continue
        raise ValueError(f"无法读取CSV文件: {self.csv_path}")

    def _auto_detect_column(self, df, keywords):
        """根据关键词自动检测列"""
        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in keywords):
                return col
        return None

    def _parse_time_series(self, df, time_col):
        """解析时间列为小时数序列"""
        if not time_col:
            return np.arange(len(df))
        try:
            df[time_col] = pd.to_datetime(df[time_col])
            start_time = df[time_col].min()
            return (df[time_col] - start_time).dt.total_seconds() / 3600.0
        except Exception:
            print("时间列解析失败，使用索引作为时间序列")
            return np.arange(len(df))

    def preprocess_data(self, remove_outliers=True, outlier_threshold=3.0):
        """预处理数据：填充缺失值、移除异常值"""
        de1_clean = self._clean_series(self.de1_flux, remove_outliers, outlier_threshold)
        de4_clean = self._clean_series(self.de4_flux, remove_outliers, outlier_threshold)
        return self.time_series, de1_clean, de4_clean

    def _clean_series(self, series, remove_outliers, threshold):
        """单序列清洗：插值填充NaN + 异常值处理"""
        series_clean = series.copy()

        # 插值填充NaN
        if np.any(np.isnan(series_clean)):
            valid_idx = np.where(~np.isnan(series_clean))[0]
            if len(valid_idx) > 1:
                interp_func = interpolate.interp1d(valid_idx, series_clean[valid_idx],
                                                   kind='linear', fill_value='extrapolate')
                series_clean = interp_func(np.arange(len(series_clean)))
            else:
                series_clean = np.nan_to_num(series_clean)

        # 异常值处理（中值滤波替换）
        if remove_outliers:
            mean = np.mean(series_clean)
            std = np.std(series_clean)
            outliers = np.abs(series_clean - mean) > threshold * std
            if np.any(outliers):
                series_clean[outliers] = median_filter(series_clean, size=5, mode='nearest')[outliers]
        return series_clean

    def plot_original_data(self, title="原始数据可视化"):
        """可视化原始数据"""
        import matplotlib.pyplot as plt
        from utils import setup_chinese_font
        setup_chinese_font()

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        axes[0].plot(self.time_series, self.de1_flux, 'b-', linewidth=1, alpha=0.7)
        axes[0].set_ylabel('DE1通量'), axes[0].set_title(f'{title} - DE1通量'), axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.time_series, self.de4_flux, 'r-', linewidth=1, alpha=0.7)
        axes[1].set_ylabel('DE4通量'), axes[1].set_title(f'{title} - DE4通量'), axes[1].grid(True, alpha=0.3)
        
        ratios = self.de4_flux / (self.de1_flux + 1e-8)
        axes[2].plot(self.time_series, ratios, 'g-', linewidth=1, alpha=0.7)
        axes[2].set_xlabel('时间（小时）'), axes[2].set_ylabel('DE4/DE1比值'), axes[2].set_title(f'{title} - DE4/DE1比值'), axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout(), plt.show()