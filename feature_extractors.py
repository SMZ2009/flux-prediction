#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""频域/时域特征提取"""
import numpy as np
from scipy.fft import rfft, rfftfreq

class FourierFeatureExtractor:
    """傅里叶特征提取器：从时间序列提取频域特征"""
    def __init__(self, sampling_rate=1.0, n_fft=128):
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft

    def extract_features(self, signal):
        """提取单段信号的频域特征"""
        # 补零保证长度
        signal_padded = np.pad(signal, (0, max(0, self.n_fft - len(signal))), 'constant')
        yf = rfft(signal_padded, n=self.n_fft)
        xf = rfftfreq(self.n_fft, 1/self.sampling_rate)
        magnitude = np.abs(yf)

        # 核心特征
        features = {
            'dominant_freq': xf[np.argmax(magnitude)],
            'spectral_energy': np.sum(magnitude**2),
            'low_freq_ratio': np.sum(magnitude[xf < 0.1]) / (np.sum(magnitude) + 1e-8),
            'spectral_entropy': self._spectral_entropy(magnitude),
            'peak_freq_ratio': np.max(magnitude) / (np.mean(magnitude) + 1e-8)
        }
        return features

    def _spectral_entropy(self, magnitude):
        """计算归一化频谱熵"""
        p = magnitude / (np.sum(magnitude) + 1e-8)
        entropy = -np.sum(p * np.log2(p + 1e-8))
        max_entropy = np.log2(len(p))
        return entropy / max_entropy

    def batch_extract(self, signals):
        """批量提取特征（返回数组）"""
        return np.array([list(self.extract_features(s).values()) for s in signals])

class TemporalFeatureExtractor:
    """时域特征提取器：从时间序列提取统计特征"""
    def __init__(self, window_size=20):
        self.window_size = window_size

    def extract_features(self, signal):
        """提取单段信号的时域特征"""
        features = {
            'mean': np.mean(signal),
            'std': np.std(signal),
            'skewness': self._skewness(signal),
            'kurtosis': self._kurtosis(signal),
            'max': np.max(signal),
            'min': np.min(signal),
            'range': np.max(signal) - np.min(signal),
            'rms': np.sqrt(np.mean(signal**2)),
            'crest_factor': np.max(np.abs(signal)) / (np.sqrt(np.mean(signal**2)) + 1e-8),
            'slope': self._calculate_slope(signal)
        }
        return features

    def _skewness(self, signal):
        """偏度计算"""
        mean = np.mean(signal)
        std = np.std(signal)
        return 0 if std == 0 else np.mean(((signal - mean) / std) ** 3)

    def _kurtosis(self, signal):
        """峰度计算（减3为正态分布峰度）"""
        mean = np.mean(signal)
        std = np.std(signal)
        return 0 if std == 0 else np.mean(((signal - mean) / std) ** 4) - 3

    def _calculate_slope(self, signal):
        """线性回归计算信号斜率"""
        if len(signal) < 2:
            return 0
        x = np.arange(len(signal))
        slope, _ = np.polyfit(x, signal, 1)
        return slope

    def batch_extract(self, signals):
        """批量提取特征（返回数组）"""
        return np.array([list(self.extract_features(s).values()) for s in signals])