#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基于流量变化的动态事件分割"""
import numpy as np
import matplotlib.pyplot as plt
from utils import setup_chinese_font

class EventSplitter:
    """事件分割器：检测DE1通量突变点，分割时间序列为事件段"""
    def __init__(self, change_threshold=0.3, min_segment_length=10, window_size=5):
        self.change_threshold = change_threshold
        self.min_segment_length = min_segment_length
        self.window_size = window_size

    def detect_mutation_points(self, de1_flux):
        """检测流量突变点（合并相邻突变点）"""
        # 计算滑动窗口变化率
        if len(de1_flux) <= self.window_size:
            return []
        
        changes = []
        for i in range(len(de1_flux) - self.window_size):
            window = de1_flux[i:i+self.window_size]
            change_rate = np.abs((window[-1] - window[0]) / (np.mean(window) + 1e-8))
            changes.append(change_rate)
        
        # 筛选超过阈值的点并合并
        changes = np.array(changes)
        mutation_indices = np.where(changes > self.change_threshold)[0]
        merged_indices = []
        if len(mutation_indices) > 0:
            current = mutation_indices[0]
            group = [current]
            for idx in mutation_indices[1:]:
                if idx - current <= self.window_size:
                    group.append(idx)
                else:
                    merged_indices.append(int(np.mean(group)))
                    group = [idx]
                current = idx
            merged_indices.append(int(np.mean(group)))
        return merged_indices

    def split_events(self, time_series, de1_flux):
        """按突变点分割为事件段"""
        mutation_points = self.detect_mutation_points(de1_flux)
        segments = []
        start_idx = 0

        for mp in mutation_points:
            if mp - start_idx >= self.min_segment_length:
                segments.append((start_idx, mp))
                start_idx = mp

        # 补充最后一段
        if len(de1_flux) - start_idx >= self.min_segment_length:
            segments.append((start_idx, len(de1_flux)))
        
        # 无突变点时返回整段
        if not segments:
            segments = [(0, len(de1_flux))]
        return segments

    def visualize_segmentation(self, time_series, de1_flux, segments):
        """可视化事件分割结果"""
        setup_chinese_font()
        plt.figure(figsize=(12, 6))
        plt.plot(time_series, de1_flux, 'b-', label='DE1通量', alpha=0.7)
        
        colors = ['r', 'g', 'm', 'c', 'y']
        for i, (start, end) in enumerate(segments):
            color = colors[i % len(colors)]
            plt.axvspan(time_series[start], time_series[end-1], 
                       alpha=0.2, color=color, label=f'事件段{i+1}')
        
        # 标记突变点
        mutation_points = self.detect_mutation_points(de1_flux)
        if mutation_points:
            plt.scatter(time_series[mutation_points], de1_flux[mutation_points], 
                       color='red', s=50, zorder=5, label='突变点')
        
        plt.xlabel('时间（小时）'), plt.ylabel('DE1通量'), plt.title('动态事件分割结果')
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1)), plt.grid(True, alpha=0.3)
        plt.tight_layout(), plt.show()