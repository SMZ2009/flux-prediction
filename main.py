#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""主程序：示例调用流程"""
import os
import numpy as np
from data_loader import DataLoaderCSV
from prediction_system import FluxPredictionSystem
from utils import setup_chinese_font, calculate_metrics

def main():
    # 1. 配置环境
    setup_chinese_font()
    csv_path = "sample_data.csv"  # 使用示例数据，可替换为你的CSV路径

    # 2. 加载数据
    data_loader = DataLoaderCSV(csv_path)
    time_series, de1_flux, de4_flux = data_loader.load_data()
    time_series, de1_flux, de4_flux = data_loader.preprocess_data()
    data_loader.plot_original_data()

    # 3. 初始化预测系统
    model_params = {'input_dim': 50, 'hidden_dims': [256, 128, 64]}
    predictor = FluxPredictionSystem(model_params=model_params, device='cuda')

    # 4. 准备训练数据
    X_time, X_fourier, X_temporal, y = predictor.prepare_data(
        de1_flux, de4_flux, time_series, window_size=50, stride=1
    )

    # 5. 训练模型
    predictor.train(
        X_time, X_fourier, X_temporal, y,
        batch_size=32, epochs=100, lr=0.001, val_split=0.2, patience=10
    )

    # 6. 预测
    pred_ratio, pred_de4 = predictor.predict(de1_flux, time_series, window_size=50)

    # 7. 评估结果
    true_ratio = de4_flux[50:] / (de1_flux[50:] + 1e-8)
    metrics = calculate_metrics(true_ratio, pred_ratio)
    print("预测指标：", metrics)

    # 8. 可视化结果
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(time_series[50:], true_ratio, 'b-', label='真实DE4/DE1比值')
    plt.plot(time_series[50:], pred_ratio, 'r--', label='预测DE4/DE1比值')
    plt.xlabel('时间（小时）'), plt.ylabel('DE4/DE1比值')
    plt.title('预测结果对比'), plt.legend(), plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()