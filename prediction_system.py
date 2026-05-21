#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通量预测系统：整合所有组件完成训练/预测"""
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from event_segmentation import EventSplitter
from feature_extractors import FourierFeatureExtractor, TemporalFeatureExtractor
from models import AttentionEnhancedMLP
from utils import save_model, load_model, set_random_seed

set_random_seed()

class FluxPredictionSystem:
    """空间粒子通量预测系统"""
    def __init__(self, model_params=None, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_params = model_params or {
            'input_dim': 50, 'fourier_dim':5, 'temporal_dim':10,
            'hidden_dims':[256,128,64], 'num_heads':4, 'dropout_rate':0.2
        }

        # 初始化组件
        self.event_splitter = EventSplitter()
        self.fourier_extractor = FourierFeatureExtractor()
        self.temporal_extractor = TemporalFeatureExtractor()
        self.model = AttentionEnhancedMLP(**self.model_params).to(self.device)

        # 标准化器
        self.scalers = {
            'time': StandardScaler(),
            'fourier': StandardScaler(),
            'temporal': StandardScaler(),
            'target': StandardScaler()
        }

        # 训练历史
        self.training_history = {
            'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []
        }

    def prepare_data(self, de1_flux, de4_flux, time_series=None, window_size=50, stride=1):
        """准备训练数据：滑动窗口+事件分割+特征提取"""
        time_series = time_series or np.arange(len(de1_flux))
        ratios = de4_flux / (de1_flux + 1e-8)  # 目标比值（防除零）
        segments = self.event_splitter.split_events(time_series, de1_flux)

        # 滑动窗口提取特征
        X_time, X_fourier, X_temporal, y = [], [], [], []
        for start, end in segments:
            if end - start < window_size:
                continue
            for i in range(start, end - window_size, stride):
                window = de1_flux[i:i+window_size]
                X_time.append(window)
                X_fourier.append(list(self.fourier_extractor.extract_features(window).values()))
                X_temporal.append(list(self.temporal_extractor.extract_features(window).values()))
                y.append(ratios[i+window_size])

        return (
            np.array(X_time), np.array(X_fourier),
            np.array(X_temporal), np.array(y).reshape(-1,1)
        )

    def train(self, X_time, X_fourier, X_temporal, y,
              batch_size=32, epochs=100, lr=0.001, val_split=0.2, patience=10):
        """训练模型（带早停+学习率调度）"""
        # 标准化
        X_time_scaled = self.scalers['time'].fit_transform(X_time)
        X_fourier_scaled = self.scalers['fourier'].fit_transform(X_fourier)
        X_temporal_scaled = self.scalers['temporal'].fit_transform(X_temporal)
        y_scaled = self.scalers['target'].fit_transform(y)

        # 分割训练/验证集
        splits = train_test_split(
            X_time_scaled, X_fourier_scaled, X_temporal_scaled, y_scaled,
            test_size=val_split, random_state=42
        )
        X_time_train, X_time_val, X_f_train, X_f_val, X_t_train, X_t_val, y_train, y_val = splits

        # 数据加载器
        train_loader = DataLoader(
            TensorDataset(*[torch.FloatTensor(x) for x in [X_time_train, X_f_train, X_t_train, y_train]]),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(*[torch.FloatTensor(x) for x in [X_time_val, X_f_val, X_t_val, y_val]]),
            batch_size=batch_size, shuffle=False
        )

        # 优化器/损失/调度器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )

        # 早停
        best_val_loss = float('inf')
        patience_counter = 0

        print(f"训练开始 | 训练集:{len(X_time_train)} | 验证集:{len(X_time_val)}")
        for epoch in range(epochs):
            # 训练轮
            self.model.train()
            train_loss, train_mae = 0.0, 0.0
            for batch in train_loader:
                batch_time, batch_f, batch_t, batch_y = [x.to(self.device) for x in batch]
                optimizer.zero_grad()
                outputs = self.model(batch_time, batch_f, batch_t)
                loss = criterion(outputs, batch_y)
                # 物理约束：比值≤1
                physical_loss = torch.mean(torch.relu(outputs - 1.0)) * 0.1
                total_loss = loss + physical_loss
                total_loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_mae += torch.mean(torch.abs(outputs - batch_y)).item()

            # 验证轮
            self.model.eval()
            val_loss, val_mae = 0.0, 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch_time, batch_f, batch_t, batch_y = [x.to(self.device) for x in batch]
                    outputs = self.model(batch_time, batch_f, batch_t)
                    val_loss += criterion(outputs, batch_y).item()
                    val_mae += torch.mean(torch.abs(outputs - batch_y)).item()

            # 平均损失
            train_loss /= len(train_loader)
            train_mae /= len(train_loader)
            val_loss /= len(val_loader)
            val_mae /= len(val_loader)

            # 更新历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_mae'].append(train_mae)
            self.training_history['val_mae'].append(val_mae)

            # 学习率调度
            scheduler.step(val_loss)

            # 打印进度
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss:{train_loss:.6f} | Val Loss:{val_loss:.6f} | Train MAE:{train_mae:.6f} | Val MAE:{val_mae:.6f}")

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_model(self.model, 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停触发于Epoch {epoch+1}")
                    break

        # 加载最佳模型
        if 'best_model.pth' in os.listdir():
            self.model = load_model(self.model, 'best_model.pth', self.device)
        print("训练完成！")

    def predict(self, de1_flux, time_series=None, window_size=50):
        """预测DE4/DE1比值及DE4通量"""
        time_series = time_series or np.arange(len(de1_flux))
        # 准备输入特征
        X_time, X_fourier, X_temporal = [], [], []
        for i in range(0, len(de1_flux) - window_size + 1):
            window = de1_flux[i:i+window_size]
            X_time.append(window)
            X_fourier.append(list(self.fourier_extractor.extract_features(window).values()))
            X_temporal.append(list(self.temporal_extractor.extract_features(window).values()))

        if not X_time:
            return np.array([]), np.array([])

        # 标准化
        X_time_scaled = self.scalers['time'].transform(np.array(X_time))
        X_fourier_scaled = self.scalers['fourier'].transform(np.array(X_fourier))
        X_temporal_scaled = self.scalers['temporal'].transform(np.array(X_temporal))

        # 预测
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(
                torch.FloatTensor(X_time_scaled).to(self.device),
                torch.FloatTensor(X_fourier_scaled).to(self.device),
                torch.FloatTensor(X_temporal_scaled).to(self.device)
            ).cpu().numpy()

        # 反标准化
        pred_ratio = self.scalers['target'].inverse_transform(pred_scaled)
        # 还原DE4通量
        pred_de4 = pred_ratio.flatten() * de1_flux[window_size:]

        return pred_ratio.flatten(), pred_de4