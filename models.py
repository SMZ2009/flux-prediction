#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""注意力增强MLP模型"""
import numpy as np
import torch
import torch.nn as nn

class AttentionEnhancedMLP(nn.Module):
    """注意力增强的MLP：融合傅里叶/时域特征预测DE4/DE1比值"""
    def __init__(self, input_dim, fourier_dim=5, temporal_dim=10, 
                 hidden_dims=[256, 128, 64], num_heads=4, dropout_rate=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.fourier_dim = fourier_dim
        self.temporal_dim = temporal_dim
        self.total_feature_dim = input_dim + fourier_dim + temporal_dim

        # 特征投影层
        self.feature_projection = nn.Sequential(
            nn.Linear(self.total_feature_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_dims[0])
        )

        # 多头注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[0], num_heads=num_heads, dropout=dropout_rate, batch_first=True
        )

        # 注意力后投影
        self.attention_projection = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_dims[1])
        )

        # MLP预测头
        mlp_layers = []
        mlp_input_dims = [hidden_dims[1]] + hidden_dims[2:] if len(hidden_dims)>=2 else [hidden_dims[0]]
        for i in range(len(mlp_input_dims)-1):
            mlp_layers.extend([
                nn.Linear(mlp_input_dims[i], mlp_input_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.LayerNorm(mlp_input_dims[i+1])
            ])
        mlp_layers.append(nn.Linear(mlp_input_dims[-1], 1))
        self.mlp = nn.Sequential(*mlp_layers)

        # 残差连接投影
        self.residual_projection = nn.Linear(self.total_feature_dim, hidden_dims[1]) if self.total_feature_dim != hidden_dims[1] else nn.Identity()

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """Xavier初始化线性层，LayerNorm初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_time, x_fourier, x_temporal):
        """前向传播：输入→特征拼接→投影→注意力→残差→预测"""
        combined = torch.cat([x_time, x_fourier, x_temporal], dim=1)  # [B, total_dim]
        projected = self.feature_projection(combined)                # [B, h0]
        attn_out, _ = self.attention(projected.unsqueeze(1), projected.unsqueeze(1), projected.unsqueeze(1))
        attn_out = attn_out.squeeze(1)                               # [B, h0]
        attn_proj = self.attention_projection(attn_out)              # [B, h1]
        residual = self.residual_projection(combined)                # [B, h1]
        output = self.mlp(attn_proj + residual)                      # [B, 1]
        return output

    def predict_with_uncertainty(self, x_time, x_fourier, x_temporal, num_samples=100):
        """MC Dropout估计预测不确定性"""
        self.train()
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(x_time, x_fourier, x_temporal)
                predictions.append(pred.cpu().numpy())
        self.eval()
        predictions = np.concatenate(predictions, axis=1)
        return np.mean(predictions, axis=1, keepdims=True), np.std(predictions, axis=1, keepdims=True)