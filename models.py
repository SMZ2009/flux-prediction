import torch
import torch.nn as nn
import numpy as np

class TransformerRegressionModel(nn.Module):

    def __init__(
        self,
        time_dim,
        fourier_dim,
        temporal_dim,
        d_model=128,
        n_heads=4,
        num_layers=3,
        dropout=0.1
    ):
        super().__init__()

        self.d_model = d_model

        # 1. Token embedding
        self.time_embedding = nn.Linear(time_dim, d_model)
        self.fourier_embedding = nn.Linear(fourier_dim, d_model)
        self.temporal_embedding = nn.Linear(temporal_dim, d_model)

        # 2. Positional encoding
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, 3, d_model)
        )

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 4. Regression head
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_time, x_fourier, x_temporal):
        """
        x_*: [batch_size, feature_dim]
        """

        # 1. Embedding → [B, 1, D]
        t = self.time_embedding(x_time).unsqueeze(1)
        f = self.fourier_embedding(x_fourier).unsqueeze(1)
        temp = self.temporal_embedding(x_temporal).unsqueeze(1)

        # 2. 拼接成序列 [B, 3, D]
        tokens = torch.cat([t, f, temp], dim=1)
        tokens = tokens + self.pos_embedding

        # 3. Transformer Encoder
        encoded = self.transformer(tokens)  # [B, 3, D]

        # 4. Mean Pooling
        pooled = encoded.mean(dim=1)        # [B, D]

        # 5. 回归
        output = self.regressor(pooled)     # [B, 1]
        return output

    def predict_with_uncertainty(self, x_time, x_fourier, x_temporal, num_samples=100):
        self.train()
        preds = []
        with torch.no_grad():
            for _ in range(num_samples):
                p = self.forward(x_time, x_fourier, x_temporal)
                preds.append(p.cpu().numpy())
        self.eval()
        preds = np.concatenate(preds, axis=1)
        return preds.mean(axis=1, keepdims=True), preds.std(axis=1, keepdims=True)