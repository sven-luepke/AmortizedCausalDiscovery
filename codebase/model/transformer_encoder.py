import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()

        self.covariate_transformer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            batch_first=True
        )

    def forward(self, x):
        # inputs.shape [batch_size, num_atoms, num_timesteps, num_dims]
        B, N, T, D = x.shape

        # covariate transformer
        x = x.permute(0, 2, 1, 3).reshape(-1, N, D)
        x = self.covariate_transformer(x)
        x = x.reshape(B, T, N, D).permute(0, 2, 1, 3)

        # temporal transformer
        x = x.reshape(-1, T, D)
        x = self.temporal_transformer(x)
        x = x.reshape(B, N, T, D)

        return x

import torch
from model.modules import *
from model.Encoder import Encoder

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix with shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape becomes (1, max_len, d_model)
        
        # Register pe as a buffer so it moves with the model's device.
        self.register_buffer("pe", pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerEncoder(Encoder):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    def __init__(self, args, n_in, n_hid, n_out, do_prob=0.0, factor=True):
        super().__init__(args, factor)

        #self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)

        self.init_weights()

        self.in_proj = nn.Linear(n_in, n_hid)
        self.pe = PositionalEncoding(n_hid, dropout=0.1)

        self.cross_transformer_0 = CrossTransformerLayer(d_model=n_hid, nhead=4, dim_feedforward=256)
        self.cross_transformer_1 = CrossTransformerLayer(d_model=n_hid, nhead=4, dim_feedforward=256)


    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]

        x = self.in_proj(inputs)
        B, N, T, D = x.shape
        x = x.reshape(-1, T, D)
        x = self.pe(x)
        x = x.reshape(B, N, T, D)

        x = self.cross_transformer_0(x)
        x = self.cross_transformer_1(x)

        # x.shape = [batch_size, num_atoms, num_timesteps, num_dims]
        x = x.permute(0, 2, 1, 3).reshape(-1, N, D)
        #x = x.mean(dim=2)
        #x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]

        #x = self.mlp1(x)  # 2-layer ELU net per node

        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)

        x = self.fc_out(x)

        # reshape batch sample back to time steps
        x = x.reshape(B, T, x.shape[-2], x.shape[-1])
        return x
