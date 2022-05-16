import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super().__init__()

        # Compute the positional encodings once in log space.
        # d_model = embed_size
        # (len, dim)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # 得到当前的batch size大小
        batch_size = x.size(0)
        # (1, len, dim) => (batch size, len, dim)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)
