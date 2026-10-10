"""PyTorch modules for the deep learners (module level so that fitted models can be pickled)."""
from __future__ import annotations

import rtdl_revisiting_models as rtdl
import torch
from torch import nn


class EmbeddingNet(nn.Module):
    """Numeric features concatenated with flattened categorical embeddings, followed by a backbone."""

    def __init__(self, cardinalities: list[int], n_num: int, d_embedding: int, backbone: nn.Module):
        super().__init__()
        self.emb = rtdl.CategoricalEmbeddings(cardinalities, d_embedding) if cardinalities else None
        self.backbone = backbone

    def forward(self, x_num, x_cat):
        parts = [x_num] + ([self.emb(x_cat).flatten(1)] if self.emb is not None else [])
        return self.backbone(torch.cat(parts, dim=1))


class CNNBackbone(nn.Module):
    def __init__(self, d_in: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Unflatten(1, (1, d_in)), nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(),
                                 nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2), nn.Flatten(),
                                 nn.Dropout(dropout), nn.Linear(64 * (d_in // 2), 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)


class LSTMBackbone(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, batch_first=True)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        _, (h, _) = self.lstm(x.unsqueeze(-1))
        return self.out(h[-1])


class FTTNet(nn.Module):
    def __init__(self, n_num: int, cardinalities: list[int], **kwargs):
        super().__init__()
        self.ftt = rtdl.FTTransformer(n_cont_features=n_num, cat_cardinalities=cardinalities, d_out=1, **kwargs)

    def forward(self, x_num, x_cat):
        return self.ftt(x_num if x_num.shape[1] else None, x_cat if x_cat.shape[1] else None)

    def make_parameter_groups(self):
        return self.ftt.make_parameter_groups()
