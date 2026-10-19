import torch
from torch import nn
import torch.nn.functional as F


class Attentive(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.w = nn.Parameter(torch.ones(input_dim))

    def forward(self, x):
        return x * self.w.to(x.device, non_blocking=True)


class ATT_learner(nn.Module):
    def __init__(self, nlayers, input_dim, k):
        super().__init__()
        self.layers = nn.ModuleList([Attentive(input_dim) for _ in range(nlayers)])
        self.k = k

    def forward(self, features):
        embeddings = features
        for index, layer in enumerate(self.layers):
            embeddings = layer(embeddings)
            if index != len(self.layers) - 1:
                embeddings = F.relu(embeddings)
        embeddings = F.normalize(embeddings, dim=1, p=2)
        similarities = torch.mm(embeddings, embeddings.t())
        _, indices = similarities.topk(k=int(self.k + 1), dim=-1)
        mask = torch.zeros_like(similarities)
        rows = torch.arange(similarities.shape[0], device=similarities.device).view(
            -1, 1
        )
        mask[rows, indices] = 1.0
        return F.relu(similarities * mask)
