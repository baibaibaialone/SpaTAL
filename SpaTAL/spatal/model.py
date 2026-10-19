import math

import torch
from torch import nn
from torch.nn import ReLU, Sequential
from torch_geometric.nn import Linear
import torch.nn.functional as F

from .attention import GraphAttentionLayerV2


class GraphConvolution2(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features):
        super(GraphConvolution2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output + self.bias


class SpaTALModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.gc2 = GraphConvolution2(256 * 3, 200)

        self.proj_head = Sequential(
            Linear(200, 512), ReLU(inplace=True), Linear(512, 200)
        )

        nheads = 3
        alpha = 0.01
        self.attentions = [
            GraphAttentionLayerV2(200, 256, 1, dropout=0, alpha=alpha, concat=True)
            for _ in range(nheads)
        ]

        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)

    def forward(self, x, adj):
        x = x.unsqueeze(0)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = x.squeeze(0)

        z = F.relu(self.gc2(x, adj))

        null_count = torch.isnan(z).sum().item()

        if null_count > 0:
            print(f"{null_count}个null")
            # 将 NaN 替换为 0，保持设备和 dtype 不变
            z = torch.nan_to_num(z, nan=0.0)

        embedding = self.proj_head(z)

        return embedding, z

    @staticmethod
    def calc_loss(x, x_aug, temperature=0.1):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum("ik,jk->ij", x, x_aug) / torch.einsum(
            "i,j->ij", x_abs, x_aug_abs
        )
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss_0 = -torch.log(loss_0).mean()
        loss_1 = -torch.log(loss_1).mean()
        return (loss_0 + loss_1) / 2.0
