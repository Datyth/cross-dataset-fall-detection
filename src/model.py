"""Single-stream CTR-GCN adapted to COCO-17 and one person.

Adapted from Yuxin Chen et al., official CTR-GCN implementation:
https://github.com/Uason-Chen/CTR-GCN/blob/main/model/ctrgcn.py
Original repository is CC BY-NC 4.0; changes: COCO graph, one-person input,
single binary logit, and accessible pre-pooling features.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

COCO_JOINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
               "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
               "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
COCO_EDGES = ((0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
              (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
              (11, 13), (13, 15), (12, 14), (14, 16))


def coco_adjacency():
    a = torch.zeros(3, 17, 17)
    a[0] = torch.eye(17)
    for u, v in COCO_EDGES:
        a[1, u, v] = 1
        a[2, v, u] = 1
    a[1:] /= a[1:].sum(dim=1, keepdim=True).clamp_min(1)
    return a


class CTRGC(nn.Module):
    """Channel-wise topology refinement in the published CTR-GCN."""
    def __init__(self, incoming, outgoing):
        super().__init__()
        rel = 8 if incoming == 3 else max(1, incoming // 8)
        self.conv1 = nn.Conv2d(incoming, rel, 1)
        self.conv2 = nn.Conv2d(incoming, rel, 1)
        self.conv3 = nn.Conv2d(incoming, outgoing, 1)
        self.conv4 = nn.Conv2d(rel, outgoing, 1)

    def forward(self, x, adjacency, alpha):
        x1 = self.conv1(x).mean(dim=2)
        x2 = self.conv2(x).mean(dim=2)
        topology = torch.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        topology = self.conv4(topology) * alpha + adjacency[None, None]
        return torch.einsum("ncuv,nctv->nctu", topology, self.conv3(x))


class GraphUnit(nn.Module):
    def __init__(self, incoming, outgoing, adjacency):
        super().__init__()
        self.adjacency = nn.Parameter(adjacency.clone())
        self.alpha = nn.Parameter(torch.zeros(1))
        self.convs = nn.ModuleList([CTRGC(incoming, outgoing) for _ in range(3)])
        self.bn = nn.BatchNorm2d(outgoing)
        nn.init.constant_(self.bn.weight, 1e-6)
        self.residual = nn.Identity() if incoming == outgoing else nn.Sequential(nn.Conv2d(incoming, outgoing, 1), nn.BatchNorm2d(outgoing))

    def forward(self, x):
        y = sum(conv(x, self.adjacency[i], self.alpha) for i, conv in enumerate(self.convs))
        return F.relu(self.bn(y) + self.residual(x), inplace=True)


class TemporalConv(nn.Module):
    def __init__(self, incoming, outgoing, kernel=5, stride=1, dilation=1):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.conv = nn.Conv2d(incoming, outgoing, (kernel, 1), stride=(stride, 1), padding=(pad, 0), dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(outgoing)

    def forward(self, x):
        return self.bn(self.conv(x))


class MultiScaleTemporal(nn.Module):
    def __init__(self, channels, stride):
        super().__init__()
        assert channels % 4 == 0
        branch = channels // 4
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, branch, 1), nn.BatchNorm2d(branch), nn.ReLU(True), TemporalConv(branch, branch, 5, stride, dilation))
            for dilation in (1, 2)
        ])
        self.branches.append(nn.Sequential(nn.Conv2d(channels, branch, 1), nn.BatchNorm2d(branch), nn.ReLU(True),
                                           nn.MaxPool2d((3, 1), stride=(stride, 1), padding=(1, 0)), nn.BatchNorm2d(branch)))
        self.branches.append(nn.Sequential(nn.Conv2d(channels, branch, 1, stride=(stride, 1)), nn.BatchNorm2d(branch)))

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class Block(nn.Module):
    def __init__(self, incoming, outgoing, adjacency, stride=1, residual=True):
        super().__init__()
        self.graph = GraphUnit(incoming, outgoing, adjacency)
        self.temporal = MultiScaleTemporal(outgoing, stride)
        if not residual:
            self.residual = None
        elif incoming == outgoing and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = TemporalConv(incoming, outgoing, 1, stride)

    def forward(self, x):
        res = 0 if self.residual is None else self.residual(x)
        return F.relu(self.temporal(self.graph(x)) + res, inplace=True)


class CTRGCN(nn.Module):
    def __init__(self, base_channels=64, dropout=0.0, classifier_init_std=0.01):
        super().__init__()
        a = coco_adjacency()
        self.data_bn = nn.BatchNorm1d(17 * 3)
        widths = (base_channels,) * 4 + (base_channels * 2,) * 3 + (base_channels * 4,) * 3
        incoming = 3
        blocks = []
        for i, outgoing in enumerate(widths):
            blocks.append(Block(incoming, outgoing, a, stride=2 if i in (4, 7) else 1, residual=i != 0))
            incoming = outgoing
        self.blocks = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.fc = nn.Linear(incoming, 1)
        # The original multi-class initialization becomes very large for one
        # BCE logit. Start the binary head close to an uninformative predictor.
        if classifier_init_std <= 0:
            raise ValueError("classifier_init_std must be positive")
        nn.init.normal_(self.fc.weight, 0, classifier_init_std)
        nn.init.zeros_(self.fc.bias)

    def features(self, x):
        if x.ndim != 5 or x.shape[1] != 3 or x.shape[3:] != (17, 1):
            raise ValueError("CTRGCN input must be [N,3,T,17,1]")
        n, c, t, v, m = x.shape
        x = x.permute(0, 4, 3, 1, 2).reshape(n, m * v * c, t)
        x = self.data_bn(x)
        x = x.reshape(n, m, v, c, t).permute(0, 1, 3, 4, 2).reshape(n * m, c, t, v)
        return self.blocks(x)

    def forward(self, x):
        return self.fc(self.dropout(self.features(x).mean(dim=(2, 3)))).squeeze(-1)


def mil_logit(window_logits):
    return window_logits.topk(min(2, window_logits.numel())).values.mean()
