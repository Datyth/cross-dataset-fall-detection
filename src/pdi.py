"""Predictive dynamics consistency from frozen skeleton windows."""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Predictor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, channels))

    def forward(self, x):
        return self.net(x)


def pair_candidates(windows, masks, min_past=36, min_future=12):
    """Return observed 48+16 pairs; confidence-zero frames are missing poses."""
    indices = []
    for i, (x, observed) in enumerate(zip(windows, masks)):
        frame_valid = (x[2, :, :, 0] > 0).any(axis=1) & observed
        if observed.all() and frame_valid[:48].sum() >= min_past and frame_valid[48:].sum() >= min_future:
            indices.append(i)
    return indices


def augment_past(x, rng, noise_std=.01, drop_probability=.1):
    result = x.copy()
    valid = result[2, :, :, 0] > 0
    noise = rng.normal(0, noise_std, size=result[:2, :, :, 0].shape)
    result[:2, :, :, 0] += (noise * valid[None]).astype(np.float32)
    joint_mask = rng.random(result.shape[2]) < drop_probability
    if joint_mask.all():
        joint_mask[int(rng.integers(len(joint_mask)))] = False
    result[:, :, joint_mask, 0] = 0
    return result


def embedding(model, x):
    return model.features(x).mean(dim=(2, 3))


def cosine_distance(a, b):
    return (2 - 2 * F.cosine_similarity(a, b, dim=-1)).mean()


def losses(online, target, predictor, past, augmented, future, use_pdi):
    z_p = embedding(online, past)
    z_aug = embedding(online, augmented)
    with torch.no_grad():
        z_f = embedding(target, future)
    y_p, y_aug = predictor(z_p), predictor(z_aug)
    prediction = .5 * (cosine_distance(y_p, z_f) + cosine_distance(y_aug, z_f))
    consistency = cosine_distance(y_aug, y_p.detach()) if use_pdi else prediction.new_zeros(())
    return prediction, consistency, y_p.detach()


@torch.no_grad()
def update_ema(target, online, decay):
    for previous, current in zip(target.parameters(), online.parameters()):
        previous.lerp_(current, 1 - decay)
    for previous, current in zip(target.buffers(), online.buffers()):
        if previous.is_floating_point():
            previous.lerp_(current, 1 - decay)
        else:
            previous.copy_(current)


def make_target(online):
    target = copy.deepcopy(online).eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    return target
