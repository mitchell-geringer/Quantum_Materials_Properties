"""Feed-forward band-gap regressor."""

import torch.nn as nn


def make_network(features, width=256, depth=4, dropout=0.0):
    layers, n = [], features
    for _ in range(int(depth)):
        layers += [nn.Linear(n, width), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        n = width
    layers.append(nn.Linear(n, 1))
    return nn.Sequential(*layers)
