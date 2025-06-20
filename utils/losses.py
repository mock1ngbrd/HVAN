import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure inputs are probabilities (after sigmoid for binary classification)
        # inputs = torch.sigmoid(inputs)
        # Compute the focal loss components
        # loss = nn.BCELoss(reduction='none')(inputs, targets)
        loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-loss)  # pt is the model’s estimated probability for the true class
        # loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        modulating_factor = (1.0 - pt) ** self.gamma
        loss *= modulating_factor
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss *= alpha_factor

        # Apply reduction (mean or sum)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MultiClassFocalLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure inputs are probabilities (after sigmoid for binary classification)
        # inputs = torch.sigmoid(inputs)
        # Compute the focal loss components
        # loss = nn.BCELoss(reduction='none')(inputs, targets)
        loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-loss)  # pt is the model’s estimated probability for the true class
        # loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        modulating_factor = (1.0 - pt) ** self.gamma
        loss *= modulating_factor
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss *= alpha_factor

        # Apply reduction (mean or sum)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def pdist_squared(x):
    xx = (x ** 2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, 255.0)
    return dist
