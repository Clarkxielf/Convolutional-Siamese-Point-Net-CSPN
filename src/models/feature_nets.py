"""Feature Extraction and Parameter Prediction networks
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from CSPN.src.models.pointnet_util import sample_and_group_multi

_raw_features_sizes = {'xyz': 3, 'dxyz': 3, 'ppf': 4}
_raw_features_order = {'xyz': 0, 'dxyz': 1, 'ppf': 2}


class feat_ParameterPredictionNet(nn.Module):
    def __init__(self):
        """PointNet based Parameter prediction network

        Args:
            weights_dim: Number of weights to predict (excluding beta), should be something like
                         [3], or [64, 3], for 3 types of features
        """

        super().__init__()

        self._logger = logging.getLogger(self.__class__.__name__)

        # Pointnet
        self.prepool = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),

            nn.Conv1d(128, 1024, 1),
            nn.GroupNorm(16, 1024),
            nn.ReLU(),
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.postpool = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.GroupNorm(16, 256),
            nn.ReLU(),

            nn.Linear(256, 2),
        )

    def forward(self, x):
        """ Returns alpha, beta, and gating_weights (if needed)

        Args:
            x: List containing two point clouds, x[0] = src (B, J, 3), x[1] = ref (B, K, 3)

        Returns:
            beta, alpha, weightings
        """

        src_padded = F.pad(x[0], (0, 1), mode='constant', value=0)
        ref_padded = F.pad(x[1], (0, 1), mode='constant', value=1)
        concatenated = torch.cat([src_padded, ref_padded], dim=1)
        # concatenated = torch.cat([x[0], x[1]], dim=1)

        prepool_feat = self.prepool(concatenated.permute(0, 2, 1))
        pooled = torch.flatten(self.pooling(prepool_feat), start_dim=-2)
        raw_weights = self.postpool(pooled)

        beta = F.softplus(raw_weights[:, 0])
        alpha = F.softplus(raw_weights[:, 1])

        return beta, alpha

class coordinate_ParameterPredictionNet(nn.Module):
    def __init__(self):
        """PointNet based Parameter prediction network

        Args:
            weights_dim: Number of weights to predict (excluding beta), should be something like
                         [3], or [64, 3], for 3 types of features
        """

        super().__init__()

        self._logger = logging.getLogger(self.__class__.__name__)

        # Pointnet
        self.prepool = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),

            nn.Conv1d(128, 1024, 1),
            nn.GroupNorm(16, 1024),
            nn.ReLU(),
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.postpool = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.GroupNorm(16, 256),
            nn.ReLU(),

            nn.Linear(256, 2),
        )

    def forward(self, x):
        """ Returns alpha, beta, and gating_weights (if needed)

        Args:
            x: List containing two point clouds, x[0] = src (B, J, 3), x[1] = ref (B, K, 3)

        Returns:
            beta, alpha, weightings
        """

        src_padded = F.pad(x[0], (0, 1), mode='constant', value=0)
        ref_padded = F.pad(x[1], (0, 1), mode='constant', value=1)
        concatenated = torch.cat([src_padded, ref_padded], dim=1)
        # concatenated = torch.cat([x[0], x[1]], dim=1)

        prepool_feat = self.prepool(concatenated.permute(0, 2, 1))
        pooled = torch.flatten(self.pooling(prepool_feat), start_dim=-2)
        raw_weights = self.postpool(pooled)

        beta = F.softplus(raw_weights[:, 0])
        alpha = F.softplus(raw_weights[:, 1])

        return beta, alpha


class src_DGCNN(nn.Module):
    def __init__(self, feature_dim=96, num_neighbors=16):
        super(src_DGCNN, self).__init__()

        self.num_neighbors = num_neighbors

        self.conv1 = nn.Conv2d(4, 8, kernel_size=1, bias=False)
        self.mlp1 = nn.Conv2d(num_neighbors, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=1, bias=False)
        self.mlp2 = nn.Conv2d(num_neighbors, 1, kernel_size=1)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=1, bias=False)
        self.mlp3 = nn.Conv2d(num_neighbors, 1, kernel_size=1)
        self.conv4 = nn.Conv2d(16, 64, kernel_size=1, bias=False)
        self.mlp4 = nn.Conv2d(num_neighbors, 1, kernel_size=1)
        self.conv5 = nn.Conv2d(96, feature_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(feature_dim)

    def forward(self, x):
        x = x.transpose(-1, -2)   # (8, 2, 32)
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x, k=self.num_neighbors)   # (8, 4, 32, 16)
        x = F.relu(self.bn1(self.conv1(x)))   # (8, 8, 32, 16)
        x1 = x.transpose(3, 1)   # (8, 16, 32, 8)
        x1 = self.mlp1(x1)   # (8, 1, 32, 8)

        x = F.relu(self.bn2(self.conv2(x)))   # (8, 8, 32, 16)
        x2 = x.transpose(3, 1)   # (8, 16, 32, 8)
        x2 = self.mlp2(x2)   # (8, 1, 32, 8)

        x = F.relu(self.bn3(self.conv3(x)))   # (8, 16, 32, 16)
        x3 = x.transpose(3, 1)   # (8, 16, 32, 16)
        x3 = self.mlp3(x3)   # (8, 1, 32, 16)

        x = F.relu(self.bn4(self.conv4(x)))   # (8, 64, 32, 16)
        x4 = x.transpose(3, 1)   # (8, 16, 32, 64)
        x4 = self.mlp4(x4)   # (8, 1, 32, 64)

        x = torch.cat((x1, x2, x3, x4), dim=-1)   # (8, 1, 32, 96)
        x = x.transpose(3, 1)   # (8, 96, 32, 1)

        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)   # (8, 96, 32)

        x = x.transpose(-2, -1)   # (8, 32, 96)
        return x/torch.norm(x, dim=-1, keepdim=True)


class ref_DGCNN(nn.Module):
    def __init__(self, feature_dim=96, num_neighbors=32):
        super(ref_DGCNN, self).__init__()

        self.num_neighbors = num_neighbors

        self.conv1 = nn.Conv2d(4, 8, kernel_size=1, bias=False)
        self.mlp1 = nn.Conv2d(num_neighbors, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=1, bias=False)
        self.mlp2 = nn.Conv2d(num_neighbors, 1, kernel_size=1)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=1, bias=False)
        self.mlp3 = nn.Conv2d(num_neighbors, 1, kernel_size=1)
        self.conv4 = nn.Conv2d(16, 64, kernel_size=1, bias=False)
        self.mlp4 = nn.Conv2d(num_neighbors, 1, kernel_size=1)
        self.conv5 = nn.Conv2d(96, feature_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(feature_dim)

    def forward(self, x):
        x = x.transpose(-1, -2)  # (8, 2, 64)
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x, k=self.num_neighbors)  # (8, 4, 64, 32)
        x = F.relu(self.bn1(self.conv1(x)))  # (8, 8, 64, 32)
        x1 = x.transpose(3, 1)  # (8, 32, 64, 8)
        x1 = self.mlp1(x1)  # (8, 1, 64, 8)

        x = F.relu(self.bn2(self.conv2(x)))  # (8, 8, 64, 32)
        x2 = x.transpose(3, 1)  # (8, 32, 64, 8)
        x2 = self.mlp2(x2)  # (8, 1, 64, 8)

        x = F.relu(self.bn3(self.conv3(x)))  # (8, 16, 64, 32)
        x3 = x.transpose(3, 1)  # (8, 32, 64, 16)
        x3 = self.mlp3(x3)  # (8, 1, 64, 16)

        x = F.relu(self.bn4(self.conv4(x)))  # (8, 64, 64, 32)
        x4 = x.transpose(3, 1)  # (8, 32, 64, 64)
        x4 = self.mlp4(x4)  # (8, 1, 64, 64)

        x = torch.cat((x1, x2, x3, x4), dim=-1)  # (8, 1, 64, 96)
        x = x.transpose(3, 1)  # (8, 96, 64, 1)

        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)  # (8, 96, 64)

        x = x.transpose(-2, -1)  # (8, 64, 96)
        return x / torch.norm(x, dim=-1, keepdim=True)


def get_graph_feature(x, k=20):

    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)

    return feature


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -(xx + inner + xx.transpose(2, 1).contiguous())

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx