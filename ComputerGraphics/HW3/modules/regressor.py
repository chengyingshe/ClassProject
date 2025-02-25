from modules.geometry import rot6d_to_rotmat
from modules.st_gcn import STGCN
from modules.se_module import SELayer
from pointnet2_ops.pointnet2_modules import PointnetSAModule
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PointNet2Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[0, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024], use_xyz=True
            )
        )

    def _break_up_pc(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(
            1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, data):
        x = data['human_points']  # (B, T, N, 3)
        # B, T, N, _ = x.shape
        B, T, _, N, _ = x.shape  # [8, 16, 1, 512, 3]
        x = x.reshape(-1, N, 3)  # (B * T, N, 3)
        xyz, features = self._break_up_pc(x)
        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        features = features.squeeze(-1).reshape(B, T, -1)
        return features


class RNN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(n_hidden, n_hidden, n_rnn_layer,
                          batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(n_input, n_hidden)

        self.linear2 = nn.Linear(n_hidden * 2, n_output)

        self.dropout = nn.Dropout()

    def forward(self, x):  # (B, T, D)
        x = self.rnn(F.relu(self.dropout(self.linear1(x)), inplace=True))[0]
        return self.linear2(x)


class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PointNet2Encoder()
        self.pose_s1 = RNN(1024, 24 * 3, 1024)
        self.pose_s2 = STGCN(3 + 1024)

    def forward(self, data):

        pred = {}

        x = self.encoder(data)  # (B, T, D)
        B, T, _ = x.shape
        full_joints = self.pose_s1(x)  # (B, T, 24 * 3)

        # rot6ds = self.pose_s2(torch.cat((full_joints, x), dim=-1))
        rot6ds = self.pose_s2(torch.cat((full_joints.reshape(
            B, T, 24, 3), x.unsqueeze(-2).repeat(1, 1, 24, 1)), dim=-1))
        rot6ds = rot6ds.reshape(-1, rot6ds.size(-1))  # (B * T, D)
        rotmats = rot6d_to_rotmat(
            rot6ds).reshape(-1, 3, 3)  # (B * T * 24, 3, 3)
        pred['pred_rotmats'] = rotmats.reshape(B, T, 24, 3, 3)
        pred['pred_full_joints'] = full_joints.reshape(B, T, 24, 3)
        pred = {**data, **pred}
        return pred


class TransformerEncoder(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_layers=2, n_heads=8):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_hidden, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=n_layers)
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_output)
        self.dropout = nn.Dropout()

    def forward(self, x):
        B, T, D = x.shape
        # [B, T, D]
        x = self.linear1(x)
        # [B, T, hidden_dim]
        x = x.permute(1, 0, 2)
        # [T, B, hidden_dim]
        x = F.relu(self.dropout(x), inplace=True)
        x = self.transformer_encoder(x)
        # [T, B, hidden_dim]
        x = x.permute(1, 0, 2)
        # [B, T, hidden_dim]
        return self.linear2(x)


class RegressorV2(nn.Module):
    """Replace the biGRU with TransformerEncoder"""
    def __init__(self):
        super().__init__()
        self.encoder = PointNet2Encoder()
        self.pose_s1 = TransformerEncoder(1024, 24 * 3, 1024, 3, 8)
        self.pose_s2 = STGCN(3 + 1024)

    def forward(self, data):
        pred = {}
        x = self.encoder(data)  # [B, T, D]
        B, T, _ = x.shape
        full_joints = self.pose_s1(x)  # [B, T, 24 * 3]
        full_joints = full_joints.reshape(B, T, 24, 3)  # [B, T, 24, 3]
        x = x.unsqueeze(-2).repeat(1, 1, 24, 1)  # [B, T, 24, 1024]
        combined_features = torch.cat((full_joints, x), dim=-1)
        rot6ds = self.pose_s2(combined_features)
        rot6ds = rot6ds.reshape(-1, rot6ds.size(-1))  # [B * T, D]
        rotmats = rot6d_to_rotmat(rot6ds).reshape(-1, 3, 3)  # [B * T * 24, 3, 3]
        pred['pred_rotmats'] = rotmats.reshape(B, T, 24, 3, 3)
        pred['pred_full_joints'] = full_joints.reshape(B, T, 24, 3)
        pred = {**data, **pred}
        return pred
    

class RegressorV3(nn.Module):
    """Add SENet to fuse the global and local feature"""
    def __init__(self):
        super().__init__()
        self.encoder = PointNet2Encoder()
        self.pose_s1 = RNN(1024, 24 * 3, 1024)
        self.pose_s2 = STGCN(3 + 1024)
        self.se_layer = SELayer(channel=3 + 1024, reduction=16)

    def forward(self, data):
        pred = {}
        x = self.encoder(data)  
        B, T, _ = x.shape  # [B, T, D]
        full_joints = self.pose_s1(x)  # [B, T, 24 * 3]
        full_joints = full_joints.reshape(B, T, 24, 3)  # [B, T, 24, 3]
        x = x.unsqueeze(-2).repeat(1, 1, 24, 1)  # [B, T, 24, 1024]
        combined_features = torch.cat((full_joints, x), dim=-1)
        se_output = self.se_layer(
            combined_features.permute(0, 3, 1, 2))  # [B, 1027, T, 24]
        se_output = se_output.permute(0, 2, 3, 1)  # [B, T, 24, 1027]
        # 提取旋转特征
        rot6ds = self.pose_s2(se_output)
        rot6ds = rot6ds.reshape(-1, rot6ds.size(-1))  # [B * T, D]
        rotmats = rot6d_to_rotmat(rot6ds).reshape(-1, 3, 3)  # [B * T * 24, 3, 3]
        pred['pred_rotmats'] = rotmats.reshape(B, T, 24, 3, 3)
        pred['pred_full_joints'] = full_joints.reshape(B, T, 24, 3)
        pred = {**data, **pred}
        return pred


class RegressorV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PointNet2Encoder()
        self.pose_s1 = TransformerEncoder(1024, 24 * 3, 1024, 3, 8)
        self.pose_s2 = STGCN(3 + 1024)
        self.se_layer = SELayer(channel=3 + 1024, reduction=16)

    def forward(self, data):
        pred = {}
        x = self.encoder(data)  # [B, T, D]
        B, T, _ = x.shape
        full_joints = self.pose_s1(x)  # [B, T, 24 * 3]
        full_joints = full_joints.reshape(B, T, 24, 3)  # [B, T, 24, 3]
        x = x.unsqueeze(-2).repeat(1, 1, 24, 1)  # [B, T, 24, 1024]
        combined_features = torch.cat((full_joints, x), dim=-1)
        se_output = self.se_layer(
            combined_features.permute(0, 3, 1, 2))  # [B, 1027, T, 24]
        se_output = se_output.permute(0, 2, 3, 1)  # [B, T, 24, 1027]
        rot6ds = self.pose_s2(se_output)
        rot6ds = rot6ds.reshape(-1, rot6ds.size(-1))  # [B * T, D]
        rotmats = rot6d_to_rotmat(rot6ds).reshape(-1, 3, 3)  # [B * T * 24, 3, 3]
        pred['pred_rotmats'] = rotmats.reshape(B, T, 24, 3, 3)
        pred['pred_full_joints'] = full_joints.reshape(B, T, 24, 3)
        pred = {**data, **pred}
        return pred
