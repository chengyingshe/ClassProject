from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from .pointnet2_utils import PointNetSetAbstractionMsg, PointNetFeaturePropagation, fix_points_num


class PointNet2Encoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        self.in_dim = in_dim  # 9
        self.out_dim = out_dim  # 128
        
        self.sa1 = PointNetSetAbstractionMsg(
            512,   # 1024
            [0.05, 0.1], [16, 32], 
            self.in_dim, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(
            256, [0.1, 0.2], [16, 32], 
            32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(
            64, [0.2, 0.4], [16, 32], 
            128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(
            16, [0.4, 0.8], [16, 32], 
            256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(
            512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(
            128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(
            32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(
            128, [128, 128, self.out_dim])
        

    def forward(self, data):
        # [N, 6] : 6->xyzrgb
        if not isinstance(data, torch.Tensor):
            data_tensor = data.features
        else:
            data_tensor = data
        data_tensor = data_tensor.unsqueeze(0).transpose(1, 2).contiguous()
        
        # [1, 6, N] -> [B, C, N]
        l0_points = data_tensor  # [1, 6, N]
        l0_xyz = data_tensor[:,:3,:]  # [1, 3, N]
        # print('####l0_xyz:', l0_xyz.shape)
        # print('####l0_points:', l0_points.shape)
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # print('####l1_xyz:', l1_xyz.shape)
        # print('####l1_points:', l1_points.shape)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # print('####l2_xyz:', l2_xyz.shape)
        # print('####l2_points:', l2_points.shape)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # print('####l3_xyz:', l3_xyz.shape)
        # print('####l3_points:', l3_points.shape)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        # print('####l4_xyz:', l4_xyz.shape)
        # print('####l4_points:', l4_points.shape)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        out = l0_points.squeeze(0).transpose(0, 1).contiguous()  # [N, C]
        # print('####out:', out.shape)
        return out

if __name__ == "__main__":
    encoder = PointNet2Encoder(6, 32)
    sample = torch.randn(161494, 6)
    print(encoder)
    print(sample.shape)
    print(encoder(sample).shape)
    