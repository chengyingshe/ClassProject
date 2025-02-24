import sys
import os

import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from oneformer3d.query_decoder import ScanNetQueryDecoder
from oneformer3d.oneformer3d import ScanNetOneFormer3D, ScanNetOneFormer3DWithPointNet2, ScanNetOneFormer3DWithPTV2Encoder, ScanNetOneFormer3DWithPTV3Encoder
from utils import count_parameters

num_channels = 32
num_instance_classes = 18
num_semantic_classes = 20

cfg = dict(
    in_channels=6,
    num_channels=num_channels,
    voxel_size=0.02,
    num_classes=num_instance_classes,
    min_spatial_shape=128,
    query_thr=0.5,
    backbone=dict(
        type='SpConvUNet',
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True),
    decoder=dict(
        type='ScanNetQueryDecoder',
        num_layers=3,  # 6
        num_instance_queries=0,
        num_semantic_queries=0,
        num_instance_classes=num_instance_classes,
        num_semantic_classes=num_semantic_classes,
        num_semantic_linears=1,
        in_channels=num_channels,
        d_model=128,  # 256
        num_heads=4,  # 8
        hidden_dim=256,  # 512
        dropout=0.1,  # 0.0
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=False),
    criterion=dict(
        type='ScanNetUnifiedCriterion',
        num_semantic_classes=num_semantic_classes,
        sem_criterion=dict(
            type='ScanNetSemanticCriterion',
            ignore_index=num_semantic_classes,
            loss_weight=0.2),
        inst_criterion=dict(
            type='InstanceCriterion',
            matcher=dict(
                type='SparseMatcher',
                costs=[
                    dict(type='QueryClassificationCost', weight=0.5),
                    dict(type='MaskBCECost', weight=1.0),
                    dict(type='MaskDiceCost', weight=1.0)],
                topk=1),
            loss_weight=[0.5, 1.0, 1.0, 0.5],
            num_classes=num_instance_classes,
            non_object_weight=0.1,
            fix_dice_loss_weight=True,
            iter_matcher=True,
            fix_mean_loss=True)),
    train_cfg=dict(),
    test_cfg=dict(
        topk_insts=600,
        inst_score_thr=0.0,
        pan_score_thr=0.5,
        npoint_thr=100,
        obj_normalization=True,
        sp_score_thr=0.4,
        nms=True,
        matrix_nms_kernel='linear',
        stuff_classes=[0, 1]))

model_v1 = ScanNetOneFormer3D(**cfg)

cfg['backbone'] = None
model_v2 = ScanNetOneFormer3DWithPointNet2(**cfg)
model_v3 = ScanNetOneFormer3DWithPTV2Encoder(**cfg)
model_v4 = ScanNetOneFormer3DWithPTV3Encoder(**cfg)

print('parameters v1:', count_parameters(model_v1))  # 11624616
print('parameters v2:', count_parameters(model_v2))  # 2516400
print('parameters v3:', count_parameters(model_v3))  # 671496
print('parameters v4:', count_parameters(model_v4))  # 688360