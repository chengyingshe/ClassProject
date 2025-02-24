#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python tools/train.py configs/oneformer3d_1xb4_scannet.py --resume
# CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/oneformer3d_1xb2_s3dis-area-5.py --resume
# CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/oneformer3d_1xb2_s3dis-area-5.py checkpoints/oneformer3d_1xb2_s3dis-area-5.pth

# 训练新模型
# max_split_size_mb：设置内存分块大小（越小分块越多，内存占用率更高）
# garbage_collection_threshold：设置垃圾回收阈值（越小越及时回收）

# 单卡训练
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32,garbage_collection_threshold:0.6 CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/oneformer3d_with_ptv2_encoder.py --resume

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32,garbage_collection_threshold:0.6 CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/oneformer3d_with_ptv3_encoder.py

# 多卡训练 nproc_per_node：显卡数量
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32,garbage_collection_threshold:0.6 python -m torch.distributed.launch --nproc_per_node=2 tools/train.py configs/oneformer3d_with_ptv2_encoder.py --resume