#!/bin/bash

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32,garbage_collection_threshold:0.1 CUDA_VISIBLE_DEVICES=1 python tools/test.py configs/oneformer3d_with_sparse3dunet_scannet-y.py checkpoints/oneformer3d_pointnet_ori_epoch_100.pth
CUDA_VISIBLE_DEVICES=1 python tools/test.py configs/oneformer3d_with_sparse3dunet_scannet.py cofnfigs/oneformer3d_pointnet_ori_epoch_100.pth
#多gpu测试
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32,garbage_collection_threshold:0.6 python -m torch.distributed.launch --nproc_per_node=2 tools/test.py configs/oneformer3d_with_sparse3dunet_scannet-y.py checkpoints/oneformer3d_pointnet_ori_epoch_100.pth