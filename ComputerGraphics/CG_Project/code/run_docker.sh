#!/bin/bash

oneformer3d_dir="$(pwd)/../temp/oneformer3d"
point_transformer_dir="$(pwd)/../temp/PointTransformerV3"

docker run -td --gpus all --shm-size="100g" --name oneformer -v ${oneformer3d_dir}:/workspace/oneformer3d -v ${point_transformer_dir}:/workspace/PointTransformerV3 c2c798b4d33c bash