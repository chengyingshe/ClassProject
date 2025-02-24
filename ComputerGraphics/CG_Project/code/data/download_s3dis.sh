#!/bin/bash

data_dir="../../temp/data/S3DIS"

if [ ! -d $data_dir ]; then
    mkdir -p $data_dir
fi
wget -P $data_dir -c https://cvg-data.inf.ethz.ch/s3dis/ReadMe.txt
# wget -P $data_dir https://cvg-data.inf.ethz.ch/s3dis/Stanford3dDataset_v1.2.mat
# wget -P $data_dir https://cvg-data.inf.ethz.ch/s3dis/Stanford3dDataset_v1.2.zip
# wget -P $data_dir https://cvg-data.inf.ethz.ch/s3dis/Stanford3dDataset_v1.2_Aligned_Version.mat
wget -P $data_dir -c https://cvg-data.inf.ethz.ch/s3dis/Stanford3dDataset_v1.2_Aligned_Version.zip
