#!/bin/bash

# http://www.semantic-kitti.org/dataset.html#download

data_dir="../../temp/data/SemanticKITTI"

if [ ! -d $data_dir ]; then
    mkdir -p $data_dir
fi

wget -P $data_dir -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip
wget -P $data_dir -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip
wget -P $data_dir -c http://www.semantic-kitti.org/assets/data_odometry_labels.zip