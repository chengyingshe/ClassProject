#!bin/bash

# data_dir="../../temp/data/scannetv2/tasks"

# if [ ! -d $data_dir ]; then
#     mkdir -p $data_dir
# fi

# wget -P $data_dir -c http://kaldir.vc.in.tum.de/scannet/v2/tasks/scannet_frames_25k.zip
# wget -P $data_dir -c http://kaldir.vc.in.tum.de/scannet/v2/tasks/scannet_frames_test.zip

# downaload partial data
# python download_scannetv2.py -o scannet/ --type  _vh_clean_2.ply
# python download_scannetv2.py -o scannet/ --type  _vh_clean_2.labels.ply
# python download_scannetv2.py -o scannet/ --type  _vh_clean_2.0.010000.segs.json
# python download_scannetv2.py -o scannet/ --type  .aggregation.json
# python download_scannetv2.py -o scannet/ --type  .txt
python download_scannetv2.py -o scannet/ --type  .sens
