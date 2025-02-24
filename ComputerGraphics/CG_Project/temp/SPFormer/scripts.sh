# train
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/spf_scannet_ep512.yaml
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/spf_v2_scannet_loss_ep512.yaml

# test
python tools/test.py configs/spf_scannet.yaml checkpoints/spf_scannet_112.pth --out ${SAVE_PATH}

# visualization
python tools/visualization.py --prediction_path ${SAVE_PATH}