# train
# python train.py --bs 8 --eval_bs 8 --gpu 0 --threads 16 --dataset lidarcap --name lidarcap-v3 --epochs 100 --version "v3"

# eval
# python train.py --threads 16 --gpu 0 --dataset lidarcap --ckpt_path models/v4-ep10/best-train-loss.pth --eval --eval_bs 2 --debug --metric-file-name metric_v4_ep10.txt --version "v4"

bash train_and_eval.sh --model-name "v4" --gpu 0 --bs 4 --eval-bs 1 --threads 16 --epochs 10 --version "v4"

# for model in "v1" "v2"; do
#     bash train_and_eval.sh --model-name $model --gpu 1 --bs 8 --eval-bs 1 --threads 16 --epochs 20 --version $model
# done

# for model in "v3"; do
#     bash train_and_eval.sh --model-name $model --gpu 0 --bs 10 --eval-bs 1 --threads 16 --epochs 20 --version $model
# done