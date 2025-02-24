model_name="senet"
gpu=1
bs=1
eval_bs=1
threads=12
epochs=100
version="v4"

# 使用 getopt 解析命令行参数
TEMP=$(getopt -o '' --long 'model-name:,gpu:,bs:,eval-bs:,threads:,epochs:,version:' -n "$0" -- "$@")
if [ $? != 0 ] ; then echo "Terminating..." >&2 ; exit 1 ; fi

# Note the quotes around `$TEMP': they are essential!
eval set -- "$TEMP"

# 解析参数
while true ; do
    case "$1" in
        --model-name) model_name="$2" ; shift 2 ;;
        --gpu) gpu="$2" ; shift 2 ;;
        --bs) bs="$2" ; shift 2 ;;
        --eval-bs) eval_bs="$2" ; shift 2 ;;
        --threads) threads="$2" ; shift 2 ;;
        --epochs) epochs="$2" ; shift 2 ;;
        --version) version="$2" ; shift 2 ;;
        --) shift ; break ;;
        *) echo "Internal error!" ; exit 1 ;;
    esac
done

# train
python train.py --bs $bs --eval_bs $eval_bs --gpu $gpu --threads $threads --epochs $epochs --dataset lidarcap --name "lidarcap-$model_name" --version $version --save-ckp-dir $model_name

# copy models
cp -r "output/$model_name" "models/${model_name}-ep${epochs}"

# eval
python train.py --threads $threads --gpu $gpu --dataset lidarcap --ckpt_path "models/$model_name/best-train-loss.pth" --eval --eval_bs $eval_bs --debug --metric-file-name "metric_${model_name}_ep${epochs}.txt" --version $version