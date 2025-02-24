import os

# 设置环境变量
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4120"

# 打印环境变量
print(os.environ["PYTORCH_CUDA_ALLOC_CONF"])