## CG_Project

### 仓库描述

仓库中保存了计算机图形学的课程作业相关文档和代码

### 仓库结构

- `code`: 存储项目代码
- `doc`: 相关markdown文档
- `pdf`: 相关pdf文档

### 小组成员

- 佘成应 2024E8018482019
- 王奔 2024E8018482014
- 张馨然 202418018427003
- 钱伟 202428018427001

### 相关论文及代码参考

| 论文名称    | 任务  | 论文链接                                                             | 代码链接                                         |
| ----------- | -------------------------------------------------------------------- | ------------------------------------------------ | ----------- |
| Superpoint Transformer    | Semantic Segmentation | [ICCV2023](http://arxiv.org/abs/2306.08045) \| [pdf](pdf/Superpoint_Transformer.pdf)  | [Github](https://github.com/drprojects/superpoint_transformer) |
| SPFormer    | Instance Segmentation | [AAAI2023](https://arxiv.org/abs/2211.15766) \| [pdf](pdf/SPFormer.pdf)  | [Github](https://github.com/sunjiahao1999/SPFormer) |
| OneFormer3D | All in One | [CVPR2024](https://arxiv.org/abs/2311.14405) \| [pdf](pdf/OneFormer.pdf) | [Github](https://github.com/filaPro/oneformer3d)    |
| PointNet++    | Point Cloud feature extraction | [NIPS2017](http://arxiv.org/abs/1706.02413) \| [pdf](pdf/PointNet++.pdf)  | [Github](https://github.com/charlesq34/pointnet2)    |


> OneFormer是基于SPFormer网络结构添加部分改进实现的

### 进度安排

- [X] 下载ScanNet数据集（下载的非完整数据集）
- [x] 基于上述框架提供代码对数据集进行预处理（实现了SPFormer框架ScannetV2数据预处理、oneformer3d）
- [x] 基于上述框架实现模型评测、训练（实现了SPFormer、oneformer3d框架模型的训练和评测）
- [x] 基于上述框架提出相关改进方法，并完成代码编写
- [x] 训练改进后的模型，并分别进行评测
- [ ] 基于改进方法和评测结果编写报告及答辩PPT

### 评测结果

评测结果位于: `temp/test_results`文件夹中

### 训练和评测方法

项目路径：`/home/scy/CG_Project` (超链接)

1. `SPFormer`

    ```shell
    cd temp/SPFormer
    conda activate cg-proj
    python tools/train.py configs/spf_scannet.yaml  # train

    python tools/test.py configs/spf_scannet.yaml checkpoints/spf_scannet_512.pth --out <output_path>  # test

    python tools/visualization.py --prediction_path <output_path>   # visualization (<output_path>就是test中的--out参数)
    ```

2. `OneFormer3d`

    ```shell
    cd code
    docker container ls  # 查看运行中的docker容器，如果存在下面所示的名为`oneformer`的容器，则跳过下一步
    # CONTAINER ID   IMAGE          COMMAND                  CREATED        STATUS        PORTS     NAMES
    # cc9af17dcc02   c2c798b4d33c   "/opt/nvidia/nvidia_…"   13 hours ago   Up 13 hours             oneformer
    bash run_docker.sh  # 存在则跳过这一步，直接执行下一条命令
    bash enter_docker.sh  # 进入docker容器
    cd oneformer3d

    # scannetv2
    # train
    python tools/train.py configs/oneformer3d_1xb4_scannet.py
    # test
    python tools/test.py configs/oneformer3d_1xb4_scannet.py checkpoints/oneformer3d_scannetv2-epoch_112.pth
    # 修改配置文件和模型可以更换使用的数据集
    ```

将评测结果保存到`temp/test_results`文件夹中，注意文件命名方式，可以参考我的文件命名方式