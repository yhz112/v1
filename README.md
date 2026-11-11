

## 环境

推荐使用与原始训练环境一致的 Python 3.8 + CUDA 11.1。

| 组件 | 版本 |
| --- | --- |
| Python | 3.8 |
| PyTorch | 1.8.1+cu111 |
| torchvision | 0.9.1+cu111 |
| CUDA | 11.1 |
| numpy | 1.24.4 |
| scipy | 1.10.1 |
| scikit-image | 0.21.0 |
| einops | 0.8.0 |
| tqdm | 4.67.0 |
| tensorboard | 2.14.0 |
| pillow | 10.4.0 |
| editdistance | 0.8.1 |



## 目录结构

```text
v1/
├── code/
│   ├── train.py                     # 训练入口：SAM 优化器 + EMA + 多指标验证
│   ├── valid.py                     # 验证函数：Loss / FM / p-FM / PSNR / DRD
│   ├── test.py                      # 测试入口：加载权重，输出二值化结果图
│   ├── model/
│   │   ├── transformer_imgbin.py    # 主网络 TransferImgBin 及 create_model
│   │   └── module/
│   │       ├── transformer.py       # Transformer 基础块（TransBlock）
│   │       ├── my_transformer.py    # 多尺度注意力块（MS_Block）等
│   │       ├── patch.py             # 图像分块/还原（ExtractImagePatches 等）
│   │       ├── shufflenetv2.py      # ShuffleNetV2 基础块
│   │       └── mabma_s4.py          # 轻量 Mamba/S4 状态空间模块
│   └── utils/
│       ├── option.py                # 命令行参数与各数据集子命令
│       ├── sam.py                   # Sharpness-Aware Minimization 优化器
│       └── utils.py                 # 日志、学习率调度、EMA 等工具
├── dataset/
│   ├── train/
│   │   ├── real/                # 训练原图（RGB，2048×288）
│   │   └── gt/                  # 对应二值化标注（单通道 PNG）
│   ├── val/
│   │   ├── real/                # 验证原图
│   │   └── gt/                  # 验证标注
│   └── test/
│       └── real/                # 测试原图
└── README.md
```



