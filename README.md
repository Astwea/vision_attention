# Vision Attention Routing Framework

一个干净的PyTorch研究框架，用于评估条件计算视觉架构。本框架实现了注意力引导的硬路由机制，能够选择性地将图像区域路由到大型或小型MLP，在保持性能的同时减少计算量。

## 研究目标

评估注意力引导的硬路由机制是否能够：

1. **减少计算量**（FLOPs / 推理时间）
2. **保持或提升任务性能**

路由机制决定哪些图像区域需要昂贵的计算。

## 架构概述

### 数据流程

```
输入图像 (B×C×H×W)
    ↓
卷积编码器（轻量级，低分辨率）
    ↓
注意力头 → 注意力分数 (B×N_patches)
    ↓
可学习阈值 + 硬路由决策
    ↓
分割Patches:
    ├─ 高注意力 (≥阈值) → 从原图裁剪 → 大型MLP（昂贵）
    └─ 低注意力 (<阈值) → 小型MLP（廉价）
    ↓
聚合区域特征
    ↓
任务头（分类）
```

### 核心组件

1. **轻量级编码器** (`backbone.py`): 提取低分辨率特征图
2. **注意力头** (`attention.py`): 预测每个patch的注意力分数
3. **路由器** (`router.py`): 使用可学习阈值进行硬路由决策，支持straight-through estimator
4. **大型MLP** (`mlp_big.py`): 处理高注意力patches（昂贵计算）
5. **小型MLP** (`mlp_small.py`): 处理低注意力patches（廉价计算）
6. **聚合器** (`aggregator.py`): 聚合所有区域特征
7. **任务头**: 分类器

### 关键特性

- **条件计算**: 大型MLP仅处理被选中的patches，实现真正的计算节约
- **硬路由决策**: 前向传播使用硬决策（binary routing）
- **Straight-Through Estimator**: 反向传播时梯度通过软版本流动
- **可学习阈值**: 阈值参数可以通过训练学习

## 基线模型

框架实现了三个基线模型用于对比：

1. **FullComputeBaseline**: 所有patches使用大型MLP处理
2. **CheapComputeBaseline**: 所有patches使用小型MLP处理
3. **NoRoutingBaseline**: 相同backbone但无注意力门控

## 数据集

当前支持：
- CIFAR-10 / CIFAR-100
- 可配置输入分辨率（32×32, 64×64等）
- 可配置patch网格大小（4×4, 8×8等）

## 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA (可选，用于GPU加速)

### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- torch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.24.0
- pyyaml >= 6.0
- matplotlib >= 3.7.0
- thop >= 0.1.0 (用于FLOPs计算)
- tensorboard >= 2.13.0

## 快速开始

### 1. 训练模型

使用默认配置训练注意力路由模型：

```bash
python experiments/train.py --config configs/default.yaml
```

自定义配置：

```bash
python experiments/train.py \
    --config configs/default.yaml \
    --dataset.name cifar10 \
    --dataset.patch_grid_size 4 \
    --training.batch_size 128 \
    --training.num_epochs 100 \
    --training.learning_rate 0.001 \
    --model.name attention_routing
```

### 2. 评估模型

评估训练好的模型：

```bash
python experiments/evaluate.py \
    --config configs/default.yaml
```

评估所有基线模型并对比：

```bash
python experiments/evaluate.py \
    --config configs/default.yaml \
    --mode eval_all
```

### 3. 消融实验

运行消融实验（阈值敏感性、patch网格大小、路由开关等）：

```bash
python experiments/run_ablation.py \
    --config configs/default.yaml
```

## 配置说明

主要配置项在 `configs/default.yaml` 中：

```yaml
# 数据集配置
dataset:
  name: "cifar10"  # cifar10 或 cifar100
  input_size: 32
  patch_grid_size: 4  # patch网格大小（4x4=16个patches）

# 模型配置
model:
  name: "attention_routing"  # attention_routing, full_compute, cheap_compute, no_routing
  router:
    mode: "learnable_threshold"  # learnable_threshold 或 sigmoid_gate
    threshold_init: 0.5
    temperature: 1.0

# 训练配置
training:
  batch_size: 128
  num_epochs: 100
  learning_rate: 0.001
```

## 实验复现

### 基本实验流程

1. **训练主模型**:
```bash
python experiments/train.py --model.name attention_routing
```

2. **训练基线模型**:
```bash
python experiments/train.py --model.name full_compute
python experiments/train.py --model.name cheap_compute
python experiments/train.py --model.name no_routing
```

3. **评估所有模型**:
```bash
python experiments/evaluate.py --mode eval_all
```

4. **运行消融实验**:
```bash
python experiments/run_ablation.py
```

### 预期输出

训练过程中会输出：
- 训练/验证准确率
- 平均路由比例（路由到大型MLP的patches百分比）
- 每张图像的FLOPs估算
- 定期保存的注意力热图可视化

结果保存在：
- `checkpoints/best_model.pth`: 最佳模型checkpoint
- `checkpoints/training_results.json`: 训练历史
- `checkpoints/evaluation_results.json`: 评估结果
- `checkpoints/visualizations/`: 注意力可视化图像

## 假设与研究问题

### 核心假设

1. **空间稀疏性**: 图像中的关键信息集中在部分区域，大部分区域可以用廉价计算处理
2. **注意力引导**: 注意力机制可以有效识别需要精细处理的关键区域
3. **条件计算效率**: 通过路由机制，可以用更少的计算达到接近全量计算的性能

### 研究问题

1. 注意力路由机制能在多大程度上减少计算量？
2. 性能损失是否可接受？
3. 路由决策是否合理（是否能识别关键区域）？
4. 不同阈值设置的影响如何？
5. Patch网格大小如何影响性能和效率？

## 路由机制详解

### Straight-Through Estimator

为了在训练时保持可导性，使用了straight-through estimator：

- **前向传播**: 使用硬决策（hard threshold）
- **反向传播**: 梯度通过软版本（sigmoid）流动

```python
# 前向: 硬决策
hard_mask = (attention_scores >= threshold).float()

# 反向: 软梯度
soft_mask = sigmoid((attention_scores - threshold) / temperature)
routing_mask = hard_mask + (soft_mask - soft_mask.detach())
```

### 条件计算实现

关键点：大型MLP仅处理选中的patches：

```python
# 使用mask筛选patches
selected_patches = patches[routing_mask]
# 只对选中patches执行大型MLP
high_attention_embeddings = mlp_big(selected_patches)
```

这确保了实际的计算节约，而非仅权重调整。

## 评估指标

框架追踪以下指标：

1. **准确率**: 分类任务准确率
2. **路由比例**: 平均路由到大型MLP的patches百分比
3. **FLOPs**: 每张图像的平均FLOPs（考虑条件计算）
4. **注意力热图**: 可视化哪些区域被路由到大型MLP

## 目录结构

```
vision_attention/
├── models/              # 模型组件
│   ├── backbone.py      # 卷积编码器
│   ├── attention.py     # 注意力头
│   ├── router.py        # 路由模块
│   ├── mlp_big.py       # 大型MLP
│   ├── mlp_small.py     # 小型MLP
│   ├── aggregator.py    # 特征聚合
│   └── model.py         # 完整模型组装
├── datasets/            # 数据集
│   └── cifar.py         # CIFAR数据加载器
├── experiments/         # 实验脚本
│   ├── train.py         # 训练脚本
│   ├── evaluate.py      # 评估脚本
│   └── run_ablation.py  # 消融实验
├── utils/               # 工具函数
│   ├── flops.py         # FLOPs计算
│   ├── visualization.py # 可视化工具
│   └── config.py        # 配置管理
├── configs/             # 配置文件
│   └── default.yaml
└── README.md
```

## 可视化

框架提供了丰富的可视化功能：

1. **注意力热图**: 在输入图像上叠加注意力分数
2. **路由决策**: 可视化哪些区域被路由到大型/小型MLP
3. **Patch网格**: 显示每个patch的注意力分数和路由决策
4. **训练曲线**: 训练和验证的损失和准确率曲线

可视化结果保存在 `checkpoints/visualizations/` 目录。

## 故障排除

### 常见问题

1. **CUDA out of memory**: 减小batch size或输入分辨率
2. **thop未安装**: `pip install thop`
3. **检查点加载失败**: 确保模型配置与训练时一致

## 扩展

框架设计易于扩展：

- **新数据集**: 在 `datasets/` 中添加新的数据加载器
- **新路由机制**: 在 `router.py` 中实现新的路由策略
- **新聚合方法**: 在 `aggregator.py` 中添加新的聚合方式
- **新评估指标**: 在 `experiments/evaluate.py` 中添加指标计算

## 引用

如果使用本框架进行研究，请引用：

```bibtex
@software{vision_attention_routing,
  title = {Vision Attention Routing Framework},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/vision-attention-routing}
}
```

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 致谢

感谢PyTorch社区和相关开源项目的支持。

# vision_attention
