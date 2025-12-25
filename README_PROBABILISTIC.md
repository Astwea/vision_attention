# 概率式视觉资源分配实验

本实验环境用于验证以下研究假设：

**假设**: Attention 不直接做特征加权，而是以概率分布的形式，决定不同空间区域使用"大计算量 MLP（精细处理）"还是"小计算量 MLP（模糊处理）"。该分配具有随机性（类似 dropout），但 attention 越高的区域，被分配到大 MLP 的概率越大。

## 实验环境

### 数据集: ClutteredMNIST

- **图像尺寸**: 64×64
- **Patch 大小**: 8×8 (共 64 个 patches)
- **主要特征**: 
  - 小尺寸 MNIST 数字（14×14）随机放置在图像中
  - 强干扰背景（随机噪声、其他数字碎片、纹理）
  - 标签仅由中心数字决定
  - 模糊视觉（小 MLP）不足以完成任务

### 模型架构

1. **ProbabilisticAttentionModel** (提出的方法)
   - Attention 输出概率值 p ∈ (0,1)
   - 根据概率随机采样路由决策
   - 训练阶段：随机分配
   - 推理阶段：可切换为期望值/Top-p/硬阈值

2. **BaselineLargeMLP**: 所有 patch 使用 Large MLP

3. **BaselineSmallMLP**: 所有 patch 使用 Small MLP

4. **DeterministicAttentionModel**: 标准 Transformer 式 attention（连续权重）

### 路由机制

- **Gumbel-Softmax**: 可微分的随机采样
- **Straight-Through**: 硬采样 + 软梯度

### 推理模式

- **expectation**: 期望值加权 `output = prob * large_mlp(x) + (1-prob) * small_mlp(x)`
- **top_p**: 选择概率最高的 p% patches 使用 Large MLP
- **hard_threshold**: 硬阈值（概率>0.5 使用 Large MLP）

## 使用方法

### 1. 训练模型

```bash
# 训练概率式模型（Gumbel-Softmax）
python experiments/train_probabilistic.py \
    --config configs/probabilistic_experiment.yaml \
    --model probabilistic \
    --router_mode gumbel_softmax

# 训练概率式模型（Straight-Through）
python experiments/train_probabilistic.py \
    --model probabilistic \
    --router_mode straight_through

# 训练基线模型
python experiments/train_probabilistic.py --model baseline_large
python experiments/train_probabilistic.py --model baseline_small
python experiments/train_probabilistic.py --model deterministic
```

### 2. 评估模型

```bash
# 评估单个模型（所有推理模式）
python experiments/evaluate_probabilistic.py \
    --checkpoint checkpoints/probabilistic/best_model.pth \
    --model probabilistic \
    --inference_mode all \
    --save_vis

# 评估所有模型并对比
python experiments/evaluate_probabilistic.py \
    --eval_all \
    --checkpoint checkpoints/probabilistic/best_model.pth \
    --save_vis
```

### 3. 可视化

可视化工具会自动在评估时生成：
- Attention 热图
- 路由决策可视化
- 概率分布
- 性能 vs 计算量 trade-off 曲线

结果保存在 `visualizations/probabilistic/` 目录。

## 评估指标

- **分类准确率**: 主要性能指标
- **平均 Large MLP 使用比例**: 计算开销 proxy
- **Attention/采样概率可视化**: 可解释性分析
- **性能 vs 计算量 trade-off 曲线**: 效率分析

## 预期实验结论

### 支持假设的现象

1. **性能优势**: 概率式路由在相同计算量下性能优于确定性 attention
2. **计算效率**: 相比全 Large MLP，显著减少计算量同时保持性能
3. **注意力聚焦**: 高 attention 区域确实更频繁被分配到 Large MLP
4. **鲁棒性**: 随机性带来更好的泛化能力

### 可能否定假设的现象

1. **性能下降**: 随机性导致训练不稳定，性能不如确定性方法
2. **计算浪费**: 低 attention 区域偶尔被分配到 Large MLP，浪费计算
3. **收敛困难**: Gumbel-Softmax 温度调参困难，训练不收敛

## 文件结构

```
datasets/
  └── cluttered_mnist.py          # ClutteredMNIST 数据集

models/
  ├── probabilistic_router.py      # 概率式路由模块
  └── probabilistic_model.py       # 概率式模型和基线

experiments/
  ├── train_probabilistic.py       # 训练脚本
  └── evaluate_probabilistic.py    # 评估脚本

utils/
  └── probabilistic_visualization.py  # 可视化工具

configs/
  └── probabilistic_experiment.yaml   # 实验配置
```

## 配置说明

主要配置项在 `configs/probabilistic_experiment.yaml`:

- `model_name`: 模型名称
- `router_mode`: 路由模式 (gumbel_softmax / straight_through)
- `temperature`: 温度参数
- `batch_size`: 批次大小
- `learning_rate`: 学习率
- `num_epochs`: 训练轮数

## 注意事项

1. 首次运行会自动下载 MNIST 数据集
2. 确保有足够的 GPU 内存（建议至少 4GB）
3. 训练时间：每个模型约 1-2 小时（取决于硬件）
4. 可视化需要 matplotlib，确保已安装

## 实验可解释性

1. **清晰的因果链**: Attention → 概率 → 路由决策 → 计算分配 → 性能
2. **对照实验**: 4 个基线模型确保对比公平
3. **可视化**: 多层次的 attention/概率/路由可视化
4. **指标透明**: 计算量、准确率、路由比例等指标全程追踪

