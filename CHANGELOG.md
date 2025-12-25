# 更新日志

## [2024-12-25] - 概率式视觉资源分配实验环境

### 新增功能

#### 概率式注意力路由机制
- **核心假设验证**: 实现了 Attention 以概率分布形式决定不同空间区域使用"大计算量 MLP（精细处理）"还是"小计算量 MLP（模糊处理）"的实验环境
- **随机性支持**: 分配具有随机性（类似 dropout），attention 越高的区域被分配到大 MLP 的概率越大

#### 数据集
- **ClutteredMNIST 数据集** (`datasets/cluttered_mnist.py`)
  - 64×64 图像，小尺寸 MNIST 数字（14×14）随机放置
  - 强干扰背景（随机噪声、其他数字碎片、纹理）
  - 标签仅由中心数字决定，模糊视觉不足以完成任务
  - 支持 8×8 patch 提取（共 64 个 patches）

#### 模型架构
- **ProbabilisticAttentionModel**: 提出的概率式路由方法
  - Attention 输出概率值 p ∈ (0,1)
  - 根据概率随机采样路由决策
  - 训练阶段：随机分配
  - 推理阶段：支持期望值/Top-p/硬阈值三种模式

- **对照基线模型**:
  - `BaselineLargeMLP`: 所有 patch 使用 Large MLP
  - `BaselineSmallMLP`: 所有 patch 使用 Small MLP
  - `DeterministicAttentionModel`: 标准 Transformer 式 attention（连续权重）

#### 路由机制
- **Gumbel-Softmax 路由** (`models/probabilistic_router.py`)
  - 可微分的随机采样
  - 支持硬采样和软采样模式

- **Straight-Through 路由**
  - 硬采样 + 软梯度
  - Straight-Through Estimator 实现

#### 推理模式
- **expectation**: 期望值加权 `output = prob * large_mlp(x) + (1-prob) * small_mlp(x)`
- **top_p**: 选择概率最高的 p% patches 使用 Large MLP
- **hard_threshold**: 硬阈值（概率>0.5 使用 Large MLP）

#### 训练与评估
- **训练脚本** (`experiments/train_probabilistic.py`)
  - 完整的训练循环和验证
  - 指标计算（准确率、路由比例等）
  - Checkpoint 保存和 TensorBoard 日志
  - 支持多种路由模式和模型配置

- **评估脚本** (`experiments/evaluate_probabilistic.py`)
  - 支持不同推理模式对比
  - 批量评估所有模型
  - 自动生成可视化

#### 可视化工具
- **可视化模块** (`utils/probabilistic_visualization.py`)
  - Attention 热图可视化
  - 路由决策可视化
  - 概率分布可视化
  - 性能 vs 计算量 trade-off 曲线
  - 训练曲线绘制

#### 配置文件
- **实验配置** (`configs/probabilistic_experiment.yaml`)
  - 完整的超参数配置
  - 模型架构配置
  - 训练配置

### 改进

#### 代码质量
- 修复了数据集导入问题，使用可选导入避免不存在的模块报错
- 修复了模型组件缺失问题，内联实现了 `LightweightEncoder` 和 `FeatureAggregator`
- 修复了变量名冲突问题（`F` 变量覆盖 `torch.nn.functional`）
- 优化了数据加载器配置，避免 "Too many open files" 错误

#### 文档
- 新增 `README_PROBABILISTIC.md` 详细说明实验环境使用方法
- 包含实验设计、使用方法、预期结论等完整文档

### 技术细节

#### 实验设计
- **数据集**: ClutteredMNIST (64×64, 8×8 patches)
- **模型参数**:
  - Large MLP: 3层, hidden_dim=512
  - Small MLP: 1层, hidden_dim=64
  - Attention head: 输出概率值

#### 评估指标
- 分类准确率
- 平均 Large MLP 使用比例（计算开销 proxy）
- Attention/采样概率可视化
- 性能 vs 计算量 trade-off 曲线

### 文件清单

#### 新增文件
- `datasets/cluttered_mnist.py` - ClutteredMNIST 数据集
- `models/probabilistic_router.py` - 概率式路由模块
- `models/probabilistic_model.py` - 概率式模型和基线
- `experiments/train_probabilistic.py` - 训练脚本
- `experiments/evaluate_probabilistic.py` - 评估脚本
- `utils/probabilistic_visualization.py` - 可视化工具
- `configs/probabilistic_experiment.yaml` - 实验配置
- `README_PROBABILISTIC.md` - 实验说明文档
- `CHANGELOG.md` - 更新日志

#### 修改文件
- `datasets/__init__.py` - 添加可选导入支持
- `datasets/factory.py` - 修复导入问题，使用可选导入

### 使用说明

详细使用方法请参考 `README_PROBABILISTIC.md`。

### 注意事项

- 首次运行会自动下载 MNIST 数据集
- 建议使用 `num_workers=0` 避免多进程文件句柄问题
- 需要安装 torch, torchvision, matplotlib 等依赖

### 后续计划

- 扩展到真实视觉数据集
- 添加更多路由策略对比
- 优化计算效率分析
- 增强可视化功能

