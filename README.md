# EasyTensor

一个基于 Python 的轻量级深度学习框架，支持自动微分、多维数组运算、GPU加速和内存管理。EasyTensor 提供了类似 PyTorch 的 API 设计，让深度学习变得更加简单易用。

## 🚀 主要特性

### 🔢 多维数组支持
- **Tensor类**: 支持多维数组的创建、操作和自动微分
- **广播机制**: 自动处理不同形状张量之间的运算
- **形状操作**: reshape、transpose、squeeze、expand等
- **数学运算**: 矩阵乘法、元素级运算、聚合操作

### 🧮 基础数学运算
- **基础运算**: 加减乘除、幂运算、开方、对数、指数
- **矩阵运算**: 矩阵乘法、转置、求和、均值
- **激活函数**: ReLU、Sigmoid、Tanh、Softmax
- **自动微分**: 反向传播、梯度计算

### 🧠 神经网络模块
- **层类型**: Linear(全连接)、Conv2d(卷积)、BatchNorm1d(批归一化)、Dropout、RNN、LSTM
- **激活函数层**: ReLU、Sigmoid、Tanh、Softmax
- **容器**: Sequential 顺序容器
- **损失函数**: MSELoss、CrossEntropyLoss、BCEWithLogitsLoss、KLDivergenceLoss
- **优化器**: SGD、Adam（支持动量和学习率调度）
- **高级模块**: BERT、GPT、Transformer、注意力机制、知识蒸馏

### 💾 内存管理
- **内存池**: 高效的内存分配和回收
- **内存监控**: 实时跟踪内存使用情况
- **自动清理**: 上下文管理器自动释放内存
- **数据类型管理**: 支持多种数据类型和类型转换
- **张量注册表**: 自动跟踪和管理张量对象
- **自动内存管理**: 智能内存清理和优化

### ⚡ GPU加速支持
- **CUDA支持**: 基于CuPy的GPU加速
- **设备管理**: CPU/GPU之间的数据传输
- **内存优化**: GPU内存池管理和限制设置
- **混合精度**: 支持不同精度的数值计算

### 🔧 数据处理
- **数据加载**: 支持自定义数据集和数据加载器
- **文本处理**: 内置分词器和词汇表管理
- **词向量**: Word2Vec嵌入支持

## 📦 安装要求

### 基础依赖
```bash
pip install numpy
pip install psutil
pip install matplotlib  # 可选，用于可视化
pip install scikit-learn  # 可选，用于数据预处理
```

### GPU支持（可选）
```bash
pip install cupy-cuda11x  # 根据CUDA版本选择，支持CUDA 11.x
pip install cupy-cuda12x  # 或支持CUDA 12.x
```

### 开发依赖
```bash
pip install jupyter  # 用于运行示例notebook
pip install pytest   # 用于运行测试
```

## 🎯 快速开始

### 基础张量操作

```python
from core.tensor import Tensor, tensor, randn, zeros, ones

# 创建张量
a = tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
b = tensor([[2, 3], [4, 5], [6, 7]], requires_grad=True)

# 矩阵乘法
c = a @ b
print(c)

# 反向传播
loss = c.sum()
loss.backward()
print(a.grad.data)  # 查看梯度
```

### 神经网络训练

```python
from core.nn.tensor_nn import Sequential, Linear, ReLU, MSELoss, Adam
from core.tensor import randn

# 创建模型
model = Sequential(
    Linear(10, 20),
    ReLU(),
    Linear(20, 1)
)

# 创建数据
X = randn(100, 10)
y = randn(100, 1)

# 定义损失函数和优化器
criterion = MSELoss()
optimizer = Adam(model.parameters(), lr=0.01)

# 训练循环
for epoch in range(100):
    # 前向传播
    output = model(X)
    loss = criterion(output, y)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.data:.4f}')
```

### GPU加速

```python
from core.tensor import tensor

# 创建CPU张量
cpu_tensor = tensor([[1, 2], [3, 4]], requires_grad=True)

# 移动到GPU
gpu_tensor = cpu_tensor.cuda()

# GPU上进行运算
result = gpu_tensor @ gpu_tensor.T

# 移回CPU
cpu_result = result.cpu()
```

### 内存管理

```python
from core.utils.memory_utils import memory_context, memory_summary

# 使用内存上下文管理器
with memory_context(monitor=True, clear_on_exit=True):
    large_tensor = randn(1000, 1000)
    # 自动清理内存

# 查看内存使用情况
memory_summary()
```

## 📚 详细文档

### Tensor 类

`Tensor` 是框架的核心数据结构，支持多维数组操作和自动微分。

#### 创建张量
```python
from core.tensor import Tensor, tensor, zeros, ones, randn, eye

# 从数据创建
a = tensor([1, 2, 3], requires_grad=True)
b = tensor([[1, 2], [3, 4]], device='cuda')

# 特殊张量
zeros_tensor = zeros(3, 4)
ones_tensor = ones(2, 3)
random_tensor = randn(5, 5)
identity_matrix = eye(4)
```

#### 张量属性
```python
print(a.shape)      # 形状
print(a.ndim)       # 维度数
print(a.size)       # 元素总数
print(a.dtype)      # 数据类型
print(a.device)     # 设备
```

#### 数学运算
```python
# 基础运算
c = a + b           # 加法
c = a * b           # 乘法
c = a / b           # 除法
c = a ** 2          # 幂运算

# 矩阵运算
c = a @ b           # 矩阵乘法
c = a.T             # 转置

# 聚合运算
c = a.sum()         # 求和
c = a.mean(axis=1)  # 按轴求均值

# 激活函数
c = a.relu()        # ReLU
c = a.sigmoid()     # Sigmoid
c = a.tanh()        # Tanh
```

#### 形状操作
```python
# 改变形状
b = a.reshape(2, 3)

# 转置
b = a.transpose()
b = a.T

# 维度操作
b = a.sum(axis=0, keepdims=True)
```

### 神经网络模块

#### 层定义
```python
from core.nn.tensor_nn import Linear, ReLU, Sigmoid, BatchNorm1d, Dropout
from core.nn.modules.conv import Conv2d
from core.nn.modules.rnn import RNN, LSTM

# 线性层
linear = Linear(in_features=10, out_features=5)

# 激活函数
relu = ReLU()
sigmoid = Sigmoid()

# 批归一化
bn = BatchNorm1d(num_features=5)

# Dropout
dropout = Dropout(p=0.5)

# 卷积层
conv = Conv2d(in_channels=3, out_channels=64, kernel_size=3)

# 循环神经网络
rnn = RNN(input_size=128, hidden_size=256, num_layers=2)
lstm = LSTM(input_size=128, hidden_size=256, num_layers=2)
```

#### 模型构建
```python
from core.nn.tensor_nn import Sequential

model = Sequential(
    Linear(784, 128),
    ReLU(),
    BatchNorm1d(128),
    Dropout(0.2),
    Linear(128, 64),
    ReLU(),
    Linear(64, 10)
)
```

#### 损失函数和优化器
```python
from core.nn.tensor_nn import MSELoss, CrossEntropyLoss, SGD, Adam
from core.nn.loss import BCEWithLogitsLoss

# 损失函数
mse_loss = MSELoss()
ce_loss = CrossEntropyLoss()
bce_loss = BCEWithLogitsLoss()

# 优化器
sgd = SGD(model.parameters(), lr=0.01, momentum=0.9)
adam = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
```

### 内存管理

#### 内存监控
```python
from core.utils.memory_utils import get_memory_monitor, memory_summary

monitor = get_memory_monitor()
usage = monitor.get_memory_usage()
memory_summary()
```

#### 内存池
```python
from core.utils.memory_utils import get_memory_pool

pool = get_memory_pool('cpu')
allocated_memory = pool.allocate(1000000, np.float32)
```

#### 上下文管理器
```python
from core.utils.memory_utils import memory_context

with memory_context(monitor=True, clear_on_exit=True):
    # 在此上下文中的内存使用会被监控
    # 退出时自动清理
    pass
```

## 🎨 完整示例

查看 `test/unit/demo_basic_tensor_operations.py` 文件获取完整的使用示例，包括：

1. 基础张量操作演示
2. 激活函数可视化
3. 神经网络训练（XOR问题）
4. 批量操作
5. GPU操作
6. 内存管理
7. 性能基准测试

运行示例：
```bash
python test/unit/demo_basic_tensor_operations.py
```

## 🏗️ 架构设计

### 核心组件

```
EasyTensor/
├── core/                    # 核心模块
│   ├── tensor.py           # 核心Tensor类
│   ├── device.py           # 设备管理
│   ├── model_io.py         # 模型保存和加载
│   ├── nn/                 # 神经网络模块
│   │   ├── tensor_nn.py    # 基础神经网络层
│   │   ├── modules/        # 具体模块实现
│   │   │   ├── conv.py     # 卷积层
│   │   │   ├── embedding.py # 嵌入层
│   │   │   ├── pooling.py  # 池化层
│   │   │   └── rnn.py      # 循环神经网络
│   │   ├── attention.py    # 注意力机制
│   │   ├── bert_gpt.py     # BERT/GPT模型
│   │   ├── transform.py    # Transformer模型
│   │   └── distill.py      # 知识蒸馏
│   ├── data/               # 数据处理
│   │   ├── dataloader.py   # 数据加载器
│   │   └── word2vec.py     # 词向量
│   ├── optim/              # 优化器
│   │   └── lr_scheduler.py # 学习率调度器
│   ├── utils/              # 工具模块
│   │   ├── memory_utils.py # 内存管理
│   │   ├── tokenizer.py    # 分词器
│   │   └── serialization.py # 序列化工具
│   └── v1/                 # 兼容层
│       ├── engine.py       # 原始Node类
│       ├── nn.py           # 原始神经网络模块
│       └── optim/          # 原始优化器
├── test/                   # 测试和示例
│   ├── unit/               # 单元测试
│   ├── forward/            # 前向传播测试
│   └── network/            # 网络测试
└── biz/                    # 业务示例
    └── cnn.py              # CNN示例
```

### 设计原则

1. **向后兼容**: 保留原有的Node和v1模块接口，确保平滑升级
2. **模块化**: 每个功能模块独立，便于维护和扩展
3. **性能优化**: 支持GPU加速和智能内存管理
4. **易用性**: 类似PyTorch的API设计，降低学习成本
5. **可扩展性**: 支持自定义层、优化器和损失函数
6. **内存效率**: 智能内存池和自动垃圾回收

## 🔬 性能对比

| 操作类型 | 矩阵大小 | NumPy时间 | EasyTensor时间 | PyTorch时间 | 相对性能 |
|----------|----------|-----------|----------------|-------------|----------|
| 矩阵乘法 | 100×100  | 0.001s    | 0.002s         | 0.001s      | 2.0x     |
| 矩阵乘法 | 500×500  | 0.050s    | 0.065s         | 0.045s      | 1.3x     |
| 矩阵乘法 | 1000×1000| 0.200s    | 0.280s         | 0.180s      | 1.4x     |
| GPU矩阵乘法 | 1000×1000| N/A | 0.120s | 0.080s | 1.5x |

*注：性能会因硬件和具体实现而异。运行 `test/横向对比测试.py` 获取详细性能对比*

## 🐛 已知限制

1. **部分实现**: 某些高级功能仍在开发中
2. **性能**: 相比专业框架（PyTorch/TensorFlow）性能仍有差距
3. **分布式**: 暂不支持分布式训练
4. **生态**: 缺少丰富的预训练模型和工具链

## 🧪 测试和示例

### 运行测试
```bash
# 基础功能测试
python test/unit/demo_basic_tensor_operations.py

# 性能对比测试
python test/横向对比测试.py

# 引擎测试
python test/引擎测试.py

# 网络测试
python test/unit/deep_network_test.py
```

### 高级示例
```bash
# BERT/GPT示例
python core/nn/bert_gpt_example.py

# 知识蒸馏示例
python core/nn/distill.py

# CNN示例
python biz/cnn.py
```

## 🤝 贡献指南

欢迎提交Pull Request来改进这个框架！

1. Fork 这个项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

这个项目使用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- **NumPy团队** - 提供了强大的数值计算基础
- **CuPy团队** - 提供了GPU加速支持
- **PyTorch团队** - 为API设计提供了灵感
- **TensorFlow团队** - 为架构设计提供了参考
- **开源社区** - 感谢所有贡献者和用户的支持

## 📞 联系方式

如果你有任何问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送邮件
- 参与讨论

---

**Happy Deep Learning with EasyTensor! 🧠✨**

*让深度学习变得更简单、更高效、更有趣！*