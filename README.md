# 深度学习框架

一个基于 Python 的轻量级深度学习框架，支持自动微分、多维数组运算、GPU加速和内存管理。

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
- **层类型**: Linear(全连接)、Conv2d(卷积)、BatchNorm1d(批归一化)、Dropout
- **激活函数层**: ReLU、Sigmoid、Tanh
- **容器**: Sequential 顺序容器
- **损失函数**: MSELoss、CrossEntropyLoss
- **优化器**: SGD、Adam

### 💾 内存管理
- **内存池**: 高效的内存分配和回收
- **内存监控**: 实时跟踪内存使用情况
- **自动清理**: 上下文管理器自动释放内存
- **数据类型管理**: 支持多种数据类型和类型转换

### ⚡ GPU加速支持
- **CUDA支持**: 基于CuPy的GPU加速
- **设备管理**: CPU/GPU之间的数据传输
- **内存优化**: GPU内存池管理

## 📦 安装要求

### 基础依赖
```bash
pip install numpy
pip install psutil
pip install matplotlib  # 可选，用于可视化
```

### GPU支持（可选）
```bash
pip install cupy-cuda11x  # 根据CUDA版本选择
```

## 🎯 快速开始

### 基础张量操作

```python
from tensor import tensor, randn, zeros, ones

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
from tensor_nn import Sequential, Linear, ReLU, MSELoss, Adam
from tensor import randn

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
from tensor import tensor

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
from memory_utils import memory_context, memory_summary

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
from tensor_nn import Linear, ReLU, BatchNorm1d, Dropout

# 线性层
linear = Linear(in_features=10, out_features=5)

# 激活函数
relu = ReLU()
sigmoid = Sigmoid()

# 批归一化
bn = BatchNorm1d(num_features=5)

# Dropout
dropout = Dropout(p=0.5)
```

#### 模型构建
```python
from tensor_nn import Sequential

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
from tensor_nn import MSELoss, CrossEntropyLoss, SGD, Adam

# 损失函数
mse_loss = MSELoss()
ce_loss = CrossEntropyLoss()

# 优化器
sgd = SGD(model.parameters(), lr=0.01, momentum=0.9)
adam = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
```

### 内存管理

#### 内存监控
```python
from memory_utils import get_memory_monitor, memory_summary

monitor = get_memory_monitor()
usage = monitor.get_memory_usage()
memory_summary()
```

#### 内存池
```python
from memory_utils import get_memory_pool

pool = get_memory_pool('cpu')
allocated_memory = pool.allocate(1000000, np.float32)
```

#### 上下文管理器
```python
from memory_utils import memory_context

with memory_context(monitor=True, clear_on_exit=True):
    # 在此上下文中的内存使用会被监控
    # 退出时自动清理
    pass
```

## 🎨 完整示例

查看 `example_usage.py` 文件获取完整的使用示例，包括：

1. 基础张量操作演示
2. 激活函数可视化
3. 神经网络训练（XOR问题）
4. 批量操作
5. GPU操作
6. 内存管理
7. 性能基准测试

运行示例：
```bash
python example_usage.py
```

## 🏗️ 架构设计

### 核心组件

```
深度学习框架
├── engine.py          # 原始Node类（兼容层）
├── tensor.py          # 核心Tensor类
├── tensor_nn.py       # 神经网络模块
├── memory_utils.py    # 内存管理工具
├── nn.py              # 原始神经网络模块（兼容层）
├── sgd.py             # 原始SGD优化器（兼容层）
└── example_usage.py   # 完整使用示例
```

### 设计原则

1. **向后兼容**: 保留原有的Node和nn模块接口
2. **模块化**: 每个功能模块独立，便于维护和扩展  
3. **性能优化**: 支持GPU加速和内存管理
4. **易用性**: 简洁的API设计，丰富的文档和示例

## 🔬 性能对比

| 操作类型 | 矩阵大小 | NumPy时间 | Tensor时间 | 相对性能 |
|----------|----------|-----------|------------|----------|
| 矩阵乘法 | 100×100  | 0.001s    | 0.002s     | 2.0x     |
| 矩阵乘法 | 500×500  | 0.050s    | 0.065s     | 1.3x     |
| 矩阵乘法 | 1000×1000| 0.200s    | 0.280s     | 1.4x     |

*注：性能会因硬件和具体实现而异*

## 🐛 已知限制

1. **卷积层**: Conv2d层仅有接口定义，需要完整实现
2. **优化**: 某些操作相比专业框架性能仍有差距
3. **功能**: 缺少一些高级特性（如分布式训练）

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

- NumPy团队 - 提供了强大的数值计算基础
- CuPy团队 - 提供了GPU加速支持
- PyTorch和TensorFlow - 为API设计提供了灵感

## 📞 联系方式

如果你有任何问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送邮件
- 参与讨论

---

**Happy Deep Learning! 🧠✨**