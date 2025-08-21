from core.tensor import Tensor, zeros, randn
import numpy as np
import os
import pickle

class Module:
    """神经网络模块基类"""

    def __init__(self):
        self.training = True
        self._parameters = []

    def parameters(self):
        """返回模块的所有参数"""
        params = []
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self):
        """清零所有参数的梯度"""
        for param in self.parameters():
            param.zero_grad()

    def train(self):
        """设置为训练模式"""
        self.training = True
        for name, module in self.__dict__.items():
            if isinstance(module, Module):
                module.train()

    def eval(self):
        """设置为评估模式"""
        self.training = False
        for name, module in self.__dict__.items():
            if isinstance(module, Module):
                module.eval()

    def to(self, device):
        """移动模块到指定设备"""
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor):
                setattr(self, name, value.to(device))
            elif isinstance(value, Module):
                value.to(device)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        item.to(device)
        return self

    def state_dict(self, prefix=''):
        """
        返回模块的状态字典，包含所有参数和缓冲区

        Args:
            prefix (str): 参数名前缀

        Returns:
            dict: 状态字典
        """
        state_dict = {}

        # 收集当前模块的参数和缓冲区
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor):
                # 参数或缓冲区
                full_name = prefix + name if prefix else name
                state_dict[full_name] = value.data.copy()
            elif isinstance(value, Module):
                # 子模块
                sub_prefix = f"{prefix}{name}." if prefix else f"{name}."
                state_dict.update(value.state_dict(sub_prefix))
            elif isinstance(value, list):
                # 模块列表（如Sequential中的layers）
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        sub_prefix = f"{prefix}{name}.{i}." if prefix else f"{name}.{i}."
                        state_dict.update(item.state_dict(sub_prefix))

        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        """
        从状态字典加载参数

        Args:
            state_dict (dict): 状态字典
            strict (bool): 是否严格匹配参数名

        Returns:
            tuple: (missing_keys, unexpected_keys)
        """
        missing_keys = []
        unexpected_keys = []

        # 获取当前模型的状态字典
        current_state_dict = self.state_dict()

        # 检查缺失的键
        for key in current_state_dict.keys():
            if key not in state_dict:
                missing_keys.append(key)

        # 检查多余的键
        for key in state_dict.keys():
            if key not in current_state_dict:
                unexpected_keys.append(key)

        # 在严格模式下，如果有缺失或多余的键，抛出异常
        if strict and (missing_keys or unexpected_keys):
            error_msg = []
            if missing_keys:
                error_msg.append(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                error_msg.append(f"Unexpected keys: {unexpected_keys}")
            raise RuntimeError("Error(s) in loading state_dict:\n" + "\n".join(error_msg))

        # 加载参数
        self._load_from_state_dict(state_dict, prefix='')

        return missing_keys, unexpected_keys

    def _load_from_state_dict(self, state_dict, prefix=''):
        """
        递归加载状态字典中的参数

        Args:
            state_dict (dict): 状态字典
            prefix (str): 当前前缀
        """
        # 加载当前模块的参数
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor):
                full_name = prefix + name if prefix else name
                if full_name in state_dict:
                    # 检查形状是否匹配
                    if value.shape != state_dict[full_name].shape:
                        raise RuntimeError(
                            f"Size mismatch for {full_name}: "
                            f"copying a param with shape {state_dict[full_name].shape} "
                            f"from checkpoint, the shape in current model is {value.shape}"
                        )
                    # 加载数据
                    value.data = state_dict[full_name].copy()

            elif isinstance(value, Module):
                # 递归加载子模块
                sub_prefix = f"{prefix}{name}." if prefix else f"{name}."
                value._load_from_state_dict(state_dict, sub_prefix)

            elif isinstance(value, list):
                # 处理模块列表
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        sub_prefix = f"{prefix}{name}.{i}." if prefix else f"{name}.{i}."
                        item._load_from_state_dict(state_dict, sub_prefix)

    def save(self, filepath):
        """
        保存模型到文件

        Args:
            filepath (str): 保存路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        # 保存状态字典
        with open(filepath, 'wb') as f:
            pickle.dump(self.state_dict(), f)

        print(f"模型已保存到: {filepath}")


class Linear(Module):
    """线性层/全连接层"""

    def __init__(self, in_features, out_features, bias=True, device='cpu'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        # Xavier/Glorot 初始化
        std = np.sqrt(2.0 / (in_features + out_features))
        self.weight = randn(out_features, in_features, requires_grad=True, device=device) * std

        if bias:
            self.bias = zeros(out_features, requires_grad=True, device=device)
        else:
            self.bias = None

    def forward(self, x):
        """前向传播"""
        # x: (batch_size, in_features) 或更高维度
        # weight: (out_features, in_features)
        # 输出: (batch_size, out_features)

        if x.ndim == 1:
            # 单个样本情况
            output = self.weight @ x.reshape(-1, 1)
            output = output.reshape(-1)
        else:
            # 批量处理: (batch_size, in_features) @ (in_features, out_features)
            output = x @ self.weight.T

        if self.bias is not None:
            output = output + self.bias

        return output

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"


class ReLU(Module):
    """ReLU激活函数层"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.relu()

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return "ReLU()"


class Sigmoid(Module):
    """Sigmoid激活函数层"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.sigmoid()

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return "Sigmoid()"


class Tanh(Module):
    """Tanh激活函数层"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.tanh()

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return "Tanh()"


class Sequential(Module):
    """顺序容器"""

    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def add(self, layer):
        """添加层"""
        self.layers.append(layer)

    def __getitem__(self, idx):
        return self.layers[idx]

    def __len__(self):
        return len(self.layers)

    def __repr__(self):
        layer_strs = [f"  ({i}): {layer}" for i, layer in enumerate(self.layers)]
        return "Sequential(\n" + "\n".join(layer_strs) + "\n)"

class Conv2d(Module):
    """2D卷积层（简化版本）"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, device='cpu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        self.device = device

        # 初始化权重 (out_channels, in_channels, kernel_h, kernel_w)
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        fan_out = out_channels * self.kernel_size[0] * self.kernel_size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))

        self.weight = randn(
            out_channels, in_channels, self.kernel_size[0], self.kernel_size[1],
            requires_grad=True, device=device
        ) * std

        if bias:
            self.bias = zeros(out_channels, requires_grad=True, device=device)
        else:
            self.bias = None

    def forward(self, x):
        """简化的卷积前向传播（仅作演示）"""
        # 这里只是一个占位实现，真正的卷积需要更复杂的实现
        # 实际应用中需要实现im2col或使用底层库
        raise NotImplementedError("Conv2d forward pass needs to be implemented with proper convolution algorithms")

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return (f"Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})")


class BatchNorm1d(Module):
    """1D批归一化层"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, device='cpu'):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.device = device

        # 可学习参数
        self.weight = Tensor(np.ones(num_features), requires_grad=True, device=device)
        self.bias = zeros(num_features, requires_grad=True, device=device)

        # 运行时统计量（不需要梯度）
        self.running_mean = zeros(num_features, device=device)
        self.running_var = Tensor(np.ones(num_features), device=device)

        self.num_batches_tracked = 0

    def forward(self, x):
        """前向传播"""
        if self.training:
            # 训练模式：计算当前批次的均值和方差
            if x.ndim == 1:
                # 处理单个样本
                mean = x.mean()
                var = ((x - mean) ** 2).mean()
            else:
                # 批量处理
                mean = x.mean(axis=0)
                var = ((x - mean) ** 2).mean(axis=0)

            # 更新运行时统计量
            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * mean.data
            self.running_var.data = (1 - self.momentum) * self.running_var.data + self.momentum * var.data
            self.num_batches_tracked += 1
        else:
            # 评估模式：使用运行时统计量
            mean = self.running_mean
            var = self.running_var

        # 归一化
        x_normalized = (x - mean) / ((var + self.eps) ** 0.5)

        # 缩放和平移
        return self.weight * x_normalized + self.bias

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"BatchNorm1d(num_features={self.num_features}, eps={self.eps}, momentum={self.momentum})"


class Dropout(Module):
    """Dropout层"""

    def __init__(self, p=0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability should be between 0 and 1, got {p}")
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        # 生成dropout掩码
        xp = x._get_array_module()
        keep_prob = 1 - self.p
        mask = xp.random.random(x.shape) < keep_prob

        # 应用dropout并缩放
        return x * Tensor(mask.astype(x.dtype), device=x.device) / keep_prob

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"Dropout(p={self.p})"


# ==================== 损失函数 ====================

class Loss(Module):
    """损失函数基类"""

    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        raise NotImplementedError

    def __call__(self, predictions, targets):
        return self.forward(predictions, targets)


class MSELoss(Loss):
    """均方误差损失"""

    def forward(self, predictions, targets):
        """
        predictions: 预测值张量
        targets: 目标值张量
        """
        targets = targets if isinstance(targets, Tensor) else Tensor(targets, device=predictions.device)
        diff = predictions - targets
        return (diff ** 2).mean()

    def __repr__(self):
        return "MSELoss()"


class CrossEntropyLoss(Loss):
    """交叉熵损失（改进版本）"""

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, predictions, targets):
        """
        predictions: 预测logits张量 (batch_size, num_classes)
        targets: 目标类别索引张量 (batch_size,) 或 one-hot编码 (batch_size, num_classes)
        """
        batch_size = predictions.shape[0]
        num_classes = predictions.shape[1]

        # 数值稳定的log_softmax
        max_vals = predictions.max(axis=1, keepdims=True)
        shifted_logits = predictions - max_vals
        exp_vals = shifted_logits.exp()
        sum_exp = exp_vals.sum(axis=1, keepdims=True)
        log_sum_exp = sum_exp.log()
        log_softmax = shifted_logits - log_sum_exp

        # 处理不同类型的targets
        if isinstance(targets, (list, tuple)):
            targets = Tensor(np.array(targets), device=predictions.device)
        elif isinstance(targets, np.ndarray):
            targets = Tensor(targets, device=predictions.device)

        if targets.ndim == 1:
            # targets是类别索引，需要转换为one-hot
            targets_data = targets.data.astype(int)
            one_hot = np.zeros((batch_size, num_classes))
            one_hot[np.arange(batch_size), targets_data] = 1
            targets_one_hot = Tensor(one_hot, device=predictions.device)
        else:
            # targets已经是one-hot编码
            targets_one_hot = targets

        # 计算交叉熵损失
        loss_per_sample = -(targets_one_hot * log_softmax).sum(axis=1)

        if self.reduction == 'mean':
            return loss_per_sample.mean()
        elif self.reduction == 'sum':
            return loss_per_sample.sum()
        else:  # 'none'
            return loss_per_sample

    def __repr__(self):
        return f"CrossEntropyLoss(reduction='{self.reduction}')"


# ==================== 优化器 ====================

class Optimizer:
    """优化器基类"""

    def __init__(self, parameters):
        self.parameters = list(parameters)

    def zero_grad(self):
        """清零所有参数的梯度"""
        for param in self.parameters:
            param.zero_grad()

    def step(self):
        """执行一步优化"""
        raise NotImplementedError


class SGD(Optimizer):
    """随机梯度下降优化器"""

    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # 为每个参数初始化动量缓冲区
        self.momentum_buffers = [None] * len(self.parameters)

    def step(self):
        # 正确的Momentum公式:
        # Dt = β * St-1 + (1 - β) * Wt
        # 其中: St-1 是历史梯度移动加权平均值, Wt 是当前梯度值, β 是权重系数
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            grad = param.grad.data

            # 权重衰减
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            # 动量
            if self.momentum != 0:
                if self.momentum_buffers[i] is None:
                    xp = param._get_array_module()
                    self.momentum_buffers[i] = xp.zeros_like(param.data)

                buf = self.momentum_buffers[i]
                buf = self.momentum * buf + grad
                self.momentum_buffers[i] = buf
                grad = buf

            # 更新参数
            param.data = param.data - self.lr * grad


class Adam(Optimizer):
    """Adam优化器"""

    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # 初始化动量缓冲区
        self.m_buffers = [None] * len(self.parameters)  # 一阶矩估计
        self.v_buffers = [None] * len(self.parameters)  # 二阶矩估计
        self.step_count = 0

    def step(self):
        """执行一步优化"""
        self.step_count += 1

        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            grad = param.grad.data

            # 权重衰减
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            xp = param._get_array_module()

            # 初始化缓冲区
            if self.m_buffers[i] is None:
                self.m_buffers[i] = xp.zeros_like(param.data)
                self.v_buffers[i] = xp.zeros_like(param.data)

            m, v = self.m_buffers[i], self.v_buffers[i]
            beta1, beta2 = self.betas

            # 更新有偏一阶和二阶矩估计
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)

            # 偏差校正
            m_corrected = m / (1 - beta1 ** self.step_count)
            v_corrected = v / (1 - beta2 ** self.step_count)

            # 更新参数
            param.data = param.data - self.lr * m_corrected / (xp.sqrt(v_corrected) + self.eps)

            # 保存更新后的缓冲区
            self.m_buffers[i] = m
            self.v_buffers[i] = v


# ==================== 实用函数 ====================

def init_weights(module, init_type='xavier'):
    """权重初始化"""
    if isinstance(module, Linear):
        if init_type == 'xavier':
            std = np.sqrt(2.0 / (module.in_features + module.out_features))
            module.weight.data = np.random.randn(*module.weight.shape) * std
        elif init_type == 'he':
            std = np.sqrt(2.0 / module.in_features)
            module.weight.data = np.random.randn(*module.weight.shape) * std
        elif init_type == 'normal':
            module.weight.data = np.random.randn(*module.weight.shape) * 0.01

        if module.bias is not None:
            module.bias.data = np.zeros_like(module.bias.data)


# ==================== 示例使用 ====================

if __name__ == "__main__":
    # 创建一个简单的多层感知机
    model = Sequential(
        Linear(784, 128),
        ReLU(),
        BatchNorm1d(128),
        Dropout(0.2),
        Linear(128, 64),
        ReLU(),
        Linear(64, 10),
        Sigmoid()
    )

    # 初始化权重
    for layer in model.layers:
        if isinstance(layer, Linear):
            init_weights(layer, 'xavier')

    # 创建优化器
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = MSELoss()

    # 模拟训练数据
    batch_size = 32
    input_size = 784
    output_size = 10

    # 创建随机数据
    x = randn(batch_size, input_size, requires_grad=False)
    y = randn(batch_size, output_size, requires_grad=False)

    print("模型结构:")
    print(model)
    print(f"\n总参数数量: {len(model.parameters())}")

    # 前向传播
    print("\n执行前向传播...")
    output = model(x)
    print(f"输出形状: {output.shape}")

    # 计算损失
    loss = criterion(output, y)
    print(f"损失值: {loss.data}")

    # 反向传播
    print("\n执行反向传播...")
    loss.backward()

    # 检查梯度
    print("梯度检查:")
    for i, param in enumerate(model.parameters()[:3]):  # 只检查前3个参数
        if param.grad is not None:
            grad_norm = (param.grad.data ** 2).sum() ** 0.5
            print(f"参数 {i}: 梯度范数 = {grad_norm}")

    # 优化步骤
    optimizer.step()
    optimizer.zero_grad()

    print("\n优化完成！")
