from core.v1.engine import Node
class BatchNorm:
    """一维批归一化层

    对输入进行批归一化：
    1. 计算批次均值和方差
    2. 标准化：(x - mean) / sqrt(var + eps)
    3. 缩放和偏移：gamma * x_norm + beta

    训练时使用当前批次统计量，推理时使用滑动平均统计量
    """

    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        """
        参数:
            num_features: 输入特征的数量（通常是神经元个数或通道数）
            momentum: 滑动平均的动量，用于更新running_mean和running_var
            eps: 数值稳定性常数，防止除零
        """
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.training = True

        # 可学习参数：缩放参数γ和偏移参数β
        self.gamma = [Node(1.0) for _ in range(num_features)]  # 缩放参数，初始化为1
        self.beta = [Node(0.0) for _ in range(num_features)]  # 偏移参数，初始化为0

        # 滑动平均的均值和方差（不参与梯度计算）
        self.running_mean = [0.0 for _ in range(num_features)]
        self.running_var = [1.0 for _ in range(num_features)]

        # 统计信息（用于调试）
        self.num_batches_tracked = 0

    def __call__(self, x):
        """
        前向传播

        参数:
            x: 输入数据
               - 对于全连接层：[[sample1_features], [sample2_features], ...]
               - 对于单个样本：[feature1, feature2, ...]

        返回:
            归一化后的数据，格式与输入相同
        """
        # 处理单个样本的情况
        if isinstance(x[0], Node):
            # 单个样本：[feature1, feature2, ...]
            x = [x]
            single_sample = True
        else:
            # 批次数据：[[sample1], [sample2], ...]
            single_sample = False

        batch_size = len(x)

        if self.training and batch_size > 1:
            # 训练模式：使用批次统计量
            batch_mean, batch_var = self._compute_batch_stats(x)

            # 更新滑动平均（使用数值，不参与梯度计算）
            self._update_running_stats(batch_mean, batch_var)

            # 使用批次统计量进行归一化
            normalized = self._normalize(x, batch_mean, batch_var)

        else:
            # 推理模式或单个样本：使用滑动平均统计量
            running_mean = [Node(m) for m in self.running_mean]
            running_var = [Node(v) for v in self.running_var]

            normalized = self._normalize(x, running_mean, running_var)

        # 如果输入是单个样本，返回单个样本
        return normalized[0] if single_sample else normalized

    def _compute_batch_stats(self, x):
        """计算批次的均值和方差"""
        batch_size = len(x)

        # 计算批次均值：E[x] = (1/N) * Σx_i
        batch_mean = []
        for j in range(self.num_features):
            feature_sum = Node(0.0)
            for sample in x:
                feature_sum = feature_sum + sample[j]
            mean = feature_sum * (1.0 / batch_size)
            batch_mean.append(mean)

        # 计算批次方差：Var[x] = (1/N) * Σ(x_i - μ)²
        batch_var = []
        for j in range(self.num_features):
            var_sum = Node(0.0)
            for sample in x:
                diff = sample[j] - batch_mean[j]
                var_sum = var_sum + diff * diff
            var = var_sum * (1.0 / batch_size)
            batch_var.append(var)

        return batch_mean, batch_var

    def _update_running_stats(self, batch_mean, batch_var):
        """更新滑动平均统计量"""
        self.num_batches_tracked += 1

        for j in range(self.num_features):
            # running_mean = (1-momentum) * running_mean + momentum * batch_mean
            self.running_mean[j] = ((1 - self.momentum) * self.running_mean[j] +
                                    self.momentum * batch_mean[j].data)

            # running_var = (1-momentum) * running_var + momentum * batch_var
            self.running_var[j] = ((1 - self.momentum) * self.running_var[j] +
                                   self.momentum * batch_var[j].data)

    def _normalize(self, x, mean, var):
        """执行归一化和缩放"""
        normalized = []

        for sample in x:
            norm_sample = []
            for j in range(self.num_features):
                # 标准化：(x - μ) / σ
                std = (var[j] + self.eps) ** 0.5  # σ = √(var + ε)
                x_norm = (sample[j] - mean[j]) / std

                # 缩放和偏移：γ * x_norm + β
                y = self.gamma[j] * x_norm + self.beta[j]
                norm_sample.append(y)

            normalized.append(norm_sample)

        return normalized

    def parameters(self):
        """返回可学习参数"""
        return self.gamma + self.beta

    def train(self):
        """设置为训练模式"""
        self.training = True

    def eval(self):
        """设置为评估模式"""
        self.training = False

    def zero_grad(self):
        """梯度清零"""
        for param in self.parameters():
            param.grad = 0

    def extra_repr(self):
        """返回层的字符串表示"""
        return f'{self.num_features}, eps={self.eps}, momentum={self.momentum}'

    def __repr__(self):
        return f"BatchNorm({self.extra_repr()})"