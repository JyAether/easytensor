from core.v1.optim.optimizer import Optimizer


class SGD(Optimizer):
    """随机梯度下降优化器（支持动量,非标准动量法）"""

    def __init__(self, parameters, lr=0.01, momentum=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum  # β 权重系数
        # 为每个参数初始化历史梯度移动加权平均值 St-1
        self.momentum_buffer = [0.0 for _ in self.parameters]

    def step(self):
        """执行一步优化"""
        for i, param in enumerate(self.parameters):
            # 正确的Momentum公式:
            # Dt = β * St-1 + (1 - β) * Wt
            # 其中: St-1 是历史梯度移动加权平均值, Wt 是当前梯度值, β 是权重系数

            current_grad = param.grad  # Wt: 当前时刻的梯度值

            # 计算当前时刻的指数加权平均梯度值 Dt
            self.momentum_buffer[i] = (self.momentum * self.momentum_buffer[i] +
                                       (1 - self.momentum) * current_grad)

            # 使用加权平均梯度更新参数: param = param - lr * Dt
            param.data -= self.lr * self.momentum_buffer[i]