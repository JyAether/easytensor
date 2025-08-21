from core.v1.optim.optimizer import Optimizer
import numpy as np

class Adam(Optimizer):
    """Adam优化器"""

    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0  # 时间步数

        # 为每个参数初始化一阶和二阶动量估计
        self.m = [0.0 for _ in self.parameters]  # 一阶动量估计
        self.v = [0.0 for _ in self.parameters]  # 二阶动量估计

    def step(self):
        """执行一步优化"""
        self.t += 1

        for i, param in enumerate(self.parameters):
            # 更新有偏一阶动量估计: m = beta1 * m + (1-beta1) * grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad

            # 更新有偏二阶动量估计: v = beta2 * v + (1-beta2) * grad^2
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)

            # 计算偏差修正的一阶动量估计: m_hat = m / (1-beta1^t)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)

            # 计算偏差修正的二阶动量估计: v_hat = v / (1-beta2^t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # 更新参数: param = param - lr * m_hat / (sqrt(v_hat) + eps)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)