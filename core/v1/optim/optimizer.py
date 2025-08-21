class Optimizer:
    """优化器基类"""

    def __init__(self, parameters):
        self.parameters = list(parameters)

    def zero_grad(self):
        """清零梯度"""
        for param in self.parameters:
            param.grad = 0

    def step(self):
        """更新参数，子类需要实现"""
        raise NotImplementedError
