# 为您的Tensor类添加的扩展方法
# 这些方法需要集成到您的core.tensor.Tensor类中

import numpy as np


def clip_method(self, min_val=None, max_val=None):
    """
    将张量的值限制在指定范围内

    Args:
        min_val: 最小值
        max_val: 最大值

    Returns:
        Tensor: 限制后的张量
    """
    from core.tensor import Tensor  # 假设您的Tensor类在这里

    clipped_data = self.data.copy()

    if min_val is not None:
        clipped_data = np.maximum(clipped_data, min_val)
    if max_val is not None:
        clipped_data = np.minimum(clipped_data, max_val)

    result = Tensor(clipped_data, requires_grad=self.requires_grad, device=self.device)

    if self.requires_grad:
        def clip_backward():
            if self.grad is None:
                self.grad = Tensor(np.zeros_like(self.data), device=self.device)

            # 计算梯度掩码
            mask = np.ones_like(self.data, dtype=bool)
            if min_val is not None:
                mask &= (self.data >= min_val)
            if max_val is not None:
                mask &= (self.data <= max_val)

            self.grad.data += result.grad.data * mask

        result._backward = clip_backward

    return result


def log_method(self):
    """
    计算张量的自然对数

    Returns:
        Tensor: 对数结果
    """
    from core.tensor import Tensor

    log_data = np.log(self.data)
    result = Tensor(log_data, requires_grad=self.requires_grad, device=self.device)

    if self.requires_grad:
        def log_backward():
            if self.grad is None:
                self.grad = Tensor(np.zeros_like(self.data), device=self.device)
            self.grad.data += result.grad.data / self.data

        result._backward = log_backward

    return result


def log1p_method(self):
    """
    计算log(1 + x)，数值稳定版本

    Returns:
        Tensor: log(1+x)结果
    """
    from core.tensor import Tensor

    log1p