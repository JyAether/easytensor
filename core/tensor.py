import numpy as np
from typing import Union, List, Tuple, Optional
import warnings

try:
    import cupy as cp

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None


class Tensor:
    """
    深度学习基本数据类型
    多维张量类，支持自动微分和GPU加速
    """

    def __init__(self, data, requires_grad=False, device='cpu', dtype=np.float32, _children=(), _op=''):
        # 数据存储 - 支持多种输入类型
        if isinstance(data, (int, float)):
            self.data = np.array(data, dtype=dtype)
        elif isinstance(data, (list, tuple)):
            self.data = np.array(data, dtype=dtype)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(dtype)
        elif CUDA_AVAILABLE and isinstance(data, cp.ndarray):
            self.data = data.astype(dtype)
        else:
            self.data = np.array(data, dtype=dtype)

        # 设备管理
        self.device = device
        if device == 'cuda' and CUDA_AVAILABLE:
            if isinstance(self.data, np.ndarray):
                self.data = cp.asarray(self.data)
        elif device == 'cpu':
            if CUDA_AVAILABLE and isinstance(self.data, cp.ndarray):
                self.data = cp.asnumpy(self.data)

        # 梯度相关
        self.requires_grad = requires_grad
        self.grad = None

        # 计算图相关
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        """返回张量形状"""
        return self.data.shape

    @property
    def ndim(self):
        """返回张量维度数"""
        return self.data.ndim

    @property
    def size(self):
        """返回张量元素总数"""
        return self.data.size

    @property
    def dtype(self):
        """返回数据类型"""
        return self.data.dtype

    # Add these methods to your Tensor class

    def exp(self):
        """指数函数 e^x"""
        xp = self._get_array_module()
        result_data = xp.exp(self.data)
        result = Tensor(result_data, device=self.device)

        if self.requires_grad:
            def _exp_backward():
                if self.grad is None:
                    self.grad = zeros(*self.shape, device=self.device)
                if result.grad is not None:
                    # d/dx(e^x) = e^x
                    self.grad.data += result.grad.data * result.data

            result._backward = _exp_backward

        return result

    def log(self):
        """自然对数函数 ln(x)"""
        xp = self._get_array_module()
        result_data = xp.log(self.data)
        result = Tensor(result_data, device=self.device)

        if self.requires_grad:
            def _log_backward():
                if self.grad is None:
                    self.grad = zeros(*self.shape, device=self.device)
                if result.grad is not None:
                    # d/dx(ln(x)) = 1/x
                    self.grad.data += result.grad.data / self.data

            result._backward = _log_backward

        return result

    def max(self, axis=None, keepdims=False):
        epsilon = 1e-10
        """返回张量的最大值"""
        xp = self._get_array_module()

        if axis is None:
            # 返回全局最大值
            result_data = xp.max(self.data)
            result = Tensor(result_data, device=self.device)
        else:
            # 沿指定轴返回最大值
            result_data = xp.max(self.data, axis=axis, keepdims=keepdims)
            result = Tensor(result_data, device=self.device)

        # 设置梯度计算
        if self.requires_grad:
            def _max_backward():
                if self.grad is None:
                    self.grad = zeros(*self.shape, device=self.device)

                if axis is None:
                    # 全局最大值的梯度
                    max_indices = (self.data == result.data)
                    grad_data = max_indices.astype(self.dtype) / xp.sum(max_indices)
                else:
                    # 沿轴最大值的梯度
                    if keepdims:
                        expanded_result = result.data
                    else:
                        # 需要扩展维度以匹配原始形状
                        expanded_result = xp.expand_dims(result.data, axis=axis)

                    max_indices = (self.data == expanded_result)
                    # 处理可能的重复最大值
                    sum_indices = xp.sum(max_indices, axis=axis, keepdims=True)
                    grad_data = max_indices.astype(self.dtype) / (sum_indices + epsilon)

                if result.grad is not None:
                    if axis is None:
                        self.grad.data += grad_data * result.grad.data
                    else:
                        if not keepdims:
                            result_grad = xp.expand_dims(result.grad.data, axis=axis)
                        else:
                            result_grad = result.grad.data
                        self.grad.data += grad_data * result_grad

            result._backward = _max_backward

        return result

    def min(self, axis=None, keepdims=False):
        """返回张量的最小值"""
        xp = self._get_array_module()

        if axis is None:
            result_data = xp.min(self.data)
            result = Tensor(result_data, device=self.device)
        else:
            result_data = xp.min(self.data, axis=axis, keepdims=keepdims)
            result = Tensor(result_data, device=self.device)

        if self.requires_grad:
            def _min_backward():
                if self.grad is None:
                    self.grad = zeros(*self.shape, device=self.device)

                if axis is None:
                    min_indices = (self.data == result.data)
                    grad_data = min_indices.astype(self.dtype) / xp.sum(min_indices)
                else:
                    if keepdims:
                        expanded_result = result.data
                    else:
                        expanded_result = xp.expand_dims(result.data, axis=axis)

                    min_indices = (self.data == expanded_result)
                    sum_indices = xp.sum(min_indices, axis=axis, keepdims=True)
                    grad_data = min_indices.astype(self.dtype) / sum_indices

                if result.grad is not None:
                    if axis is None:
                        self.grad.data += grad_data * result.grad.data
                    else:
                        if not keepdims:
                            result_grad = xp.expand_dims(result.grad.data, axis=axis)
                        else:
                            result_grad = result.grad.data
                        self.grad.data += grad_data * result_grad

            result._backward = _min_backward

        return result

    def sqrt(self):
        """平方根函数"""
        xp = self._get_array_module()
        result_data = xp.sqrt(self.data)
        result = Tensor(result_data, device=self.device)

        if self.requires_grad:
            def _sqrt_backward():
                if self.grad is None:
                    self.grad = zeros(*self.shape, device=self.device)
                if result.grad is not None:
                    # d/dx(sqrt(x)) = 1/(2*sqrt(x))
                    self.grad.data += result.grad.data / (2 * result.data)

            result._backward = _sqrt_backward

        return result

    def abs(self):
        """绝对值函数"""
        xp = self._get_array_module()
        result_data = xp.abs(self.data)
        result = Tensor(result_data, device=self.device)

        if self.requires_grad:
            def _abs_backward():
                if self.grad is None:
                    self.grad = zeros(*self.shape, device=self.device)
                if result.grad is not None:
                    # d/dx(|x|) = sign(x), but 0 at x=0
                    sign_data = xp.sign(self.data)
                    self.grad.data += result.grad.data * sign_data

            result._backward = _abs_backward

        return result

    def sin(self):
        """正弦函数"""
        xp = self._get_array_module()
        result_data = xp.sin(self.data)
        result = Tensor(result_data, device=self.device)

        if self.requires_grad:
            def _sin_backward():
                if self.grad is None:
                    self.grad = zeros(*self.shape, device=self.device)
                if result.grad is not None:
                    # d/dx(sin(x)) = cos(x)
                    self.grad.data += result.grad.data * xp.cos(self.data)

            result._backward = _sin_backward

        return result

    def cos(self):
        """余弦函数"""
        xp = self._get_array_module()
        result_data = xp.cos(self.data)
        result = Tensor(result_data, device=self.device)

        if self.requires_grad:
            def _cos_backward():
                if self.grad is None:
                    self.grad = zeros(*self.shape, device=self.device)
                if result.grad is not None:
                    # d/dx(cos(x)) = -sin(x)
                    self.grad.data += result.grad.data * (-xp.sin(self.data))

            result._backward = _cos_backward

        return result

    def argmax(self, axis=None, keepdims=False):
        """返回张量最大值的索引"""
        xp = self._get_array_module()
        result_data = xp.argmax(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result_data, device=self.device, requires_grad=False)

    def argmin(self, axis=None, keepdims=False):
        """返回张量最小值的索引"""
        xp = self._get_array_module()
        result_data = xp.argmin(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result_data, device=self.device, requires_grad=False)

    def clip(self, min_val=None, max_val=None):
        """裁剪张量值到指定范围"""
        xp = self._get_array_module()
        result_data = xp.clip(self.data, min_val, max_val)
        result = Tensor(result_data, device=self.device)

        if self.requires_grad:
            def _clip_backward():
                if self.grad is None:
                    self.grad = zeros(*self.shape, device=self.device)
                if result.grad is not None:
                    # 梯度只在未被裁剪的地方传递
                    mask = (self.data >= (min_val if min_val is not None else -float('inf'))) & \
                           (self.data <= (max_val if max_val is not None else float('inf')))
                    self.grad.data += result.grad.data * mask.astype(self.dtype)

            result._backward = _clip_backward

        return result

    # def argmax(self, axis=None, keepdims=False):
    #     """返回张量最大值的索引"""
    #     xp = self._get_array_module()
    #     result_data = xp.argmax(self.data, axis=axis, keepdims=keepdims)
    #     return Tensor(result_data, device=self.device, requires_grad=False)

    def numpy(self):
        """转换为numpy数组"""
        if CUDA_AVAILABLE and isinstance(self.data, cp.ndarray):
            return cp.asnumpy(self.data)
        return self.data.copy()

    def cuda(self):
        """移动到GPU"""
        if not CUDA_AVAILABLE:
            warnings.warn("CUDA not available, tensor remains on CPU")
            return self

        new_tensor = Tensor(
            cp.asarray(self.data),
            requires_grad=self.requires_grad,
            device='cuda',
            _children=self._prev,
            _op=self._op
        )
        new_tensor.grad = self.grad
        new_tensor._backward = self._backward
        return new_tensor

    def cpu(self):
        """移动到CPU"""
        if CUDA_AVAILABLE and isinstance(self.data, cp.ndarray):
            data = cp.asnumpy(self.data)
        else:
            data = self.data

        new_tensor = Tensor(
            data,
            requires_grad=self.requires_grad,
            device='cpu',
            _children=self._prev,
            _op=self._op
        )
        new_tensor.grad = self.grad
        new_tensor._backward = self._backward
        return new_tensor

    def to(self, device):
        """移动到指定设备"""
        if device == 'cuda':
            return self.cuda()
        elif device == 'cpu':
            return self.cpu()
        else:
            raise ValueError(f"Unsupported device: {device}")

    def _get_array_module(self):
        """获取对应的数组模块 (numpy 或 cupy)"""
        if CUDA_AVAILABLE and isinstance(self.data, cp.ndarray):
            return cp
        return np

    def _ensure_tensor(self, other):
        """确保other是Tensor对象"""
        if not isinstance(other, Tensor):
            return Tensor(other, device=self.device, dtype=self.dtype)
        return other

    def _init_grad_if_needed(self):
        """如果需要梯度但grad为None，则初始化"""
        if self.requires_grad and self.grad is None:
            xp = self._get_array_module()
            self.grad = Tensor(xp.zeros_like(self.data), device=self.device)

    # ==================== 基础数学运算 ====================

    def __add__(self, other):
        """加法运算"""
        other = self._ensure_tensor(other)
        xp = self._get_array_module()

        # 广播处理
        try:
            result_data = self.data + other.data
        except ValueError as e:
            raise ValueError(f"Cannot broadcast shapes {self.shape} and {other.shape}: {e}")

        out = Tensor(
            result_data,
            requires_grad=self.requires_grad or other.requires_grad,
            device=self.device,
            _children=(self, other),
            _op='+'
        )

        def _backward():
            if self.requires_grad:
                self._init_grad_if_needed()
                # 处理广播的反向传播
                grad = out.grad.data
                # 如果形状不同，需要求和并reshape
                if self.shape != out.shape:
                    # 计算需要求和的轴
                    ndims_added = out.ndim - self.ndim
                    # 先对新增的维度求和
                    for _ in range(ndims_added):
                        grad = xp.sum(grad, axis=0)
                    # 再对广播的维度求和
                    for i, (dim_self, dim_out) in enumerate(zip(self.shape, grad.shape)):
                        if dim_self == 1 and dim_out > 1:
                            grad = xp.sum(grad, axis=i, keepdims=True)

                self.grad.data += grad

            if other.requires_grad:
                other._init_grad_if_needed()
                grad = out.grad.data
                if other.shape != out.shape:
                    ndims_added = out.ndim - other.ndim
                    for _ in range(ndims_added):
                        grad = xp.sum(grad, axis=0)
                    for i, (dim_other, dim_out) in enumerate(zip(other.shape, grad.shape)):
                        if dim_other == 1 and dim_out > 1:
                            grad = xp.sum(grad, axis=i, keepdims=True)

                other.grad.data += grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        """逐元素乘法"""
        other = self._ensure_tensor(other)
        xp = self._get_array_module()

        result_data = self.data * other.data
        out = Tensor(
            result_data,
            requires_grad=self.requires_grad or other.requires_grad,
            device=self.device,
            _children=(self, other),
            _op='*'
        )

        def _backward():
            if self.requires_grad:
                self._init_grad_if_needed()
                grad = other.data * out.grad.data
                # 处理广播
                if self.shape != out.shape:
                    ndims_added = out.ndim - self.ndim
                    for _ in range(ndims_added):
                        grad = xp.sum(grad, axis=0)
                    for i, (dim_self, dim_out) in enumerate(zip(self.shape, grad.shape)):
                        if dim_self == 1 and dim_out > 1:
                            grad = xp.sum(grad, axis=i, keepdims=True)
                self.grad.data += grad

            if other.requires_grad:
                other._init_grad_if_needed()
                grad = self.data * out.grad.data
                if other.shape != out.shape:
                    ndims_added = out.ndim - other.ndim
                    for _ in range(ndims_added):
                        grad = xp.sum(grad, axis=0)
                    for i, (dim_other, dim_out) in enumerate(zip(other.shape, grad.shape)):
                        if dim_other == 1 and dim_out > 1:
                            grad = xp.sum(grad, axis=i, keepdims=True)
                other.grad.data += grad

        out._backward = _backward
        return out

    def matmul(self, other):
        """矩阵乘法"""
        other = self._ensure_tensor(other)
        xp = self._get_array_module()

        # 检查维度兼容性
        if self.ndim < 2 or other.ndim < 2:
            raise ValueError(
                f"matmul requires both tensors to have at least 2 dimensions, got {self.ndim} and {other.ndim}")

        if self.shape[-1] != other.shape[-2]:
            raise ValueError(f"matmul dimension mismatch: {self.shape[-1]} != {other.shape[-2]}")

        result_data = xp.matmul(self.data, other.data)
        out = Tensor(
            result_data,
            requires_grad=self.requires_grad or other.requires_grad,
            device=self.device,
            _children=(self, other),
            _op='@'
        )

        def _backward():
            if self.requires_grad:
                self._init_grad_if_needed()
                # self.grad += out.grad @ other.T
                grad = xp.matmul(out.grad.data, xp.swapaxes(other.data, -2, -1))
                self.grad.data += grad

            if other.requires_grad:
                other._init_grad_if_needed()
                # other.grad += self.T @ out.grad
                grad = xp.matmul(xp.swapaxes(self.data, -2, -1), out.grad.data)
                other.grad.data += grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        """矩阵乘法操作符 @"""
        return self.matmul(other)

    # ==================== 形状操作 ====================

    def reshape(self, *shape):
        """改变张量形状"""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]

        xp = self._get_array_module()
        result_data = self.data.reshape(shape)

        out = Tensor(
            result_data,
            requires_grad=self.requires_grad,
            device=self.device,
            _children=(self,),
            _op='reshape'
        )

        def _backward():
            if self.requires_grad:
                self._init_grad_if_needed()
                self.grad.data += out.grad.data.reshape(self.shape)

        out._backward = _backward
        return out

    def transpose(self, *axes):
        """转置张量"""
        xp = self._get_array_module()
        if len(axes) == 0:
            # 默认转置最后两个维度
            if self.ndim < 2:
                raise ValueError("transpose requires at least 2 dimensions")
            axes = list(range(self.ndim))
            axes[-2], axes[-1] = axes[-1], axes[-2]
        elif len(axes) == 1:
            axes = axes[0]

        result_data = xp.transpose(self.data, axes)

        out = Tensor(
            result_data,
            requires_grad=self.requires_grad,
            device=self.device,
            _children=(self,),
            _op='transpose'
        )

        def _backward():
            if self.requires_grad:
                self._init_grad_if_needed()
                # 反向转置
                inv_axes = [0] * len(axes)
                for i, ax in enumerate(axes):
                    inv_axes[ax] = i
                self.grad.data += xp.transpose(out.grad.data, inv_axes)

        out._backward = _backward
        return out

    @property
    def T(self):
        """转置属性"""
        return self.transpose()

    def sum(self, axis=None, keepdims=False):
        """求和操作"""
        xp = self._get_array_module()
        result_data = xp.sum(self.data, axis=axis, keepdims=keepdims)

        out = Tensor(
            result_data,
            requires_grad=self.requires_grad,
            device=self.device,
            _children=(self,),
            _op='sum'
        )

        def _backward():
            if self.requires_grad:
                self._init_grad_if_needed()
                grad = out.grad.data
                if axis is not None:
                    if not keepdims:
                        # 需要恢复被求和的维度
                        if isinstance(axis, int):
                            grad = xp.expand_dims(grad, axis)
                        else:
                            for ax in sorted(axis):
                                grad = xp.expand_dims(grad, ax)
                    # 广播到原始形状
                    grad = xp.broadcast_to(grad, self.shape)
                else:
                    # 全局求和的情况
                    grad = xp.full(self.shape, grad.item())

                self.grad.data += grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        """均值操作"""
        xp = self._get_array_module()
        result_data = xp.mean(self.data, axis=axis, keepdims=keepdims)

        out = Tensor(
            result_data,
            requires_grad=self.requires_grad,
            device=self.device,
            _children=(self,),
            _op='mean'
        )

        def _backward():
            if self.requires_grad:
                self._init_grad_if_needed()
                grad = out.grad.data

                # 计算均值的反向传播需要除以元素个数
                if axis is None:
                    count = self.size
                    grad = xp.full(self.shape, grad.item() / count)
                else:
                    if isinstance(axis, int):
                        count = self.shape[axis]
                        if not keepdims:
                            grad = xp.expand_dims(grad, axis)
                    else:
                        count = 1
                        for ax in axis:
                            count *= self.shape[ax]
                        if not keepdims:
                            for ax in sorted(axis):
                                grad = xp.expand_dims(grad, ax)

                    grad = xp.broadcast_to(grad, self.shape) / count

                self.grad.data += grad

        out._backward = _backward
        return out

    # ==================== 激活函数 ====================

    def relu(self):
        """ReLU激活函数"""
        xp = self._get_array_module()
        result_data = xp.maximum(0, self.data)

        out = Tensor(
            result_data,
            requires_grad=self.requires_grad,
            device=self.device,
            _children=(self,),
            _op='relu'
        )

        def _backward():
            if self.requires_grad:
                self._init_grad_if_needed()
                mask = (self.data > 0).astype(self.dtype)
                self.grad.data += mask * out.grad.data

        out._backward = _backward
        return out

    def sigmoid(self):
        """Sigmoid激活函数"""
        xp = self._get_array_module()
        # 数值稳定版本
        stable_data = xp.clip(self.data, -500, 500)
        result_data = 1 / (1 + xp.exp(-stable_data))

        out = Tensor(
            result_data,
            requires_grad=self.requires_grad,
            device=self.device,
            _children=(self,),
            _op='sigmoid'
        )

        def _backward():
            if self.requires_grad:
                self._init_grad_if_needed()
                sigmoid_grad = result_data * (1 - result_data)
                self.grad.data += sigmoid_grad * out.grad.data

        out._backward = _backward
        return out

    def tanh(self):
        """Tanh激活函数"""
        xp = self._get_array_module()
        result_data = xp.tanh(self.data)

        out = Tensor(
            result_data,
            requires_grad=self.requires_grad,
            device=self.device,
            _children=(self,),
            _op='tanh'
        )

        def _backward():
            if self.requires_grad:
                self._init_grad_if_needed()
                tanh_grad = 1 - result_data ** 2
                self.grad.data += tanh_grad * out.grad.data

        out._backward = _backward
        return out

    # ==================== 反向传播 ====================

    def backward(self):
        """反向传播"""
        if not self.requires_grad:
            raise RuntimeError("Tensor does not require gradients")

        # 拓扑排序
        topo = []
        visited = set()

        def build_topological_order(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topological_order(child)
                topo.append(v)

        build_topological_order(self)

        # 初始化输出梯度
        xp = self._get_array_module()
        self.grad = Tensor(xp.ones_like(self.data), device=self.device)

        # 反向传播
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        """清零梯度"""
        if self.grad is not None:
            xp = self._get_array_module()
            self.grad.data = xp.zeros_like(self.grad.data)

    # ==================== 其他运算符重载 ====================

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-self._ensure_tensor(other))

    def __rsub__(self, other):
        return self._ensure_tensor(other) + (-self)

    def __truediv__(self, other):
        other = self._ensure_tensor(other)
        return self * (other ** -1)

    def __rtruediv__(self, other):
        other = self._ensure_tensor(other)
        return other * (self ** -1)

    def __pow__(self, power):
        xp = self._get_array_module()
        result_data = xp.power(self.data, power)

        out = Tensor(
            result_data,
            requires_grad=self.requires_grad,
            device=self.device,
            _children=(self,),
            _op=f'**{power}'
        )

        def _backward():
            if self.requires_grad:
                self._init_grad_if_needed()
                grad = power * xp.power(self.data, power - 1) * out.grad.data
                self.grad.data += grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __repr__(self):
        # return f"Tensor(data={self.data},grad={self.grad},shape={self.shape}, dtype={self.dtype}, device={self.device}, requires_grad={self.requires_grad})\n{self.data}"
        return f"\nNode(当前的权重={self.data},grad={self.grad.data}"

    def __str__(self):
        return self.__repr__()


# ==================== 便利函数 ====================

def tensor(data, requires_grad=False, device='cpu', dtype=np.float32):
    """创建张量的便利函数"""
    return Tensor(data, requires_grad=requires_grad, device=device, dtype=dtype)


def zeros(*shape, requires_grad=False, device='cpu', dtype=np.float32):
    """创建全零张量"""
    if CUDA_AVAILABLE and device == 'cuda':
        data = cp.zeros(shape, dtype=dtype)
    else:
        data = np.zeros(shape, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad, device=device)


def ones(*shape, requires_grad=False, device='cpu', dtype=np.float32):
    """创建全一张量"""
    if CUDA_AVAILABLE and device == 'cuda':
        data = cp.ones(shape, dtype=dtype)
    else:
        data = np.ones(shape, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad, device=device)


def randn(*shape, requires_grad=False, device='cpu', dtype=np.float32):
    """创建正态分布随机张量"""
    if CUDA_AVAILABLE and device == 'cuda':
        data = cp.random.randn(*shape).astype(dtype)
    else:
        data = np.random.randn(*shape).astype(dtype)
    return Tensor(data, requires_grad=requires_grad, device=device)


def eye(n, requires_grad=False, device='cpu', dtype=np.float32):
    """创建单位矩阵"""
    if CUDA_AVAILABLE and device == 'cuda':
        data = cp.eye(n, dtype=dtype)
    else:
        data = np.eye(n, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad, device=device)
