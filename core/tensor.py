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
                    # d/dx(|x|) = sign(x), 但是如果 0 是 x=0
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

    def expand(self, *sizes):
        """
        扩展张量到指定大小（不复制数据，使用广播）

        Args:
            *sizes: 目标大小

        Returns:
            Tensor: 扩展后的张量

        Example:
            >>> x = Tensor([[1], [2]])  # shape: (2, 1)
            >>> y = x.expand(2, 3)      # shape: (2, 3)
        """
        if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
            sizes = sizes[0]

        xp = self._get_array_module()

        # 检查扩展的有效性
        if len(sizes) < self.ndim:
            raise ValueError(f"expand: target size ({len(sizes)}) must be >= source size ({self.ndim})")

        # 对齐维度（在前面补1）
        expanded_shape = list(sizes)
        original_shape = list(self.shape)


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
            # 首先确保out有梯度
            if out.grad is None:
                return  # 如果输出没有梯度，直接返回

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
        """矩阵乘法 - 支持各种维度组合"""
        other = self._ensure_tensor(other)
        xp = self._get_array_module()

        # 记录原始形状用于反向传播
        self_original_shape = self.shape
        other_original_shape = other.shape

        # 处理不同的维度组合
        if self.ndim == 1 and other.ndim == 1:
            # 向量点积
            if self.shape[0] != other.shape[0]:
                raise ValueError(f"Vector dot product dimension mismatch: {self.shape[0]} != {other.shape[0]}")
            result_data = xp.dot(self.data, other.data)

        elif self.ndim == 1 and other.ndim == 2:
            # 行向量 × 矩阵: (n,) × (n, m) -> (m,)
            if self.shape[0] != other.shape[0]:
                raise ValueError(f"matmul dimension mismatch: {self.shape[0]} != {other.shape[0]}")
            result_data = xp.dot(self.data, other.data)

        elif self.ndim == 2 and other.ndim == 1:
            # 矩阵 × 列向量: (m, n) × (n,) -> (m,)
            if self.shape[1] != other.shape[0]:
                raise ValueError(f"matmul dimension mismatch: {self.shape[1]} != {other.shape[0]}")
            result_data = xp.dot(self.data, other.data)

        elif self.ndim >= 2 and other.ndim >= 2:
            # 标准矩阵乘法
            if self.shape[-1] != other.shape[-2]:
                raise ValueError(f"matmul dimension mismatch: {self.shape[-1]} != {other.shape[-2]}")
            result_data = xp.matmul(self.data, other.data)

        else:
            raise ValueError(f"Unsupported matmul dimensions: {self.ndim}D @ {other.ndim}D")

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

                # 根据原始维度组合计算梯度
                if len(self_original_shape) == 1 and len(other_original_shape) == 1:
                    # 向量点积的梯度
                    self.grad.data += out.grad.data * other.data

                elif len(self_original_shape) == 1 and len(other_original_shape) == 2:
                    # 行向量 × 矩阵的梯度: grad_self = out.grad @ other.T
                    grad = xp.dot(out.grad.data, other.data.T)
                    self.grad.data += grad

                elif len(self_original_shape) == 2 and len(other_original_shape) == 1:
                    # 矩阵 × 列向量的梯度: grad_self = out.grad @ other.T (列向量转行向量)
                    grad = xp.outer(out.grad.data, other.data)
                    self.grad.data += grad

                else:
                    # 标准矩阵乘法的梯度
                    grad = xp.matmul(out.grad.data, xp.swapaxes(other.data, -2, -1))
                    self.grad.data += grad

            if other.requires_grad:
                other._init_grad_if_needed()

                # 根据原始维度组合计算梯度
                if len(self_original_shape) == 1 and len(other_original_shape) == 1:
                    # 向量点积的梯度
                    other.grad.data += out.grad.data * self.data

                elif len(self_original_shape) == 1 and len(other_original_shape) == 2:
                    # 行向量 × 矩阵的梯度: grad_other = self.T @ out.grad (转置后外积)
                    grad = xp.outer(self.data, out.grad.data)
                    other.grad.data += grad

                elif len(self_original_shape) == 2 and len(other_original_shape) == 1:
                    # 矩阵 × 列向量的梯度: grad_other = self.T @ out.grad
                    grad = xp.dot(self.data.T, out.grad.data)
                    other.grad.data += grad

                else:
                    # 标准矩阵乘法的梯度
                    grad = xp.matmul(xp.swapaxes(self.data, -2, -1), out.grad.data)
                    other.grad.data += grad

        out._backward = _backward
        return out

    # def matmul(self, other):
    #     """矩阵乘法"""
    #     other = self._ensure_tensor(other)
    #     xp = self._get_array_module()
    #
    #     # 检查维度兼容性
    #     if self.ndim < 2 or other.ndim < 2:
    #         raise ValueError(
    #             f"matmul requires both tensors to have at least 2 dimensions, got {self.ndim} and {other.ndim}")
    #
    #     if self.shape[-1] != other.shape[-2]:
    #         raise ValueError(f"matmul dimension mismatch: {self.shape[-1]} != {other.shape[-2]}")
    #
    #     result_data = xp.matmul(self.data, other.data)
    #     out = Tensor(
    #         result_data,
    #         requires_grad=self.requires_grad or other.requires_grad,
    #         device=self.device,
    #         _children=(self, other),
    #         _op='@'
    #     )
    #
    #     def _backward():
    #         if self.requires_grad:
    #             self._init_grad_if_needed()
    #             # self.grad += out.grad @ other.T
    #             grad = xp.matmul(out.grad.data, xp.swapaxes(other.data, -2, -1))
    #             self.grad.data += grad
    #
    #         if other.requires_grad:
    #             other._init_grad_if_needed()
    #             # other.grad += self.T @ out.grad
    #             grad = xp.matmul(xp.swapaxes(self.data, -2, -1), out.grad.data)
    #             other.grad.data += grad
    #
    #     out._backward = _backward
    #     return out

    def __matmul__(self, other):
        """矩阵乘法操作符 @"""
        return self.matmul(other)

    def __getitem__(self, key):
        """张量索引操作"""
        xp = self._get_array_module()
        result_data = self.data[key]

        out = Tensor(
            result_data,
            requires_grad=self.requires_grad,
            device=self.device,
            _children=(self,),
            _op='getitem'
        )

        if self.requires_grad:
            def _getitem_backward():
                if self.grad is None:
                    self.grad = zeros(*self.shape, device=self.device)
                if out.grad is not None:
                    grad_data = xp.zeros_like(self.data)
                    grad_data[key] += out.grad.data
                    self.grad.data += grad_data

            out._backward = _getitem_backward

        return out

    def softmax(self, dim=-1):
        """
        Softmax激活函数
        softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j

        Args:
            dim: 计算softmax的维度，默认为-1（最后一个维度）
        """
        xp = self._get_array_module()

        # 为了数值稳定性，减去最大值
        max_vals = self.max(axis=dim, keepdims=True)
        shifted = self - max_vals

        # 计算指数
        exp_vals = shifted.exp()

        # 计算softmax
        sum_exp = exp_vals.sum(axis=dim, keepdims=True)
        result = exp_vals / sum_exp

        # 设置梯度计算
        if self.requires_grad:
            def _softmax_backward():
                if self.grad is None:
                    self.grad = zeros(*self.shape, device=self.device)
                if result.grad is not None:
                    # softmax的梯度: softmax * (grad - sum(softmax * grad))
                    # 沿着softmax维度计算
                    softmax_grad_sum = (result.data * result.grad.data).sum(axis=dim, keepdims=True)
                    grad_input = result.data * (result.grad.data - softmax_grad_sum)
                    self.grad.data += grad_input

            result._backward = _softmax_backward

        return result

    def log_softmax(self, dim=-1):
        """
        Log-Softmax函数，数值更稳定的log(softmax(x))
        log_softmax(x_i) = x_i - log(sum(exp(x_j))) for all j

        Args:
            dim: 计算log_softmax的维度，默认为-1（最后一个维度）
        """
        xp = self._get_array_module()

        # 为了数值稳定性，减去最大值
        max_vals = self.max(axis=dim, keepdims=True)
        shifted = self - max_vals

        # 计算log_sum_exp
        exp_vals = shifted.exp()
        sum_exp = exp_vals.sum(axis=dim, keepdims=True)
        log_sum_exp = sum_exp.log()

        # log_softmax = shifted - log_sum_exp
        result = shifted - log_sum_exp

        if self.requires_grad:
            def _log_softmax_backward():
                if self.grad is None:
                    self.grad = zeros(*self.shape, device=self.device)
                if result.grad is not None:
                    # log_softmax的梯度: grad - softmax * sum(grad)
                    softmax_vals = exp_vals / sum_exp
                    grad_sum = result.grad.data.sum(axis=dim, keepdims=True)
                    grad_input = result.grad.data - softmax_vals.data * grad_sum
                    self.grad.data += grad_input

            result._backward = _log_softmax_backward

        return result


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

    def transpose(self, dim0=None, dim1=None):
        """
        转置张量

        Args:
            dim0: 第一个要交换的维度，如果为None则转置所有维度
            dim1: 第二个要交换的维度，只在dim0不为None时使用

        Returns:
            转置后的张量

        Examples:
            # 交换两个指定维度（类似PyTorch）
            tensor.transpose(1, 0)  # 交换维度1和0

            # 转置所有维度（默认行为）
            tensor.transpose()  # 完全转置
        """
        xp = self._get_array_module()

        if dim0 is None and dim1 is None:
            # 默认转置最后两个维度
            if self.ndim < 2:
                raise ValueError("transpose requires at least 2 dimensions")
            axes = list(range(self.ndim))
            axes[-2], axes[-1] = axes[-1], axes[-2]
        elif dim0 is not None and dim1 is not None:
            # PyTorch风格：只交换两个指定的维度
            # 转换负索引
            dim0 = dim0 if dim0 >= 0 else dim0 + self.ndim
            dim1 = dim1 if dim1 >= 0 else dim1 + self.ndim

            # 检查维度有效性
            if dim0 >= self.ndim or dim0 < 0:
                raise ValueError(
                    f"Dimension out of range (expected to be in range of [-{self.ndim}, {self.ndim - 1}], but got {dim0 - self.ndim if dim0 >= self.ndim else dim0})")
            if dim1 >= self.ndim or dim1 < 0:
                raise ValueError(
                    f"Dimension out of range (expected to be in range of [-{self.ndim}, {self.ndim - 1}], but got {dim1 - self.ndim if dim1 >= self.ndim else dim1})")

            # 创建轴序列，只交换指定的两个维度
            axes = list(range(self.ndim))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        else:
            raise ValueError("transpose() takes either 0 or 2 positional arguments")

        axes = tuple(axes)
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
                # 反向转置：再次应用相同的转置来恢复原始形状
                self.grad.data += xp.transpose(out.grad.data, axes)

        out._backward = _backward
        return out

    # def transpose(self, *axes):
    #     """转置张量"""
    #     xp = self._get_array_module()
    #
    #     if len(axes) == 0:
    #         # 默认转置最后两个维度
    #         if self.ndim < 2:
    #             raise ValueError("transpose requires at least 2 dimensions")
    #         axes = list(range(self.ndim))
    #         axes[-2], axes[-1] = axes[-1], axes[-2]
    #     elif len(axes) == 1 and hasattr(axes[0], '__iter__'):
    #         # 如果传入的是一个序列（如列表或元组）
    #         axes = axes[0]
    #     # 如果 len(axes) > 1，直接使用 axes（这是多个单独参数的情况）
    #
    #     result_data = xp.transpose(self.data, axes)
    #
    #     out = Tensor(
    #         result_data,
    #         requires_grad=self.requires_grad,
    #         device=self.device,
    #         _children=(self,),
    #         _op='transpose'
    #     )
    #
    #     def _backward():
    #         if self.requires_grad:
    #             self._init_grad_if_needed()
    #             # 反向转置
    #             inv_axes = [0] * len(axes)
    #             for i, ax in enumerate(axes):
    #                 inv_axes[ax] = i
    #             self.grad.data += xp.transpose(out.grad.data, inv_axes)
    #
    #     out._backward = _backward
    #     return out

    @property
    def T(self):
        """转置属性"""
        return self.transpose()

    def item(self):
        """返回张量中的单个标量值"""
        if self.data.size == 1:
            return self.data.item()  # 如果内部用numpy数组存储
        else:
            raise ValueError("只能对包含单个元素的张量调用item()")

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

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

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

        # ==================== 基本运算符重载 ====================
    # 为Tensor类补充的张量操作方法
    def permute(self, *dims):
        """
        重新排列张量的维度

        Args:
            *dims: 新的维度顺序

        Returns:
            Tensor: 重新排列维度后的张量

        Example:
            >>> x = Tensor([[1, 2], [3, 4]])  # shape: (2, 2)
            >>> y = x.permute(1, 0)  # shape: (2, 2), 转置
        """
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = dims[0]

        if len(dims) != self.ndim:
            raise ValueError(
                f"Number of dimensions in permute ({len(dims)}) doesn't match tensor dimensions ({self.ndim})")

        # 检查维度是否有效
        dims = list(dims)
        for i, dim in enumerate(dims):
            if dim < 0:
                dims[i] = self.ndim + dim
            if not (0 <= dims[i] < self.ndim):
                raise IndexError(f"Dimension {dim} is out of range for {self.ndim}D tensor")

        # 检查是否有重复维度
        if len(set(dims)) != len(dims):
            raise ValueError("Repeated dimension in permute")

        xp = self._get_array_module()
        result_data = xp.transpose(self.data, dims)

        out = Tensor(
            result_data,
            requires_grad=self.requires_grad,
            device=self.device,
            _children=(self,),
            _op='permute'
        )

        if self.requires_grad:
            def _permute_backward():
                if self.grad is None:
                    self.grad = zeros(*self.shape, device=self.device)
                if out.grad is not None:
                    # 反向permute：找到逆排列
                    inv_dims = [0] * len(dims)
                    for i, d in enumerate(dims):
                        inv_dims[d] = i
                    self.grad.data += xp.transpose(out.grad.data, inv_dims)

            out._backward = _permute_backward

        return out

    def squeeze(self, dim=None):
        """
        移除长度为1的维度

        Args:
            dim: 要移除的维度，如果为None则移除所有长度为1的维度

        Returns:
            Tensor: 压缩后的张量

        Example:
            >>> x = Tensor([[[1], [2]]])  # shape: (1, 2, 1)
            >>> y = x.squeeze()  # shape: (2,)
            >>> z = x.squeeze(0)  # shape: (2, 1)
        """
        xp = self._get_array_module()

        if dim is None:
            # 移除所有长度为1的维度
            result_data = xp.squeeze(self.data)
            # 记录被移除的维度用于反向传播
            squeezed_dims = [i for i, size in enumerate(self.shape) if size == 1]
        else:
            # 移除指定维度
            if dim < 0:
                dim = self.ndim + dim

            if not (0 <= dim < self.ndim):
                raise IndexError(f"Dimension {dim} is out of range for {self.ndim}D tensor")

            if self.shape[dim] != 1:
                raise RuntimeError(f"Cannot squeeze dimension {dim} with size {self.shape[dim]} != 1")

            result_data = xp.squeeze(self.data, axis=dim)
            squeezed_dims = [dim]

        out = Tensor(
            result_data,
            requires_grad=self.requires_grad,
            device=self.device,
            _children=(self,),
            _op='squeeze'
        )

        if self.requires_grad:
            def _squeeze_backward():
                if self.grad is None:
                    self.grad = zeros(*self.shape, device=self.device)
                if out.grad is not None:
                    # 恢复被squeeze的维度
                    grad_data = out.grad.data
                    for dim in sorted(squeezed_dims):
                        grad_data = xp.expand_dims(grad_data, axis=dim)
                    self.grad.data += grad_data

            out._backward = _squeeze_backward

        return out

    def unsqueeze(self, dim):
        """
        在指定位置添加长度为1的维度

        Args:
            dim: 要添加维度的位置

        Returns:
            Tensor: 添加维度后的张量

        Example:
            >>> x = Tensor([1, 2, 3])  # shape: (3,)
            >>> y = x.unsqueeze(0)  # shape: (1, 3)
            >>> z = x.unsqueeze(1)  # shape: (3, 1)
        """
        # 处理负数索引
        if dim < 0:
            dim = self.ndim + 1 + dim

        if not (0 <= dim <= self.ndim):
            raise IndexError(f"Dimension {dim} is out of range for {self.ndim}D tensor")

        xp = self._get_array_module()
        result_data = xp.expand_dims(self.data, axis=dim)

        out = Tensor(
            result_data,
            requires_grad=self.requires_grad,
            device=self.device,
            _children=(self,),
            _op='unsqueeze'
        )

        if self.requires_grad:
            def _unsqueeze_backward():
                if self.grad is None:
                    self.grad = zeros(*self.shape, device=self.device)
                if out.grad is not None:
                    # 移除添加的维度
                    self.grad.data += xp.squeeze(out.grad.data, axis=dim)

            out._backward = _unsqueeze_backward

        return out

    def view(self, *shape):
        """
        改变张量形状（类似reshape，但要求内存连续）

        Args:
            *shape: 新的形状

        Returns:
            Tensor: 重新整形后的张量

        Example:
            >>> x = Tensor([[1, 2], [3, 4]])  # shape: (2, 2)
            >>> y = x.view(4)  # shape: (4,)
            >>> z = x.view(1, 4)  # shape: (1, 4)
        """
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]

        # 计算总元素数
        total_elements = self.size

        # 处理-1（自动推断维度）
        shape = list(shape)
        auto_dim = -1
        auto_dim_count = 0

        for i, dim in enumerate(shape):
            if dim == -1:
                if auto_dim_count > 0:
                    raise ValueError("Only one dimension can be inferred (-1)")
                auto_dim = i
                auto_dim_count += 1

        if auto_dim != -1:
            # 计算自动推断的维度大小
            known_size = 1
            for i, dim in enumerate(shape):
                if i != auto_dim:
                    known_size *= dim

            if total_elements % known_size != 0:
                raise ValueError(f"Cannot reshape tensor of size {total_elements} to shape {shape}")

            shape[auto_dim] = total_elements // known_size

        # 检查新形状的元素总数是否匹配
        new_total = 1
        for dim in shape:
            new_total *= dim

        if new_total != total_elements:
            raise ValueError(f"Cannot reshape tensor of size {total_elements} to shape {shape}")

        return self.reshape(*shape)

    def cat(tensors, dim=0):
        """
        在指定维度上连接张量

        Args:
            tensors: 要连接的张量列表
            dim: 连接的维度

        Returns:
            Tensor: 连接后的张量

        Example:
            >>> x = Tensor([[1, 2]])
            >>> y = Tensor([[3, 4]])
            >>> z = cat([x, y], dim=0)  # shape: (2, 2)
        """
        if not tensors:
            raise ValueError("Cannot concatenate empty list of tensors")

        if len(tensors) == 1:
            return tensors[0]

        # 确保所有输入都是Tensor
        tensors = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]

        # 获取第一个tensor的信息
        first_tensor = tensors[0]
        xp = first_tensor._get_array_module()
        device = first_tensor.device

        # 处理负数维度
        if dim < 0:
            dim = first_tensor.ndim + dim

        if not (0 <= dim < first_tensor.ndim):
            raise IndexError(f"Dimension {dim} is out of range for {first_tensor.ndim}D tensor")

        # 检查所有张量的形状兼容性
        for i, tensor in enumerate(tensors[1:], 1):
            if tensor.ndim != first_tensor.ndim:
                raise ValueError(f"All tensors must have the same number of dimensions. "
                                 f"Tensor {i} has {tensor.ndim} dims, expected {first_tensor.ndim}")

            for d in range(first_tensor.ndim):
                if d != dim and tensor.shape[d] != first_tensor.shape[d]:
                    raise ValueError(f"All tensors must have the same size in all dimensions except {dim}. "
                                     f"Tensor {i} has size {tensor.shape[d]} in dim {d}, expected {first_tensor.shape[d]}")

        # 确保所有张量在同一设备上
        tensor_data = []
        for tensor in tensors:
            if tensor.device != device:
                tensor = tensor.to(device)
            tensor_data.append(tensor.data)

        # 执行连接
        result_data = xp.concatenate(tensor_data, axis=dim)

        # 检查是否需要梯度
        requires_grad = any(t.requires_grad for t in tensors)

        out = Tensor(
            result_data,
            requires_grad=requires_grad,
            device=device,
            _children=tuple(tensors),
            _op='cat'
        )

        if requires_grad:
            def _cat_backward():
                if out.grad is None:
                    return

                # 计算每个tensor在cat维度上的起始和结束位置
                start_idx = 0
                for tensor in tensors:
                    if tensor.requires_grad:
                        if tensor.grad is None:
                            tensor.grad = zeros(*tensor.shape, device=tensor.device)

                        end_idx = start_idx + tensor.shape[dim]

                        # 使用切片提取对应的梯度
                        slices = [slice(None)] * out.ndim
                        slices[dim] = slice(start_idx, end_idx)

                        tensor.grad.data += out.grad.data[tuple(slices)]
                        start_idx = end_idx

            out._backward = _cat_backward

        return out

    def __getitem__(self, key):
        """
        张量索引操作
        支持多种索引方式：整数、切片、元组等

        Args:
            key: 索引键，可以是整数、切片、元组等

        Returns:
            Tensor: 索引后的张量

        Examples:
            >>> x = Tensor([[1, 2, 3], [4, 5, 6]])
            >>> x[0]        # 第一行
            >>> x[:, 1]     # 第二列
            >>> x[0:2, 1:3] # 子矩阵
            >>> x[(0, 1)]   # 元组索引
        """
        xp = self._get_array_module()

        # 执行索引操作
        result_data = self.data[key]

        # 创建结果张量
        out = Tensor(
            result_data,
            requires_grad=self.requires_grad,
            device=self.device,
            _children=(self,),
            _op='getitem'
        )

        if self.requires_grad:
            def _getitem_backward():
                if self.grad is None:
                    self.grad = zeros(*self.shape, device=self.device)
                if out.grad is not None:
                    # 创建与原张量同样形状的零张量
                    grad_data = xp.zeros_like(self.data)
                    # 将梯度加到对应的索引位置
                    grad_data[key] += out.grad.data
                    self.grad.data += grad_data

            out._backward = _getitem_backward

        return out

    def __setitem__(self, key, value):
        """
        张量赋值操作

        Args:
            key: 索引键
            value: 要赋值的数据
        """
        if isinstance(value, Tensor):
            self.data[key] = value.data
        else:
            self.data[key] = value

    def chunk(self, chunks, dim=0):
        """
        将张量分割成指定数量的块

        Args:
            chunks: 要分割的块数
            dim: 分割的维度

        Returns:
            List[Tensor]: 分割后的张量列表

        Example:
            >>> x = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
            >>> chunks = x.chunk(2, dim=1)  # 分割成2块
        """
        if chunks <= 0:
            raise ValueError(f"chunks must be positive, got {chunks}")

        # 处理负数维度
        if dim < 0:
            dim = self.ndim + dim

        if not (0 <= dim < self.ndim):
            raise IndexError(f"Dimension {dim} is out of range for {self.ndim}D tensor")

        size = self.shape[dim]
        chunk_size = (size + chunks - 1) // chunks  # 向上取整

        result = []
        for i in range(chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, size)

            if start >= size:
                break

            # 创建切片
            slices = [slice(None)] * self.ndim
            slices[dim] = slice(start, end)

            # 使用索引操作
            chunk_tensor = self[tuple(slices)]
            result.append(chunk_tensor)

        return result

    def split(self, split_size_or_sections, dim=0):
        """
        将张量分割成指定大小的块

        Args:
            split_size_or_sections: 分割大小或分割点列表
            dim: 分割的维度

        Returns:
            List[Tensor]: 分割后的张量列表

        Example:
            >>> x = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
            >>> splits = x.split(2, dim=1)  # 每块大小为2
            >>> splits = x.split([1, 3], dim=1)  # 指定每块的大小
        """
        # 处理负数维度
        if dim < 0:
            dim = self.ndim + dim

        if not (0 <= dim < self.ndim):
            raise IndexError(f"Dimension {dim} is out of range for {self.ndim}D tensor")

        size = self.shape[dim]

        if isinstance(split_size_or_sections, int):
            # 均匀分割
            split_size = split_size_or_sections
            if split_size <= 0:
                raise ValueError(f"split_size must be positive, got {split_size}")

            sections = []
            start = 0
            while start < size:
                end = min(start + split_size, size)
                sections.append(end - start)
                start = end
        else:
            # 按指定大小分割
            sections = list(split_size_or_sections)
            if sum(sections) != size:
                raise ValueError(f"Sum of sections ({sum(sections)}) doesn't equal tensor size ({size}) in dim {dim}")

        result = []
        start = 0
        for section_size in sections:
            end = start + section_size

            # 创建切片
            slices = [slice(None)] * self.ndim
            slices[dim] = slice(start, end)

            # 使用索引操作
            section_tensor = self[tuple(slices)]
            result.append(section_tensor)
            start = end

        return result

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
        return f"Tensor(data={self.data},grad={self.grad},shape={self.shape}, dtype={self.dtype}, device={self.device}, requires_grad={self.requires_grad})\n{self.data}"
        # return f"\nNode(当前的权重={self.data},grad={self.grad.data}"

    def __str__(self):
        return self.__repr__()

    def tolist(self):
        """
        将张量数据转换为 Python 原生列表（list）

        适配逻辑：
        - 若数据在 GPU（CuPy 数组），先转为 CPU 上的 NumPy 数组
        - 再调用数组的 tolist() 方法，确保输出为 Python 原生列表
        - 保持原始数据的维度结构（如二维张量转二维列表）

        Returns:
            list: 与张量形状一致的 Python 原生列表
        """
        # 1. 先确保数据在 CPU 上（CuPy 数组需转 NumPy）
        if self.device == 'cuda' and CUDA_AVAILABLE and isinstance(self.data, cp.ndarray):
            # GPU 数据转 CPU 的 NumPy 数组
            cpu_data = cp.asnumpy(self.data)
        else:
            # CPU 数据直接使用（已为 NumPy 数组）
            cpu_data = self.data

        # 2. 调用 NumPy 数组的 tolist() 转为 Python 列表
        return cpu_data.tolist()

    def stack(tensors, dim=0):
        """
        在新维度上堆叠张量

        Args:
            tensors: 要堆叠的张量列表
            dim: 新维度的位置

        Returns:
            Tensor: 堆叠后的张量

        Example:
            >>> x = Tensor([1, 2])
            >>> y = Tensor([3, 4])
            >>> z = stack([x, y], dim=0)  # shape: (2, 2)
        """
        if not tensors:
            raise ValueError("Cannot stack empty list of tensors")

        if len(tensors) == 1:
            return tensors[0].unsqueeze(dim)

        # 确保所有输入都是Tensor
        tensors = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]

        # 检查所有张量的形状是否相同
        first_shape = tensors[0].shape
        for i, tensor in enumerate(tensors[1:], 1):
            if tensor.shape != first_shape:
                raise ValueError(f"All tensors must have the same shape. "
                                 f"Tensor {i} has shape {tensor.shape}, expected {first_shape}")

        # 处理负数维度
        ndim_new = tensors[0].ndim + 1
        if dim < 0:
            dim = ndim_new + dim

        if not (0 <= dim < ndim_new):
            raise IndexError(f"Dimension {dim} is out of range for {ndim_new}D tensor")

        # 在指定维度上为每个tensor添加一个维度，然后连接
        unsqueezed_tensors = [tensor.unsqueeze(dim) for tensor in tensors]
        return cat(unsqueezed_tensors, dim=dim)


    def maximum(self, other):
        """
        逐元素计算两个张量的最大值
        """
        other = self._ensure_tensor(other)
        xp = self._get_array_module()

        result_data = xp.maximum(self.data, other.data)
        result = Tensor(
            result_data,
            requires_grad=self.requires_grad or other.requires_grad,
            device=self.device,
            _children=(self, other),
            _op='maximum'
        )

        if self.requires_grad or other.requires_grad:
            def _maximum_backward():
                if result.grad is None:
                    return

                if self.requires_grad:
                    if self.grad is None:
                        self.grad = zeros(*self.shape, device=self.device)
                    # 当 self >= other 时，梯度传递给 self
                    mask_self = (self.data >= other.data).astype(self.dtype)
                    self.grad.data += result.grad.data * mask_self

                if other.requires_grad:
                    if other.grad is None:
                        other.grad = zeros(*other.shape, device=other.device)
                    # 当 other > self 时，梯度传递给 other
                    mask_other = (other.data > self.data).astype(self.dtype)
                    other.grad.data += result.grad.data * mask_other

            result._backward = _maximum_backward

        return result


    def minimum(self, other):
        """
        逐元素计算两个张量的最小值
        """
        other = self._ensure_tensor(other)
        xp = self._get_array_module()

        result_data = xp.minimum(self.data, other.data)
        result = Tensor(
            result_data,
            requires_grad=self.requires_grad or other.requires_grad,
            device=self.device,
            _children=(self, other),
            _op='minimum'
        )

        if self.requires_grad or other.requires_grad:
            def _minimum_backward():
                if result.grad is None:
                    return

                if self.requires_grad:
                    if self.grad is None:
                        self.grad = zeros(*self.shape, device=self.device)
                    # 当 self <= other 时，梯度传递给 self
                    mask_self = (self.data <= other.data).astype(self.dtype)
                    self.grad.data += result.grad.data * mask_self

                if other.requires_grad:
                    if other.grad is None:
                        other.grad = zeros(*other.shape, device=other.device)
                    # 当 other < self 时，梯度传递给 other
                    mask_other = (other.data < self.data).astype(self.dtype)
                    other.grad.data += result.grad.data * mask_other

            result._backward = _minimum_backward

        return result


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

# 在模块级别定义（不在Tensor类内）：
def cat(tensors, dim=0):
    """连接张量的静态函数"""
    if not tensors:
        raise ValueError("Cannot concatenate empty list of tensors")

    if len(tensors) == 1:
        return tensors[0]

    # 确保所有输入都是Tensor
    tensors = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
    first_tensor = tensors[0]
    xp = first_tensor._get_array_module()

    # 执行连接
    tensor_data = [t.data for t in tensors]
    result_data = xp.concatenate(tensor_data, axis=dim)

    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(result_data, requires_grad=requires_grad, device=first_tensor.device)

    if requires_grad:
        def _cat_backward():
            if out.grad is None:
                return
            start_idx = 0
            for tensor in tensors:
                if tensor.requires_grad:
                    if tensor.grad is None:
                        tensor.grad = zeros(*tensor.shape, device=tensor.device)

                    end_idx = start_idx + tensor.shape[dim]
                    slices = [slice(None)] * out.ndim
                    slices[dim] = slice(start_idx, end_idx)
                    tensor.grad.data += out.grad.data[tuple(slices)]
                    start_idx = end_idx

        out._backward = _cat_backward

    return out