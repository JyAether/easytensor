from core.nn.module.conv import Conv2d
from core.nn.tensor_nn import Module, Dropout, Linear, ReLU, Adam, CrossEntropyLoss
from core.tensor import Tensor, zeros, randn
import numpy as np
class MaxPool2d(Module):
    """2D最大池化层 - 完整实现"""

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

    def forward(self, x):
        """最大池化前向传播"""
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input, got {x.ndim}D")

        batch_size, channels, input_height, input_width = x.shape
        xp = x._get_array_module()

        # 计算输出尺寸
        output_height = (input_height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        output_width = (input_width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        # Padding
        if self.padding != (0, 0):
            pad_h, pad_w = self.padding
            pad_config = ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w))
            x_padded = xp.pad(x.data, pad_config, mode='constant', constant_values=-float('inf'))
        else:
            x_padded = x.data

        # 输出张量和索引记录
        output_data = xp.zeros((batch_size, channels, output_height, output_width), dtype=x.dtype)
        # 记录最大值的原始位置用于反向传播
        max_indices_h = xp.zeros((batch_size, channels, output_height, output_width), dtype=xp.int32)
        max_indices_w = xp.zeros((batch_size, channels, output_height, output_width), dtype=xp.int32)

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride

        # 池化操作
        for i in range(output_height):
            for j in range(output_width):
                # 计算池化窗口的位置
                h_start = i * stride_h
                h_end = h_start + kernel_h
                w_start = j * stride_w
                w_end = w_start + kernel_w

                # 提取池化区域
                pool_region = x_padded[:, :, h_start:h_end, w_start:w_end]
                pool_shape = pool_region.shape

                # 展平池化区域以找到最大值
                pool_flat = pool_region.reshape(batch_size, channels, -1)
                max_vals = xp.max(pool_flat, axis=2)
                max_idx_flat = xp.argmax(pool_flat, axis=2)

                # 保存结果
                output_data[:, :, i, j] = max_vals

                # 将一维索引转换为二维索引
                max_idx_h = max_idx_flat // kernel_w
                max_idx_w = max_idx_flat % kernel_w

                # 转换为原始坐标（考虑padding）
                orig_h = max_idx_h + h_start - self.padding[0] if self.padding[0] > 0 else max_idx_h + h_start
                orig_w = max_idx_w + w_start - self.padding[1] if self.padding[1] > 0 else max_idx_w + w_start

                # 确保坐标在有效范围内
                orig_h = xp.clip(orig_h, 0, input_height - 1)
                orig_w = xp.clip(orig_w, 0, input_width - 1)

                max_indices_h[:, :, i, j] = orig_h
                max_indices_w[:, :, i, j] = orig_w

        result = Tensor(output_data, requires_grad=x.requires_grad, device=x.device)

        # 反向传播
        if result.requires_grad:
            def _maxpool2d_backward():
                if result.grad is None or x.grad is None:
                    return

                grad_output = result.grad.data

                # 将梯度传播到最大值位置
                for i in range(output_height):
                    for j in range(output_width):
                        for b in range(batch_size):
                            for c in range(channels):
                                h_idx = max_indices_h[b, c, i, j]
                                w_idx = max_indices_w[b, c, i, j]
                                x.grad.data[b, c, h_idx, w_idx] += grad_output[b, c, i, j]

            result._backward = _maxpool2d_backward

        return result

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class AvgPool2d(Module):
    """2D平均池化层"""

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

    def forward(self, x):
        """平均池化前向传播"""
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input, got {x.ndim}D")

        batch_size, channels, input_height, input_width = x.shape
        xp = x._get_array_module()

        # 计算输出尺寸
        output_height = (input_height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        output_width = (input_width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        # Padding
        if self.padding != (0, 0):
            pad_h, pad_w = self.padding
            pad_config = ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w))
            x_padded = xp.pad(x.data, pad_config, mode='constant', constant_values=0)
        else:
            x_padded = x.data

        # 输出张量
        output_data = xp.zeros((batch_size, channels, output_height, output_width), dtype=x.dtype)

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        pool_size = kernel_h * kernel_w

        # 池化操作
        for i in range(output_height):
            for j in range(output_width):
                h_start = i * stride_h
                h_end = h_start + kernel_h
                w_start = j * stride_w
                w_end = w_start + kernel_w

                pool_region = x_padded[:, :, h_start:h_end, w_start:w_end]
                output_data[:, :, i, j] = xp.mean(pool_region, axis=(2, 3))

        result = Tensor(output_data, requires_grad=x.requires_grad, device=x.device)

        # 反向传播
        if result.requires_grad:
            def _avgpool2d_backward():
                if result.grad is None or x.grad is None:
                    return

                grad_output = result.grad.data

                # 将梯度均匀分配到池化区域的所有位置
                for i in range(output_height):
                    for j in range(output_width):
                        h_start = i * stride_h
                        h_end = h_start + kernel_h
                        w_start = j * stride_w
                        w_end = w_start + kernel_w

                        # 计算有效的池化区域（考虑padding）
                        h_start_orig = max(0, h_start - self.padding[0])
                        h_end_orig = min(input_height, h_end - self.padding[0])
                        w_start_orig = max(0, w_start - self.padding[1])
                        w_end_orig = min(input_width, w_end - self.padding[1])

                        if h_start_orig < h_end_orig and w_start_orig < w_end_orig:
                            grad_per_element = grad_output[:, :, i, j] / pool_size
                            x.grad.data[:, :, h_start_orig:h_end_orig, w_start_orig:w_end_orig] += \
                                grad_per_element[:, :, None, None]

            result._backward = _avgpool2d_backward

        return result

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"AvgPool2d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class Flatten(Module):
    """展平层 - 将多维张量展平为2D"""

    def __init__(self, start_dim=1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x):
        if x.ndim <= self.start_dim:
            return x

        # 保持前start_dim维度，展平后面的维度
        new_shape = list(x.shape[:self.start_dim]) + [-1]
        return x.reshape(*new_shape)

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"Flatten(start_dim={self.start_dim})"


# ==================== CNN案例：手写数字识别 ====================

class SimpleCNN(Module):
    """简单的CNN模型用于手写数字识别"""

    def __init__(self, num_classes=10):
        super().__init__()

        # 卷积层
        self.conv1 = Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 28x28 -> 28x28
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14

        self.conv2 = Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 14x14 -> 14x14
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7

        self.conv3 = Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 7x7 -> 7x7
        self.pool3 = AvgPool2d(kernel_size=7, stride=1)  # 7x7 -> 1x1

        # 全连接层
        self.flatten = Flatten()
        self.dropout = Dropout(0.5)
        self.fc1 = Linear(128, 64)
        self.fc2 = Linear(64, num_classes)

        # 激活函数
        self.relu = ReLU()

    def forward(self, x):
        # 卷积块1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        # 卷积块2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        # 卷积块3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)

        # 全连接部分
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

    def __call__(self, x):
        return self.forward(x)
