import numpy as np

from core.nn.tensor_nn import Module
from core.tensor import randn, zeros, Tensor
from core.nn.tensor_nn import init_weights


class Conv2d(Module):
    """2D卷积层 - 完整实现，包含完整的梯度反向传播"""

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

        # He初始化权重 (out_channels, in_channels, kernel_h, kernel_w)
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        std = np.sqrt(2.0 / fan_in)
        self.weight = randn(
            out_channels, in_channels, self.kernel_size[0], self.kernel_size[1],
            requires_grad=True, device=device
        ) * std
        # init_weights(self.weight, init_type='normal', in_features=in_channels, out_features=out_channels)
        if bias:
            self.bias = zeros(out_channels, requires_grad=True, device=device)
        else:
            self.bias = None

    def _pad_input(self, x):
        """对输入进行padding"""
        if self.padding == (0, 0):
            return x

        xp = x._get_array_module()
        pad_h, pad_w = self.padding

        # 创建padding配置 [(batch_pad), (channel_pad), (height_pad), (width_pad)]
        pad_config = ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w))
        padded_data = xp.pad(x.data, pad_config, mode='constant', constant_values=0)

        return Tensor(padded_data, requires_grad=x.requires_grad, device=x.device)

    def _im2col(self, x_padded, output_height, output_width):
        """Im2col实现 - 将卷积转换为矩阵乘法"""
        xp = x_padded._get_array_module()
        batch_size, in_channels, padded_height, padded_width = x_padded.shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride

        # 创建输出矩阵 (kernel_size, batch*output_h*output_w)
        col_matrix = xp.zeros((
            in_channels * kernel_h * kernel_w,
            batch_size * output_height * output_width
        ), dtype=x_padded.dtype)

        # 填充矩阵
        for y_out in range(output_height):
            for x_out in range(output_width):
                # 计算在原图中的位置
                y_start = y_out * stride_h
                x_start = x_out * stride_w

                # 提取kernel大小的patch
                patch = x_padded.data[:, :, y_start:y_start + kernel_h, x_start:x_start + kernel_w]

                # 展平patch并放入col_matrix
                patch_flat = patch.reshape(batch_size, -1)  # (batch, in_ch*kh*kw)

                # 计算在col_matrix中的列索引
                col_start = y_out * output_width + x_out

                for b in range(batch_size):
                    col_idx = b * output_height * output_width + col_start
                    col_matrix[:, col_idx] = patch_flat[b]

        return col_matrix

    def _col2im(self, grad_col, x_shape, output_height, output_width):
        """Col2im实现 - im2col的逆操作，用于反向传播"""
        xp = grad_col._get_array_module() if hasattr(grad_col, '_get_array_module') else np
        if not hasattr(grad_col, 'shape'):
            grad_col = xp.array(grad_col)

        batch_size, in_channels, padded_height, padded_width = x_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride

        # 初始化输入梯度
        grad_input = xp.zeros(x_shape, dtype=grad_col.dtype)

        # 从col_matrix恢复到原始形状
        for y_out in range(output_height):
            for x_out in range(output_width):
                # 计算在原图中的位置
                y_start = y_out * stride_h
                x_start = x_out * stride_w

                # 计算在col_matrix中的列索引
                col_start = y_out * output_width + x_out

                for b in range(batch_size):
                    col_idx = b * output_height * output_width + col_start
                    patch_grad = grad_col[:, col_idx].reshape(in_channels, kernel_h, kernel_w)

                    # 累加梯度到对应位置
                    grad_input[b, :, y_start:y_start + kernel_h, x_start:x_start + kernel_w] += patch_grad

        return grad_input

    def forward(self, x):
        """完整的卷积前向传播"""
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (N, C, H, W), got {x.ndim}D")

        batch_size, in_channels, input_height, input_width = x.shape

        if in_channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {in_channels}")

        xp = x._get_array_module()

        # 计算输出尺寸
        output_height = (input_height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        output_width = (input_width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        if output_height <= 0 or output_width <= 0:
            raise ValueError(f"Invalid output size: {output_height}x{output_width}")

        # Padding
        x_padded = self._pad_input(x)

        # Im2col转换
        col_matrix = self._im2col(x_padded, output_height, output_width)

        # 权重reshape为(out_channels, kernel_size_total)
        weight_matrix = self.weight.data.reshape(self.out_channels, -1)

        # 矩阵乘法: (out_ch, kernel_size) @ (kernel_size, batch*out_h*out_w)
        output_flat = weight_matrix @ col_matrix

        # 添加bias
        if self.bias is not None:
            bias_expanded = self.bias.data.reshape(-1, 1)  # (out_ch, 1)
            output_flat = output_flat + bias_expanded

        # Reshape输出到正确形状
        output_data = output_flat.reshape(self.out_channels, batch_size, output_height, output_width)
        output_data = output_data.transpose(1, 0, 2, 3)  # (batch, out_ch, out_h, out_w)

        result = Tensor(output_data, requires_grad=x.requires_grad or self.weight.requires_grad, device=self.device)

        # 完整的反向传播实现
        if result.requires_grad:
            def _conv2d_backward():
                if result.grad is None:
                    return

                grad_output = result.grad.data  # (batch, out_ch, out_h, out_w)

                # 1. 计算权重梯度
                if self.weight.requires_grad:
                    if self.weight.grad is None:
                        self.weight.grad = zeros(*self.weight.shape, device=self.device)

                    # grad_output reshape: (out_ch, batch*out_h*out_w)
                    grad_output_flat = grad_output.transpose(1, 0, 2, 3).reshape(self.out_channels, -1)

                    # 权重梯度: grad_output @ col_matrix^T
                    weight_grad = grad_output_flat @ col_matrix.T
                    weight_grad = weight_grad.reshape(self.weight.shape)

                    self.weight.grad.data += weight_grad

                # 2. 计算bias梯度
                if self.bias is not None and self.bias.requires_grad:
                    if self.bias.grad is None:
                        self.bias.grad = zeros(*self.bias.shape, device=self.device)

                    # bias梯度是grad_output在batch, height, width维度上的求和
                    bias_grad = xp.sum(grad_output, axis=(0, 2, 3))
                    self.bias.grad.data += bias_grad

                # 3. 计算输入梯度 - 完整实现
                if x.requires_grad:
                    if x.grad is None:
                        x.grad = zeros(*x.shape, device=x.device)

                    # 输入梯度计算：weight^T @ grad_output
                    weight_matrix = self.weight.data.reshape(self.out_channels, -1)  # (out_ch, kernel_size)
                    grad_output_flat = grad_output.transpose(1, 0, 2, 3).reshape(self.out_channels,
                                                                                 -1)  # (out_ch, batch*out_h*out_w)

                    # 计算col_matrix的梯度
                    grad_col = weight_matrix.T @ grad_output_flat  # (kernel_size, batch*out_h*out_w)

                    # 使用col2im将梯度转换回输入空间
                    padded_shape = x_padded.shape
                    grad_input_padded = self._col2im(grad_col, padded_shape, output_height, output_width)

                    # 如果有padding，需要去掉padding部分的梯度
                    if self.padding != (0, 0):
                        pad_h, pad_w = self.padding
                        if pad_h > 0 and pad_w > 0:
                            grad_input = grad_input_padded[:, :, pad_h:-pad_h, pad_w:-pad_w]
                        elif pad_h > 0:
                            grad_input = grad_input_padded[:, :, pad_h:-pad_h, :]
                        elif pad_w > 0:
                            grad_input = grad_input_padded[:, :, :, pad_w:-pad_w]
                        else:
                            grad_input = grad_input_padded
                    else:
                        grad_input = grad_input_padded

                    x.grad.data += grad_input

            result._backward = _conv2d_backward

        return result

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return (f"Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})")
