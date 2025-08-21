from core.v1.nn import Module
from core.v1.engine import Node
import random as python_random

class Conv2d(Module):
    """二维卷积层

    支持多通道输入输出，可指定卷积核大小、步长、填充等参数
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        """
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小，可以是int或(height, width)
        stride: 步长，可以是int或(height, width)
        padding: 填充，可以是int或(height, width)
        bias: 是否使用偏置
        """
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 处理kernel_size参数
        if isinstance(kernel_size, int):
            self.kernel_h = self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size

        # 处理stride参数
        if isinstance(stride, int):
            self.stride_h = self.stride_w = stride
        else:
            self.stride_h, self.stride_w = stride

        # 处理padding参数
        if isinstance(padding, int):
            self.pad_h = self.pad_w = padding
        else:
            self.pad_h, self.pad_w = padding

        # 初始化权重：形状为 (out_channels, in_channels, kernel_h, kernel_w)
        self.weight = []
        for _ in range(out_channels):
            out_channel_weights = []
            for _ in range(in_channels):
                kernel_weights = []
                for _ in range(self.kernel_h):
                    row = []
                    for _ in range(self.kernel_w):
                        # 使用Xavier初始化
                        fan_in = in_channels * self.kernel_h * self.kernel_w
                        std = (2.0 / fan_in) ** 0.5
                        weight_val = python_random.gauss(0, std)
                        row.append(Node(weight_val))
                    kernel_weights.append(row)
                out_channel_weights.append(kernel_weights)
            self.weight.append(out_channel_weights)

        # 初始化偏置
        if bias:
            self.bias = [Node(0.0) for _ in range(out_channels)]
        else:
            self.bias = None

    def __call__(self, x):
        """
        前向传播
        x: 输入张量，格式为 [batch_size, in_channels, height, width]
           对于单个样本：[in_channels, height, width]
        """
        # 处理单个样本的情况
        if len(x) == self.in_channels and isinstance(x[0], list):
            x = [x]  # 添加batch维度
            single_sample = True
        else:
            single_sample = False

        batch_size = len(x)
        in_h, in_w = len(x[0][0]), len(x[0][0][0])

        # 计算输出尺寸
        out_h = (in_h + 2 * self.pad_h - self.kernel_h) // self.stride_h + 1
        out_w = (in_w + 2 * self.pad_w - self.kernel_w) // self.stride_w + 1

        # 初始化输出
        output = []

        for batch_idx in range(batch_size):
            batch_output = []

            for out_ch in range(self.out_channels):
                channel_output = []

                for i in range(out_h):
                    row_output = []

                    for j in range(out_w):
                        # 计算卷积
                        conv_sum = Node(0.0)

                        for in_ch in range(self.in_channels):
                            for ki in range(self.kernel_h):
                                for kj in range(self.kernel_w):
                                    # 计算输入位置
                                    input_i = i * self.stride_h - self.pad_h + ki
                                    input_j = j * self.stride_w - self.pad_w + kj

                                    # 检查边界（padding区域为0）
                                    if 0 <= input_i < in_h and 0 <= input_j < in_w:
                                        input_val = x[batch_idx][in_ch][input_i][input_j]
                                        weight_val = self.weight[out_ch][in_ch][ki][kj]
                                        conv_sum = conv_sum + input_val * weight_val

                        # 添加偏置
                        if self.bias is not None:
                            conv_sum = conv_sum + self.bias[out_ch]

                        row_output.append(conv_sum)
                    channel_output.append(row_output)
                batch_output.append(channel_output)
            output.append(batch_output)

        # 如果输入是单个样本，返回单个样本
        return output[0] if single_sample else output

    def parameters(self):
        """返回所有可学习参数"""
        params = []

        # 添加权重参数
        for out_ch_weights in self.weight:
            for in_ch_weights in out_ch_weights:
                for kernel_row in in_ch_weights:
                    for weight_node in kernel_row:
                        params.append(weight_node)

        # 添加偏置参数
        if self.bias is not None:
            params.extend(self.bias)

        return params

    def __repr__(self):
        return (f"Conv2d({self.in_channels}, {self.out_channels}, "
                f"kernel_size=({self.kernel_h}, {self.kernel_w}), "
                f"stride=({self.stride_h}, {self.stride_w}), "
                f"padding=({self.pad_h}, {self.pad_w}))")