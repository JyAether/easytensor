from core.v1.nn import Module
from core.v1.engine import Node


class MaxPool2d(Module):
    """二维最大池化层"""

    def __init__(self, kernel_size, stride=None, padding=0):
        """
        kernel_size: 池化核大小
        stride: 步长，默认等于kernel_size
        padding: 填充
        """
        if isinstance(kernel_size, int):
            self.kernel_h = self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size

        if stride is None:
            self.stride_h = self.kernel_h
            self.stride_w = self.kernel_w
        elif isinstance(stride, int):
            self.stride_h = self.stride_w = stride
        else:
            self.stride_h, self.stride_w = stride

        if isinstance(padding, int):
            self.pad_h = self.pad_w = padding
        else:
            self.pad_h, self.pad_w = padding

    def __call__(self, x):
        """
        前向传播
        x: 输入张量 [batch_size, channels, height, width] 或 [channels, height, width]
        """
        # 处理单个样本的情况
        if not isinstance(x[0][0][0], list):
            x = [x]  # 添加batch维度
            single_sample = True
        else:
            single_sample = False

        batch_size = len(x)
        channels = len(x[0])
        in_h, in_w = len(x[0][0]), len(x[0][0][0])

        # 计算输出尺寸
        out_h = (in_h + 2 * self.pad_h - self.kernel_h) // self.stride_h + 1
        out_w = (in_w + 2 * self.pad_w - self.kernel_w) // self.stride_w + 1

        output = []

        for batch_idx in range(batch_size):
            batch_output = []

            for ch in range(channels):
                channel_output = []

                for i in range(out_h):
                    row_output = []

                    for j in range(out_w):
                        # 找到池化窗口内的最大值
                        max_val = Node(float('-inf'))

                        for ki in range(self.kernel_h):
                            for kj in range(self.kernel_w):
                                input_i = i * self.stride_h - self.pad_h + ki
                                input_j = j * self.stride_w - self.pad_w + kj

                                if 0 <= input_i < in_h and 0 <= input_j < in_w:
                                    current_val = x[batch_idx][ch][input_i][input_j]
                                    # 实现max操作（需要在Node类中添加max方法）
                                    max_val = self._max_node(max_val, current_val)

                        row_output.append(max_val)
                    channel_output.append(row_output)
                batch_output.append(channel_output)
            output.append(batch_output)

        return output[0] if single_sample else output

    def _max_node(self, a, b):
        """实现两个Node的max操作"""
        if a.data == float('-inf'):
            return b
        if b.data == float('-inf'):
            return a

        # 创建一个新的Node来表示max操作
        if a.data >= b.data:
            out = Node(a.data, (a, b), 'max')

            def _backward():
                a.grad += out.grad  # 梯度只传给较大的值
                # b.grad += 0  # 较小的值梯度为0

            out._backward = _backward
            return out
        else:
            out = Node(b.data, (a, b), 'max')

            def _backward():
                # a.grad += 0  # 较小的值梯度为0
                b.grad += out.grad  # 梯度只传给较大的值

            out._backward = _backward
            return out

    def parameters(self):
        """池化层没有可学习参数"""
        return []


class AvgPool2d(Module):
    """二维平均池化层"""

    def __init__(self, kernel_size, stride=None, padding=0):
        """参数与MaxPool2d相同"""
        if isinstance(kernel_size, int):
            self.kernel_h = self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size

        if stride is None:
            self.stride_h = self.kernel_h
            self.stride_w = self.kernel_w
        elif isinstance(stride, int):
            self.stride_h = self.stride_w = stride
        else:
            self.stride_h, self.stride_w = stride

        if isinstance(padding, int):
            self.pad_h = self.pad_w = padding
        else:
            self.pad_h, self.pad_w = padding

    def __call__(self, x):
        """前向传播"""
        # 处理单个样本的情况
        if not isinstance(x[0][0][0], list):
            x = [x]
            single_sample = True
        else:
            single_sample = False

        batch_size = len(x)
        channels = len(x[0])
        in_h, in_w = len(x[0][0]), len(x[0][0][0])

        # 计算输出尺寸
        out_h = (in_h + 2 * self.pad_h - self.kernel_h) // self.stride_h + 1
        out_w = (in_w + 2 * self.pad_w - self.kernel_w) // self.stride_w + 1

        output = []

        for batch_idx in range(batch_size):
            batch_output = []

            for ch in range(channels):
                channel_output = []

                for i in range(out_h):
                    row_output = []

                    for j in range(out_w):
                        # 计算池化窗口内的平均值
                        pool_sum = Node(0.0)
                        count = 0

                        for ki in range(self.kernel_h):
                            for kj in range(self.kernel_w):
                                input_i = i * self.stride_h - self.pad_h + ki
                                input_j = j * self.stride_w - self.pad_w + kj

                                if 0 <= input_i < in_h and 0 <= input_j < in_w:
                                    pool_sum = pool_sum + x[batch_idx][ch][input_i][input_j]
                                    count += 1

                        # 计算平均值
                        if count > 0:
                            avg_val = pool_sum / count
                        else:
                            avg_val = Node(0.0)

                        row_output.append(avg_val)
                    channel_output.append(row_output)
                batch_output.append(channel_output)
            output.append(batch_output)

        return output[0] if single_sample else output

    def parameters(self):
        """池化层没有可学习参数"""
        return []
