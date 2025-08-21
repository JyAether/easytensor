

from core.v1.nn import Module
from core.v1.nn2.module.conv import Conv2d
from core.v1.nn2.module.pooling import MaxPool2d, AvgPool2d
from core.v1.engine import Node
from core.v1.nn2.module.dropout import Dropout
from core.v1.nn import Layer
from core.v1.optim import Adam
import random as python_random
from core.v1.nn2.module.lr_scheduler import StepLR



class CNN(Module):
    """简单的卷积神经网络"""

    def __init__(self, input_channels=1, num_classes=10):
        """
        input_channels: 输入图像通道数
        num_classes: 分类类别数
        """
        self.input_channels = input_channels
        self.num_classes = num_classes

        # 卷积层
        self.conv1 = Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = Conv2d(64, 128, kernel_size=3, padding=1)

        # 池化层
        self.pool = MaxPool2d(kernel_size=2, stride=2)

        # 全连接层（假设输入为28x28图像）
        # 经过3次池化后：28 -> 14 -> 7 -> 3（向下取整）
        self.fc1 = Layer(128 * 3 * 3, 512, activation='relu')
        self.fc2 = Layer(512, 256, activation='relu')
        self.fc3 = Layer(256, num_classes, activation='linear')

        # Dropout层用于正则化
        self.dropout = Dropout(0.5)

    def __call__(self, x):
        """
        前向传播
        x: 输入图像 [batch_size, channels, height, width] 或 [channels, height, width]
        """
        # 第一个卷积块
        x = self.conv1(x)
        x = self._apply_relu(x)
        x = self.pool(x)

        # 第二个卷积块
        x = self.conv2(x)
        x = self._apply_relu(x)
        x = self.pool(x)

        # 第三个卷积块
        x = self.conv3(x)
        x = self._apply_relu(x)
        x = self.pool(x)

        # 展平为一维向量
        x = self._flatten(x)

        # 全连接层
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x

    def _apply_relu(self, x):
        """对张量的每个元素应用ReLU激活函数"""

        def apply_relu_recursive(data):
            if isinstance(data, Node):
                return data.relu()
            elif isinstance(data, list):
                return [apply_relu_recursive(item) for item in data]
            else:
                raise TypeError(f"Unexpected data type: {type(data)}")

        return apply_relu_recursive(x)

    def _flatten(self, x):
        """将多维张量展平为一维向量"""

        def flatten_recursive(data, result=None):
            if result is None:
                result = []

            if isinstance(data, Node):
                result.append(data)
            elif isinstance(data, list):
                for item in data:
                    flatten_recursive(item, result)
            else:
                raise TypeError(f"Unexpected data type: {type(data)}")

            return result

        return flatten_recursive(x)

    def parameters(self):
        """返回所有可学习参数"""
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.conv2.parameters())
        params.extend(self.conv3.parameters())
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        params.extend(self.fc3.parameters())
        return params

    def train(self):
        """设置为训练模式"""
        self.dropout.train()

    def eval(self):
        """设置为评估模式"""
        self.dropout.eval()


class LeNet(Module):
    """经典的LeNet-5架构"""

    def __init__(self, input_channels=1, num_classes=10):
        self.conv1 = Conv2d(input_channels, 6, kernel_size=5)
        self.conv2 = Conv2d(6, 16, kernel_size=5)
        self.pool = AvgPool2d(kernel_size=2, stride=2)

        # LeNet的全连接层
        self.fc1 = Layer(16 * 5 * 5, 120, activation='tanh')
        self.fc2 = Layer(120, 84, activation='tanh')
        self.fc3 = Layer(84, num_classes, activation='linear')

    def __call__(self, x):
        # 第一个卷积层 + 池化
        x = self.conv1(x)
        x = self._apply_tanh(x)
        x = self.pool(x)

        # 第二个卷积层 + 池化
        x = self.conv2(x)
        x = self._apply_tanh(x)
        x = self.pool(x)

        # 展平
        x = self._flatten(x)

        # 全连接层
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

    def _apply_tanh(self, x):
        """应用tanh激活函数"""

        def apply_tanh_recursive(data):
            if isinstance(data, Node):
                return data.tanh()
            elif isinstance(data, list):
                return [apply_tanh_recursive(item) for item in data]
            else:
                raise TypeError(f"Unexpected data type: {type(data)}")

        return apply_tanh_recursive(x)

    def _flatten(self, x):
        """展平操作"""

        def flatten_recursive(data, result=None):
            if result is None:
                result = []

            if isinstance(data, Node):
                result.append(data)
            elif isinstance(data, list):
                for item in data:
                    flatten_recursive(item, result)
            else:
                raise TypeError(f"Unexpected data type: {type(data)}")

            return result

        return flatten_recursive(x)

    def parameters(self):
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.conv2.parameters())
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        params.extend(self.fc3.parameters())
        return params


def debug_data_structure(data, name="data", level=0):
    """调试辅助函数：打印数据结构"""
    indent = "  " * level
    if isinstance(data, Node):
        print(f"{indent}{name}: Node({data.data:.4f})")
    elif isinstance(data, list):
        print(f"{indent}{name}: List[{len(data)}]")
        if len(data) > 0:
            if isinstance(data[0], Node):
                print(f"{indent}  -> Contains Node objects")
            elif isinstance(data[0], list):
                print(f"{indent}  -> Contains nested lists")
                if len(data) <= 3:  # 只展示前几个元素
                    for i, item in enumerate(data):
                        debug_data_structure(item, f"[{i}]", level + 1)
            else:
                print(f"{indent}  -> Contains: {type(data[0])}")
    else:
        print(f"{indent}{name}: {type(data)}")


def generate_dummy_image_data(batch_size=10, channels=1, height=28, width=28):
    """生成虚拟图像数据用于测试"""
    images = []
    labels = []

    for _ in range(batch_size):
        # 生成随机图像
        image = []
        for c in range(channels):
            channel = []
            for h in range(height):
                row = []
                for w in range(width):
                    # 生成0到1之间的随机值
                    pixel_value = python_random.random()
                    row.append(Node(pixel_value))
                channel.append(row)
            image.append(channel)

        images.append(image)

        # 生成随机标签（one-hot编码）
        label_idx = python_random.randint(0, 9)
        label = [Node(1.0 if i == label_idx else 0.0) for i in range(10)]
        labels.append(label)

    return images, labels


def softmax_cross_entropy_loss(predictions, targets):
    """计算softmax交叉熵损失"""
    # 先计算softmax
    exp_preds = [pred.exp() for pred in predictions]  # 需要在Node中添加exp方法
    sum_exp = sum(exp_preds)
    softmax_preds = [exp_pred / sum_exp for exp_pred in exp_preds]

    # 计算交叉熵损失
    loss = Node(0.0)
    for pred, target in zip(softmax_preds, targets):
        # 添加小常数防止log(0)
        epsilon = Node(1e-7)
        log_pred = (pred + epsilon).log()  # 需要在Node中添加log方法
        loss = loss - target * log_pred

    return loss


def train_cnn_example():
    """训练CNN的完整示例"""
    print("=== CNN训练示例 ===")

    # 创建模型
    model = CNN(input_channels=1, num_classes=10)
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.8)

    print(f"模型参数数量: {len(model.parameters())}")

    # 生成训练数据
    train_images, train_labels = generate_dummy_image_data(batch_size=5)  # 减少批次大小用于调试
    val_images, val_labels = generate_dummy_image_data(batch_size=2)

    print("训练数据形状:", len(train_images), "x", len(train_images[0]), "x",
          len(train_images[0][0]), "x", len(train_images[0][0][0]))

    # 调试：检查第一个图像的数据结构
    print("\n=== 调试信息 ===")
    debug_data_structure(train_images[0], "train_images[0]")

    # 训练循环
    print("\n开始训练...")
    print("Epoch | Train Loss | Learning Rate")
    print("-" * 35)

    for epoch in range(3):  # 减少epoch数用于调试
        model.train()

        total_loss = 0
        for i, (image, label) in enumerate(zip(train_images[:2], train_labels[:2])):  # 使用更小批量
            try:
                print(f"\n处理样本 {i + 1}...")

                # 调试：检查卷积前的数据结构
                print("卷积前数据结构:")
                debug_data_structure(image, "input_image")

                # 前向传播
                print("开始前向传播...")
                x = model.conv1(image)
                print("Conv1完成")
                debug_data_structure(x, "after_conv1")

                x = model._apply_relu(x)
                print("ReLU完成")
                debug_data_structure(x, "after_relu")

                # 如果到这里没问题，继续完整的前向传播
                pred = model(image)

                # 计算损失（简化的MSE损失）
                loss = Node(0.0)
                for p, t in zip(pred, label):
                    loss = loss + (p - t) ** 2

                total_loss += loss.data

                # 反向传播
                model.zero_grad()  # 需要确保这个方法存在
                loss.backward()
                optimizer.step()

                print(f"样本 {i + 1} 完成，损失: {loss.data:.6f}")

            except Exception as e:
                print(f"处理样本 {i + 1} 时出错: {e}")
                import traceback
                traceback.print_exc()
                return  # 出错时退出

        avg_loss = total_loss / 2
        current_lr = scheduler.step()

        print(f"{epoch:5d} | {avg_loss:10.6f} | {current_lr:.6f}")

    print("\n训练完成！")


def test_simple_conv():
    """简单的卷积测试"""
    print("=== 简单卷积测试 ===")

    # 创建最小的测试用例
    # 单通道 3x3 图像
    image = []
    channel = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(Node(float(i * 3 + j + 1)))  # 值从1到9
        channel.append(row)
    image.append(channel)

    print("输入图像 (1x3x3):")
    for row in channel:
        print([f"{node.data:.1f}" for node in row])

    # 创建卷积层
    conv = Conv2d(1, 1, kernel_size=3, padding=1)

    try:
        output = conv(image)
        print("\n卷积输出:")
        debug_data_structure(output, "conv_output")

        # 测试ReLU
        print("\n测试ReLU...")
        model = CNN(input_channels=1, num_classes=10)
        relu_output = model._apply_relu(output)
        print("ReLU输出:")
        debug_data_structure(relu_output, "relu_output")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 先运行简单测试
    test_simple_conv()

    # 如果简单测试通过，再运行完整训练
    print("\n" + "=" * 50 + "\n")
    train_cnn_example()