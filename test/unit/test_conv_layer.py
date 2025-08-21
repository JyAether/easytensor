from core.v1.engine import Node
from core.v1.nn2.module.conv import Conv2d
from core.v1.nn2.module.pooling import MaxPool2d, AvgPool2d


def test_conv_layer():
    """测试卷积层的功能"""
    print("=== 卷积层测试 ===")

    # 创建简单的输入数据：1个通道，4x4图像
    input_data = []
    for i in range(4):
        row = []
        for j in range(4):
            row.append(Node(float(i * 4 + j + 1)))  # 1到16的数字
        input_data.append(row)

    x = [input_data]  # 1个通道

    print("输入数据形状: [1, 4, 4]")
    print("输入数据:")
    for row in input_data:
        print([node.data for node in row])

    # 创建卷积层：1输入通道，2输出通道，3x3卷积核
    conv = Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)

    # 前向传播
    output = conv(x)
    print(f"\n卷积输出形状: [2, 4, 4]")
    print("输出通道0:")
    for row in output[0]:
        print([f"{node.data:.2f}" for node in row])
    print("输出通道1:")
    for row in output[1]:
        print([f"{node.data:.2f}" for node in row])

    # 测试反向传播
    print("\n=== 反向传播测试 ===")

    # 计算简单的损失：所有输出的平方和
    loss = Node(0.0)
    for channel in output:
        for row in channel:
            for node in row:
                loss = loss + node ** 2

    print(f"损失值: {loss.data:.4f}")

    # 反向传播
    conv.zero_grad()
    loss.backward()

    # 检查梯度
    print("输入梯度:")
    for row in input_data:
        print([f"{node.grad:.4f}" for node in row])

    print(f"\n权重参数数量: {len(conv.parameters())}")
    print(f"权重梯度示例: {conv.parameters()[0].grad:.4f}")


def test_pooling_layers():
    """测试池化层"""
    print("\n=== 池化层测试 ===")

    # 创建输入数据：1个通道，4x4图像
    input_data = []
    for i in range(4):
        row = []
        for j in range(4):
            row.append(Node(float(i * 4 + j + 1)))
        input_data.append(row)

    x = [input_data]  # 1个通道

    print("输入数据:")
    for row in input_data:
        print([f"{node.data:.1f}" for node in row])

    # 测试MaxPool2d
    max_pool = MaxPool2d(kernel_size=2, stride=2)
    max_output = max_pool(x)

    print("\nMaxPool2d输出 (2x2):")
    for row in max_output[0]:
        print([f"{node.data:.1f}" for node in row])

    # 测试AvgPool2d
    avg_pool = AvgPool2d(kernel_size=2, stride=2)
    avg_output = avg_pool(x)

    print("\nAvgPool2d输出 (2x2):")
    for row in avg_output[0]:
        print([f"{node.data:.1f}" for node in row])


if __name__ == "__main__":
    test_conv_layer()
    test_pooling_layers()
