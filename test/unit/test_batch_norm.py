from core.v1.nn2.module.batchnorm import BatchNorm
from core.v1.engine import Node
import random as python_random

def test_batch_norm():
    """测试BatchNorm的功能"""
    print("=== BatchNorm测试 ===")

    # 1. 基本功能测试
    print("1. 基本功能测试:")
    bn = BatchNorm(num_features=3)

    # 创建测试数据：3个样本，每个样本3个特征
    x = [
        [Node(1.0), Node(2.0), Node(3.0)],
        [Node(4.0), Node(5.0), Node(6.0)],
        [Node(7.0), Node(8.0), Node(9.0)]
    ]

    print("输入数据:")
    for i, sample in enumerate(x):
        print(f"样本{i}: [{', '.join(f'{f.data:.1f}' for f in sample)}]")

    # 训练模式前向传播
    bn.train()
    output = bn(x)

    print("\n训练模式输出:")
    for i, sample in enumerate(output):
        print(f"样本{i}: [{', '.join(f'{f.data:.4f}' for f in sample)}]")

    print(f"\n滑动平均统计:")
    print(f"均值: [{', '.join(f'{m:.4f}' for m in bn.running_mean)}]")
    print(f"方差: [{', '.join(f'{v:.4f}' for v in bn.running_var)}]")

    # 2. 推理模式测试
    print("\n2. 推理模式测试:")
    bn.eval()

    # 测试单个样本
    test_sample = [Node(2.5), Node(3.5), Node(4.5)]
    output_single = bn(test_sample)

    print(f"测试样本: [{', '.join(f'{f.data:.1f}' for f in test_sample)}]")
    print(f"推理输出: [{', '.join(f'{f.data:.4f}' for f in output_single)}]")

    # 3. 反向传播测试
    print("\n3. 反向传播测试:")
    bn.train()

    # 清零梯度
    bn.zero_grad()
    for sample in x:
        for feature in sample:
            feature.grad = 0

    # 前向传播
    output = bn(x)

    # 计算简单的损失：所有输出的平方和
    loss = Node(0.0)
    for sample in output:
        for feature in sample:
            loss = loss + feature * feature

    print(f"损失值: {loss.data:.4f}")

    # 反向传播
    loss.backward()

    print("参数梯度:")
    print(f"gamma梯度: [{', '.join(f'{g.grad:.4f}' for g in bn.gamma)}]")
    print(f"beta梯度:  [{', '.join(f'{b.grad:.4f}' for b in bn.beta)}]")

    print("输入梯度:")
    for i, sample in enumerate(x):
        print(f"样本{i}: [{', '.join(f'{f.grad:.4f}' for f in sample)}]")


def test_batch_norm_training():
    """测试BatchNorm在训练过程中的行为"""
    print("\n=== BatchNorm训练过程测试 ===")

    bn = BatchNorm(num_features=2, momentum=0.1)
    bn.train()

    print("Batch | 批次均值 | 批次方差 | 滑动均值 | 滑动方差")
    print("-" * 60)

    # 模拟多个批次的训练
    for batch_idx in range(5):
        # 生成随机批次数据
        batch_data = []
        for _ in range(4):  # 4个样本
            sample = [Node(python_random.gauss(batch_idx, 1)),
                      Node(python_random.gauss(batch_idx * 2, 1.5))]
            batch_data.append(sample)

        # 前向传播（会更新滑动统计量）
        output = bn(batch_data)

        # 计算这个批次的统计量用于显示
        batch_mean = [sum(s[j].data for s in batch_data) / len(batch_data)
                      for j in range(2)]
        batch_var = [sum((s[j].data - batch_mean[j]) ** 2 for s in batch_data) / len(batch_data)
                     for j in range(2)]

        print(f"  {batch_idx}   | [{batch_mean[0]:5.2f},{batch_mean[1]:5.2f}] | "
              f"[{batch_var[0]:5.2f},{batch_var[1]:5.2f}] | "
              f"[{bn.running_mean[0]:5.2f},{bn.running_mean[1]:5.2f}] | "
              f"[{bn.running_var[0]:5.2f},{bn.running_var[1]:5.2f}]")

    print(f"\n总批次数: {bn.num_batches_tracked}")


def usage_example():
    """BatchNorm使用示例"""
    print("\n=== BatchNorm使用示例 ===")

    # 在神经网络中使用BatchNorm
    class SimpleLayer:
        def __init__(self, input_size, output_size, use_bn=False):
            self.weights = [[Node(python_random.gauss(0, 0.1))
                             for _ in range(input_size)] for _ in range(output_size)]
            self.bias = [Node(0.0) for _ in range(output_size)]

            if use_bn:
                self.bn = BatchNorm(output_size)
            else:
                self.bn = None

        def __call__(self, x):
            # 线性变换
            output = []
            for sample in x:
                sample_out = []
                for i in range(len(self.weights)):
                    out = sum(w * f for w, f in zip(self.weights[i], sample))
                    out = out + self.bias[i]
                    sample_out.append(out)
                output.append(sample_out)

            # 批归一化
            if self.bn:
                output = self.bn(output)

            return output

        def parameters(self):
            params = []
            for w_row in self.weights:
                params.extend(w_row)
            params.extend(self.bias)
            if self.bn:
                params.extend(self.bn.parameters())
            return params

    # 创建带BatchNorm的层
    layer = SimpleLayer(3, 2, use_bn=True)

    # 训练数据
    train_data = [
        [Node(1.0), Node(2.0), Node(3.0)],
        [Node(4.0), Node(5.0), Node(6.0)],
        [Node(7.0), Node(8.0), Node(9.0)]
    ]

    print("使用BatchNorm的层:")
    print(f"参数数量: {len(layer.parameters())}")

    # 前向传播
    output = layer(train_data)
    print("输出形状:", len(output), "x", len(output[0]))

    print("\n使用建议:")
    print("1. 通常在线性层之后、激活函数之前使用BatchNorm")
    print("2. 训练时使用 bn.train()，推理时使用 bn.eval()")
    print("3. BatchNorm的参数需要包含在优化器中")
    print("4. 小批量大小太小时BatchNorm效果可能不好")


if __name__ == "__main__":
    test_batch_norm()
    # test_batch_norm_training()
    # usage_example()