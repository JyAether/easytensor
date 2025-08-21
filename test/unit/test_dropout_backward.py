from core.v1.nn2.module.dropout import Dropout
from core.v1.engine import Node

def test_dropout_backward():
    """测试Dropout的backward传播"""
    print("=== 测试Dropout的backward传播 ===")

    # 创建简单的输入
    x = [Node(1.0), Node(2.0), Node(3.0), Node(4.0)]

    # 创建Dropout层
    dropout = Dropout(p=0.5)
    dropout.train()  # 训练模式

    print("原始输入:", [xi.data for xi in x])

    # 前向传播
    y = dropout(x)
    print("Dropout后输出:", [yi.data for yi in y])

    # 计算一个简单的损失：所有输出的平方和
    loss = sum(yi ** 2 for yi in y)
    print("损失值:", loss.data)

    # 反向传播
    loss.backward()

    # 检查梯度
    print("输入梯度:", [xi.grad for xi in x])
    print("输出梯度:", [yi.grad for yi in y])

    # 验证梯度传播的正确性
    print("\n梯度传播验证:")
    for i, (xi, yi) in enumerate(zip(x, y)):
        expected_grad = 2 * yi.data  # d(yi^2)/dyi = 2*yi
        print(f"节点{i}: 输出梯度={yi.grad:.4f}, 期望梯度={expected_grad:.4f}")

        # 如果该节点被dropout掉了(输出为0)，输入梯度应该为0
        # 如果该节点被保留了，输入梯度应该等于输出梯度乘以scale_factor
        if yi.data == 0:
            assert xi.grad == 0, f"被dropout的节点{i}的输入梯度应该为0"
        else:
            scale_factor = yi.data / xi.data
            expected_input_grad = yi.grad * scale_factor
            print(f"节点{i}: 输入梯度={xi.grad:.4f}, 期望输入梯度={expected_input_grad:.4f}")

    print("\n测试evaluatio模式:")
    dropout.eval()  # 评估模式
    y_eval = dropout(x)
    print("评估模式输出:", [yi.data for yi in y_eval])
    print("评估模式下输出应该等于输入:", [xi.data for xi in x])

    print("\n测试train模式:")
    dropout.train()  # 训练模式
    y_train = dropout(x)
    print("训练模式下原始输入:", [xi.data for xi in x])
    print("训练模式dropout概率:", dropout.p)
    print("训练模式输出:", [yi.data for yi in y_train])


if __name__ == "__main__":
    test_dropout_backward()