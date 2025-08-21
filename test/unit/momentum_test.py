import numpy as np
from test.network.custom_network import CustomNetwork
from core.v1.optim.sgd import SGD
from core.v1.optim import Adam
from core.v1.engine import Node
# 创建测试网络
net_sgd = CustomNetwork(hidden_activation='sigmoid', output_activation='sigmoid')
net_adam = CustomNetwork(hidden_activation='sigmoid', output_activation='sigmoid')

# 训练数据
X_train = np.array([[0.05, 0.10]])
y_train = np.array([[0.01, 0.99]])

print("比较不同优化器的训练效果:")
print(f"输入: {X_train[0]}")
print(f"目标: {y_train[0]}")
print("\nMomentum公式说明:")
print("Dt = β * St-1 + (1-β) * Wt")
print("- St-1: 历史梯度移动加权平均值")
print("- Wt: 当前时刻的梯度值")
print("- Dt: 当前时刻的指数加权平均梯度值")
print("- β: 权重系数(momentum参数)")

# 训练参数
epochs = 2000

# 测试不同优化器
optimizers = [
    ("SGD (lr=0.5)", SGD(net_sgd.parameters(), lr=0.5, momentum=0.0)),
    ("SGD+Momentum (lr=0.5, β=0.9)", SGD(CustomNetwork().parameters(), lr=0.5, momentum=0.9)),
    ("Adam (lr=0.1)", Adam(net_adam.parameters(), lr=0.1, betas=(0.9, 0.999)))
]

for name, optimizer in optimizers:
    print(f"\n--- {name} ---")

    # 重置网络（使用相同的初始权重）
    if "SGD+Momentum" in name:
        net = CustomNetwork()
        optimizer = SGD(net.parameters(), lr=0.5, momentum=0.9)
    elif "SGD" in name and "Momentum" not in name:
        net = CustomNetwork()
        optimizer = SGD(net.parameters(), lr=0.5, momentum=0.0)
    else:
        net = CustomNetwork()
        optimizer = Adam(net.parameters(), lr=0.1)

    losses = []

    for epoch in range(epochs):
        # 前向传播
        inputs = [Node(X_train[0][0]), Node(X_train[0][1])]
        targets = [Node(y_train[0][0]), Node(y_train[0][1])]

        outputs = net(inputs)

        # 计算损失
        loss1 = (outputs[0] - targets[0]) ** 2
        loss2 = (outputs[1] - targets[1]) ** 2
        total_loss = (loss1 + loss2) * 0.5

        losses.append(total_loss.data)

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()

        # 参数更新
        optimizer.step()

        # 打印进度
        if epoch % 500 == 0 or epoch == epochs - 1:
            print(
                f"  Epoch {epoch:4d}: Loss = {total_loss.data:.6f}, Output = [{outputs[0].data:.4f}, {outputs[1].data:.4f}]")

    # 最终结果
    final_inputs = [Node(0.05), Node(0.10)]
    final_outputs = net(final_inputs)
    print(f"  最终结果: [{final_outputs[0].data:.4f}, {final_outputs[1].data:.4f}]")
    print(f"  损失变化: {losses[0]:.6f} -> {losses[-1]:.6f}")

    # 如果是Momentum，展示一下缓存的梯度平均值
    if "Momentum" in name:
        print(f"  最终动量缓存值示例: {optimizer.momentum_buffer[0]:.6f}, {optimizer.momentum_buffer[1]:.6f}")
