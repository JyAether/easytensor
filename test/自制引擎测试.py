from core.v1.nn import MLP
from core.v1.nn import Node
from test.my_custom_dataset import X_train
from test.my_custom_dataset import y_train

print("=== 方法1：自制自动微分引擎 ===")

# 创建网络：2输入 -> 8隐藏 -> 8隐藏 -> 1输出
net_custom = MLP(2, [8, 8, 1])
print(f"自制网络结构: {net_custom}")
print(f"参数总数: {len(net_custom.parameters())}")

# 训练参数
learning_rate = 0.01
epochs = 200
losses_custom = []

print("\n开始训练...")
for epoch in range(epochs):
    total_loss = Node(0.0)

    # 对每个训练样本计算损失
    for i in range(len(X_train)):
        # 转换为Node对象
        x = [Node(X_train[i, 0]), Node(X_train[i, 1])]
        y_true = Node(y_train[i])

        # 前向传播
        y_pred = net_custom(x)

        # 计算损失 (MSE)
        diff = y_pred - y_true
        loss = diff * diff
        total_loss = total_loss + loss

    # 平均损失
    avg_loss = total_loss * Node(1.0 / len(X_train))
    losses_custom.append(avg_loss.data)

    # 反向传播
    net_custom.zero_grad()
    avg_loss.backward()

    # 手动梯度下降
    for param in net_custom.parameters():
        param.data -= learning_rate * param.grad

    if epoch % 40 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d}: Loss = {avg_loss.data:.6f}")

print("自制引擎训练完成!")