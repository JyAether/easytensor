import numpy as np
from core.v1.nn import Node
from test.network.custom_network import CustomNetwork
from core.v1.optim.adam import Adam

print("=== 方法1：自制自动微分引擎 ===")

# 创建网络：2输入 -> 2隐藏-> 1输出
# 创建自定义网络
net = CustomNetwork()
net.print_weights()

# 输入数据
X_train = np.array([[0.05, 0.10]])
y_train = np.array([[0.01, 0.99]])

print(f"\n输入: {X_train[0]}")
print(f"目标输出: {y_train[0]}")

# 测试前向传播
x_input = [Node(X_train[0][0]), Node(X_train[0][1])]
output = net(x_input)
print(f"初始预测: [{output[0].data:.4f}, {output[1].data:.4f}]")

# 训练参数
learning_rate = 0.5
epochs = 1

optimizer = Adam(net.parameters(), lr=0.5, betas=(0.9, 0.99))

print(f"\n=== 开始训练 (学习率: {learning_rate}, 轮数: {epochs}) ===")

losses = []
for epoch in range(epochs):
    # 转换输入为Node对象
    inputs = [Node(X_train[0][0]), Node(X_train[0][1])]
    targets = [Node(y_train[0][0]), Node(y_train[0][1])]

    # 前向传播
    outputs = net(inputs)

    # 计算损失 (MSE)
    loss1 = (outputs[0] - targets[0]) ** 2
    loss2 = (outputs[1] - targets[1]) ** 2
    total_loss = (loss1 + loss2) * 0.5  # 总损失的一半

    losses.append(total_loss.data)

    # 反向传播
    # net.zero_grad()
    optimizer.zero_grad()
    total_loss.backward()

    print("===" * 10)
    # 参数更新
    optimizer.step()
    # for param in net.parameters():
    #     param.data -= learning_rate * param.grad

    # 打印训练进度
    if epoch % 2000 == 0 or epoch == epochs - 1:
        print(
            f"Epoch {epoch:5d}: Loss = {total_loss.data:.6f}, Output = [{outputs[0].data:.4f}, {outputs[1].data:.4f}]")

print("\n=== 训练完成 ===")
net.print_weights()

# 最终测试
inputs_test = [Node(0.05), Node(0.10)]
final_outputs = net(inputs_test)
print(f"\n最终预测: [{final_outputs[0].data:.4f}, {final_outputs[1].data:.4f}]")
print(f"目标输出: [0.0100, 0.9900]")
print(f"误差: [{abs(final_outputs[0].data - 0.01):.4f}, {abs(final_outputs[1].data - 0.99):.4f}]")

print(f"\n损失变化: {losses[0]:.6f} -> {losses[-1]:.6f}")
print(f"损失diff: {abs(losses[0] - losses[-1]):.6f}")

print(f'\n{net.parameters()}')
print("自制引擎训练完成!")
