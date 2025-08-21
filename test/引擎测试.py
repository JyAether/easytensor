import numpy as np
import matplotlib.pyplot as plt

from core.v1.nn import MLP
from core.v1.engine import Node
import torch
import torch.nn as nn
import torch.optim as optim
from test.my_custom_dataset import X_train
from test.my_custom_dataset import y_train
from test.my_custom_dataset import target_function
from core.v1.torch_mlp import TorchMLP

# 使用苹方字体显示中文

plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']  # 或 ['PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# 创建网络：2输入 -> 8隐藏 -> 8隐藏 -> 1输出
net_custom = MLP(2, [8, 8, 1])
losses_custom = []
net_torch = TorchMLP()


def train_easy_grad_engine():
    print("\n=== 方法1：自制自动微分引擎 ===")
    print(f"自制网络结构: {net_custom}")
    print(f"参数总数: {len(net_custom.parameters())}")

    # 训练参数
    learning_rate = 0.01
    epochs = 20

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


losses_torch = []


def train_pytorch_engine():
    print("\n=== 方法2：PyTorch实现 ===")

    # 转换数据为torch tensor
    X_torch = torch.FloatTensor(X_train)
    y_torch = torch.FloatTensor(y_train).reshape(-1, 1)

    # 训练参数
    learning_rate = 0.01
    epochs = 20

    net_torch = TorchMLP()
    print(f"PyTorch网络结构:\n{net_torch}")

    # 计算参数总数
    total_params = sum(p.numel() for p in net_torch.parameters())
    print(f"参数总数: {total_params}")

    # 定义优化器和损失函数
    optimizer = optim.SGD(net_torch.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    print("\n开始训练...")
    for epoch in range(epochs):
        # 前向传播
        y_pred = net_torch(X_torch)
        loss = criterion(y_pred, y_torch)
        losses_torch.append(loss.item())

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 40 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: Loss = {loss.item():.6f}")

    print("PyTorch训练完成!")


def perfermance_calculte_test():
    print("\n=== 性能对比分析 ===")

    # 最终损失对比
    final_loss_custom = losses_custom[-1]
    final_loss_torch = losses_torch[-1]

    print(f"最终训练损失:")
    print(f"  自制引擎: {final_loss_custom:.6f}")
    print(f"  PyTorch:   {final_loss_torch:.6f}")
    print(f"  差异:     {abs(final_loss_custom - final_loss_torch):.6f}")

    # 测试集评估
    test_X = np.random.uniform(-2, 2, (50, 2))
    test_y = target_function(test_X[:, 0], test_X[:, 1])

    # 自制引擎预测
    custom_predictions = []
    for i in range(len(test_X)):
        x = [Node(test_X[i, 0]), Node(test_X[i, 1])]
        pred = net_custom(x)
        custom_predictions.append(pred.data)
    custom_predictions = np.array(custom_predictions)

    # PyTorch预测
    with torch.no_grad():
        torch_predictions = net_torch(torch.FloatTensor(test_X)).numpy().flatten()

    # 计算测试误差
    custom_mse = np.mean((custom_predictions - test_y) ** 2)
    torch_mse = np.mean((torch_predictions - test_y) ** 2)

    print(f"\n测试集MSE:")
    print(f"  自制引擎: {custom_mse:.6f}")
    print(f"  PyTorch:   {torch_mse:.6f}")

    # 可视化对比
    plt.figure(figsize=(15, 5))

    # 损失曲线对比
    plt.subplot(1, 3, 1)
    plt.plot(losses_custom, label='自制引擎', alpha=0.8)
    plt.plot(losses_torch, label='PyTorch', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失对比')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

    # 预测结果对比
    plt.subplot(1, 3, 2)
    plt.scatter(test_y, custom_predictions, alpha=0.6, label='自制引擎')
    plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'r--')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('自制引擎预测 vs 真实值')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.scatter(test_y, torch_predictions, alpha=0.6, label='PyTorch', color='orange')
    plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'r--')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('PyTorch预测 vs 真实值')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def grade_calculate_test():
    print("\n=== 梯度计算验证 ===")

    # 选择一个测试点
    test_x = [1.0, -0.5]

    # 自制引擎的梯度计算
    x_custom = [Node(test_x[0]), Node(test_x[1])]
    y_custom = net_custom(x_custom)
    y_custom.backward()

    custom_grad_x1 = x_custom[0].grad
    custom_grad_x2 = x_custom[1].grad

    print(f"自制引擎在点({test_x[0]}, {test_x[1]})处的梯度:")
    print(f"  ∂y/∂x1 = {custom_grad_x1:.6f}")
    print(f"  ∂y/∂x2 = {custom_grad_x2:.6f}")

    # PyTorch的梯度计算
    x_torch = torch.tensor([test_x], requires_grad=True, dtype=torch.float32)
    y_torch_val = net_torch(x_torch)
    y_torch_val.backward()

    torch_grad = x_torch.grad[0]

    print(f"PyTorch在相同点处的梯度:")
    print(f"  ∂y/∂x1 = {torch_grad[0].item():.6f}")
    print(f"  ∂y/∂x2 = {torch_grad[1].item():.6f}")

    print(f"\n梯度差异:")
    print(f"  Δ(∂y/∂x1) = {abs(custom_grad_x1 - torch_grad[0].item()):.8f}")
    print(f"  Δ(∂y/∂x2) = {abs(custom_grad_x2 - torch_grad[1].item()):.8f}")


def analyze_training_process_detail():
    print("\n=== 训练过程详细分析 ===")

    # 分析单个epoch的计算过程
    print("单个样本的前向传播分析:")

    # 选择第一个训练样本
    sample_idx = 0
    x_sample = X_train[sample_idx]
    y_sample = y_train[sample_idx]

    print(f"样本输入: x1={x_sample[0]:.4f}, x2={x_sample[1]:.4f}")
    print(f"真实输出: y={y_sample:.4f}")

    # 自制引擎的逐层输出
    x_val = [Node(x_sample[0]), Node(x_sample[1])]
    print(f"\n自制引擎逐层计算:")

    # 第一层 (隐藏层1)
    layer1_out = net_custom.layers[0](x_val)
    print(f"第1层输出: {[neuron.data for neuron in layer1_out]}")

    # 第二层 (隐藏层2)
    layer2_out = net_custom.layers[1](layer1_out)
    print(f"第2层输出: {[neuron.data for neuron in layer2_out]}")

    # 第三层 (输出层)
    final_out = net_custom.layers[2](layer2_out)
    print(f"最终输出: {final_out.data:.6f}")

    # 对比PyTorch的输出
    with torch.no_grad():
        x_torch_sample = torch.tensor([x_sample], dtype=torch.float32)
        torch_out = net_torch(x_torch_sample)
        print(f"PyTorch输出: {torch_out.item():.6f}")
        print(f"输出差异: {abs(final_out.data - torch_out.item()):.8f}")

    print(f"\n训练收敛分析:")
    print(f"自制引擎:")
    print(f"  初始损失: {losses_custom[0]:.6f}")
    print(f"  最终损失: {losses_custom[-1]:.6f}")
    print(f"  损失降低: {(losses_custom[0] - losses_custom[-1]) / losses_custom[0] * 100:.2f}%")

    print(f"PyTorch:")
    print(f"  初始损失: {losses_torch[0]:.6f}")
    print(f"  最终损失: {losses_torch[-1]:.6f}")
    print(f"  损失降低: {(losses_torch[0] - losses_torch[-1]) / losses_torch[0] * 100:.2f}%")


print("\n=== 正确的结果分析方法 ===")


def analyze_results_correctly(losses_custom, losses_torch, net_custom, net_torch):
    """正确分析两个引擎的结果"""

    print("1. 收敛性分析:")
    final_custom = losses_custom[-1]
    final_torch = losses_torch[-1]

    print(f"   最终损失: 自制={final_custom:.6f}, PyTorch={final_torch:.6f}")

    # 判断标准：差异应该很小（< 1e-3）
    if abs(final_custom - final_torch) < 1e-3:
        print("   ✓ 两个引擎收敛到相似的损失值")
    else:
        print("   ✗ 收敛结果差异过大，需要检查实现")

    print("\n2. 梯度一致性验证:")
    # 在相同点计算梯度
    test_point = [1.0, -0.5]

    # 自制引擎梯度
    x_custom = [Node(test_point[0]), Node(test_point[1])]
    y_custom = net_custom(x_custom)
    y_custom.backward()

    grad_custom = [x_custom[0].grad, x_custom[1].grad]

    # PyTorch梯度
    x_torch = torch.tensor([test_point], requires_grad=True, dtype=torch.float32)
    y_torch = net_torch(x_torch)
    y_torch.backward()

    grad_torch = x_torch.grad[0].tolist()

    print(f"   自制引擎梯度: [{grad_custom[0]:.6f}, {grad_custom[1]:.6f}]")
    print(f"   PyTorch梯度:  [{grad_torch[0]:.6f}, {grad_torch[1]:.6f}]")

    grad_diff = [abs(grad_custom[i] - grad_torch[i]) for i in range(2)]
    print(f"   梯度差异: [{grad_diff[0]:.8f}, {grad_diff[1]:.8f}]")

    # 判断标准：梯度差异应该很小（< 1e-6）
    if max(grad_diff) < 1e-6:
        print("   ✓ 梯度计算高度一致")
    else:
        print("   ✗ 梯度差异过大")

    print("\n3. 学习能力评估:")
    initial_custom = losses_custom[0]
    initial_torch = losses_torch[0]

    improvement_custom = (initial_custom - final_custom) / initial_custom * 100
    improvement_torch = (initial_torch - final_torch) / initial_torch * 100

    print(f"   自制引擎损失降低: {improvement_custom:.2f}%")
    print(f"   PyTorch损失降低:  {improvement_torch:.2f}%")

    print("\n4. 总体评估:")
    if (abs(final_custom - final_torch) < 1e-3 and
            max(grad_diff) < 1e-6 and
            abs(improvement_custom - improvement_torch) < 5):
        print("   ✓ 两个引擎表现基本一致，自制引擎实现正确")
    else:
        print("   ✗ 存在实现差异，需要进一步调试")


if __name__ == '__main__':
    pass
    train_easy_grad_engine()
    train_pytorch_engine()
    perfermance_calculte_test()
    grade_calculate_test()
    analyze_training_process_detail()

    # 分析修复后的结果
    analyze_results_correctly(losses_custom, losses_torch, net_custom, net_torch)
