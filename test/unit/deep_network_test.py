import numpy as np
from core.v1.optim import Adam
from core.v1.engine import Node
from core.v1.nn import Module,Layer

print("创建 2-4-4-2 的深层网络测试优化器...")

np.random.seed(42)
n_samples = 100
X_train = np.random.uniform(-2, 2, (n_samples, 2))

def target_function(x1, x2):
    return 0.5*np.sin(x1) + 0.3*x2**2 + 0.1*x1*x2 + 0.2

y_train =  np.random.randn(n_samples,2)

class DeepNetwork(Module):
    def __init__(self):
        self.layer1 = Layer(2, 4, activation='relu')  # 输入层到第一隐藏层
        self.layer2 = Layer(4, 4, activation='relu')  # 第一到第二隐藏层
        self.layer3 = Layer(4, 2, activation='sigmoid')  # 第二隐藏层到输出层

    def __call__(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def parameters(self):
        return self.layer1.parameters() + self.layer2.parameters() + self.layer3.parameters()


# 创建深层网络
deep_net = DeepNetwork()
print(f"网络参数数量: {len(deep_net.parameters())}")

# 使用Adam优化器训练深层网络
adam_optimizer = Adam(deep_net.parameters(), lr=0.01, betas=(0.9, 0.999))

print("\n使用Adam优化器训练深层网络...")
for epoch in range(1000):
    inputs = [Node(X_train[0][0]), Node(X_train[0][1])]
    targets = [Node(y_train[0][0]), Node(y_train[0][1])]

    outputs = deep_net(inputs)

    # 计算损失
    if isinstance(outputs, list):
        loss1 = (outputs[0] - targets[0]) ** 2
        loss2 = (outputs[1] - targets[1]) ** 2
        total_loss = (loss1 + loss2) * 0.5
    else:
        # 单个输出的情况
        total_loss = (outputs - targets[0]) ** 2

    # 反向传播和优化
    adam_optimizer.zero_grad()
    total_loss.backward()
    adam_optimizer.step()

    if epoch % 200 == 0 or epoch == 999:
        if isinstance(outputs, list):
            print(
                f"Epoch {epoch:4d}: Loss = {total_loss.data:.6f}, Output = [{outputs[0].data:.4f}, {outputs[1].data:.4f}]")
        else:
            print(f"Epoch {epoch:4d}: Loss = {total_loss.data:.6f}, Output = {outputs.data:.4f}")

print("\n优化器功能总结:")
print("✓ SGD: 基础随机梯度下降")
print("✓ SGD + Momentum: 使用正确的动量公式 Dt = β * St-1 + (1-β) * Wt")
print("  - 减少梯度震荡，加速收敛")
print("  - β控制历史梯度的权重")
print("✓ Adam: 自适应学习率，结合动量和RMSprop的优点")

print("\nMomentum算法原理:")
print("- 通过指数加权平均平滑梯度变化")
print("- β接近1时更重视历史信息，β接近0时更重视当前梯度")
print("- 典型值: β = 0.9 (相当于平均最近10次梯度)")
print("- 有效减少在峡谷和鞍点处的震荡")