import torch
import torch.nn as nn
import torch.optim as optim
from test.my_custom_dataset import X_train
from test.my_custom_dataset import y_train
from core.v1.torch_mlp import TorchMLP

print("\n=== 方法2：PyTorch实现 ===")

# 转换数据为torch tensor
X_torch = torch.FloatTensor(X_train)
y_torch = torch.FloatTensor(y_train).reshape(-1, 1)

# 训练参数
learning_rate = 0.01
epochs = 200
losses_custom = []

net_torch = TorchMLP()
print(f"PyTorch网络结构:\n{net_torch}")

# 计算参数总数
total_params = sum(p.numel() for p in net_torch.parameters())
print(f"参数总数: {total_params}")

# 定义优化器和损失函数
optimizer = optim.SGD(net_torch.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

losses_torch = []

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

