from core.v1.optim.sgd import SGD
from core.v1.engine import Node

# 创建简单示例来演示Momentum算法的工作原理
print("演示Momentum算法的梯度平滑效果:")


class SimpleFunction:
    """简单的二次函数 f(x) = x^2，用于演示优化过程"""

    def __init__(self, x_init=5.0):
        self.x = Node(x_init)

    def forward(self):
        return self.x ** 2

    def parameters(self):
        return [self.x]


# 比较有无Momentum的优化过程
functions = [
    ("无Momentum", SimpleFunction(5.0), SGD([SimpleFunction(5.0).parameters()[0]], lr=0.1, momentum=0.0)),
    ("有Momentum(β=0.9)", SimpleFunction(5.0), SGD([SimpleFunction(5.0).parameters()[0]], lr=0.1, momentum=0.9))
]

for name, func, optimizer in functions:
    print(f"\n{name}:")
    print("步骤  |    x值    |   梯度   |  动量缓存  |   损失")
    print("-" * 50)

    for step in range(8):
        # 前向传播
        loss = func.forward()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 记录当前状态
        x_val = func.x.data
        grad_val = func.x.grad
        momentum_val = optimizer.momentum_buffer[0] if hasattr(optimizer, 'momentum_buffer') else 0.0
        loss_val = loss.data

        print(f"{step:3d}   | {x_val:8.4f} | {grad_val:8.4f} | {momentum_val:8.4f} | {loss_val:8.4f}")

        # 参数更新
        optimizer.step()

        # 如果已经收敛就停止
        if abs(x_val) < 0.01:
            print(f"在第{step + 1}步收敛!")
            break