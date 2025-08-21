from core.v1.optim import Adam
from core.v1.engine import Node
from core.v1.nn import Module,Layer
from core.v1.nn2 import CrossEntropyLoss
# 创建一个3分类网络
class MultiClassNetwork(Module):
    def __init__(self):
        # 2个输入 -> 4个隐藏 -> 3个输出（3分类）
        self.hidden_layer = Layer(2, 4, activation='relu')
        self.output_layer = Layer(4, 3, activation='linear')  # 输出logits，不用softmax

    def __call__(self, x):
        hidden = self.hidden_layer(x)
        logits = self.output_layer(hidden)
        return logits

    def parameters(self):
        return self.hidden_layer.parameters() + self.output_layer.parameters()


# 创建多分类训练数据
print("创建3分类数据集:")
X_multiclass = [
    [Node(0.1), Node(0.2)],  # 样本1 -> 类别0
    [Node(0.8), Node(0.1)],  # 样本2 -> 类别1
    [Node(0.2), Node(0.9)],  # 样本3 -> 类别2
    [Node(0.1), Node(0.1)],  # 样本4 -> 类别0
]
y_multiclass = [0, 1, 2, 0]  # 对应的真实类别

for i, (x, y) in enumerate(zip(X_multiclass, y_multiclass)):
    print(f"  样本{i + 1}: 输入[{x[0].data:.1f}, {x[1].data:.1f}] -> 类别{y}")

# 创建网络和训练组件
net_multiclass = MultiClassNetwork()
criterion = CrossEntropyLoss()
optimizer = Adam(net_multiclass.parameters(), lr=0.01)

print(f"\n网络参数数量: {len(net_multiclass.parameters())}")
print("开始多分类训练...")

epochs = 1000
for epoch in range(epochs):
    total_loss = Node(0.0)
    correct_predictions = 0

    # 遍历所有训练样本
    for inputs, target in zip(X_multiclass, y_multiclass):
        # 重新创建输入节点以避免梯度累积
        fresh_inputs = [Node(inputs[0].data), Node(inputs[1].data)]

        # 前向传播
        logits = net_multiclass(fresh_inputs)

        # 计算损失
        loss = criterion(logits, target)
        total_loss = total_loss + loss

        # 计算准确率
        predicted_class = max(range(len(logits)), key=lambda i: logits[i].data)
        if predicted_class == target:
            correct_predictions += 1

    # 平均损失
    avg_loss = total_loss * Node(1.0 / len(X_multiclass))
    accuracy = correct_predictions / len(X_multiclass)

    # 反向传播
    optimizer.zero_grad()
    avg_loss.backward()
    optimizer.step()

    # 打印训练进度
    if epoch % 200 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:4d}: Loss = {avg_loss.data:.6f}, 准确率 = {accuracy:.3f}")

# 最终测试
print("\n=== 最终测试结果 ===")
for i, (inputs, target) in enumerate(zip(X_multiclass, y_multiclass)):
    fresh_inputs = [Node(inputs[0].data), Node(inputs[1].data)]
    logits = net_multiclass(fresh_inputs)

    # 计算softmax概率
    probabilities = criterion._softmax(logits)
    predicted_class = max(range(len(logits)), key=lambda j: logits[j].data)

    print(f"样本{i + 1}: 输入[{inputs[0].data:.1f}, {inputs[1].data:.1f}]")
    print(f"  真实类别: {target}, 预测类别: {predicted_class}")
    print(f"  Logits: [{', '.join(f'{l.data:.3f}' for l in logits)}]")
    print(f"  概率: [{', '.join(f'{p.data:.3f}' for p in probabilities)}]")
    print()


