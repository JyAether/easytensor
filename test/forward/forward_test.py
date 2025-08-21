from core.v1.engine import Node

# 案例：简单神经元计算
# y = ReLU(w1*x1 + w2*x2 + b)

print("=== 前向传播过程 ===")

# 输入数据
x1 = Node(2.0, _op='x1')
x2 = Node(3.0, _op='x2')
print(f"输入: x1={x1}, x2={x2}")

# 权重和偏置
w1 = Node(0.5, _op='w1')
w2 = Node(-0.8, _op='w2')
b = Node(0.1, _op='b')
print(f"参数: w1={w1}, w2={w2}, b={b}")

# 第一步：w1 * x1
step1 = w1 * x1
print(f"步骤1: w1*x1 = {step1}")

# 第二步：w2 * x2
step2 = w2 * x2
print(f"步骤2: w2*x2 = {step2}")

# 第三步：加法求和
step3 = step1 + step2
print(f"步骤3: w1*x1 + w2*x2 = {step3}")

# 第四步：加偏置
step4 = step3 + b
print(f"步骤4: w1*x1 + w2*x2 + b = {step4}")

# 第五步：ReLU激活
y = step4.relu()
print(f"步骤5: ReLU({step4.data:.4f}) = {y}")

print(f"\n前向传播完成，最终输出: {y.data}")

print("\n=== 反向传播过程 ===")

# 开始反向传播
y.backward()

print("反向传播完成后的梯度:")
print(f"∂y/∂x1 = {x1.grad:.4f}")
print(f"∂y/∂x2 = {x2.grad:.4f}")
print(f"∂y/∂w1 = {w1.grad:.4f}")
print(f"∂y/∂w2 = {w2.grad:.4f}")
print(f"∂y/∂b = {b.grad:.4f}")

# 分析梯度结果
print(f"\n梯度分析:")
print(f"由于ReLU({step4.data:.4f}) = 0，所有梯度都应该为0")
print(f"这是因为在负值区域，ReLU的导数为0，梯度无法向前传播")