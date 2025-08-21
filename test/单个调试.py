import torch
import torch.nn as nn
from core.v1.engine import Node
from core.v1.nn import MLP, Neuron


class EngineDebugger:
    """深度调试自制引擎的问题"""

    def __init__(self):
        pass

    def debug_basic_operations(self):
        """调试基础运算操作"""
        print("=" * 50)
        print("基础运算调试")
        print("=" * 50)

        # 测试加法
        print("\n1. 加法测试:")
        a = Node(2.0)
        b = Node(3.0)
        c = a + b
        c.backward()

        print(f"   计算: {a.data} + {b.data} = {c.data}")
        print(f"   梯度: da/da={a.grad}, db/db={b.grad} (期望都是1.0)")

        if abs(a.grad - 1.0) > 1e-8 or abs(b.grad - 1.0) > 1e-8:
            print("   ❌ 加法反向传播有问题!")
        else:
            print("   ✅ 加法正确")

        # 测试乘法
        print("\n2. 乘法测试:")
        a = Node(2.0)
        b = Node(3.0)
        c = a * b
        c.backward()

        print(f"   计算: {a.data} * {b.data} = {c.data}")
        print(f"   梯度: da/da={a.grad}, db/db={b.grad} (期望: {b.data}, {a.data})")

        if abs(a.grad - b.data) > 1e-8 or abs(b.grad - a.data) > 1e-8:
            print("   ❌ 乘法反向传播有问题!")
        else:
            print("   ✅ 乘法正确")

        # 测试ReLU
        print("\n3. ReLU测试:")

        # 正数情况
        a = Node(2.0)
        b = a.relu()
        b.backward()
        print(f"   ReLU(2.0) = {b.data}, 梯度 = {a.grad} (期望: 2.0, 1.0)")

        # 负数情况
        a = Node(-1.0)
        b = a.relu()
        b.backward()
        print(f"   ReLU(-1.0) = {b.data}, 梯度 = {a.grad} (期望: 0.0, 0.0)")

    def debug_neuron(self):
        """调试单个神经元"""
        print("\n" + "=" * 50)
        print("单神经元调试")
        print("=" * 50)

        # 创建一个简单的神经元
        neuron = Neuron(2, nonlin=False)  # 线性神经元便于验证

        # 手动设置权重便于验证
        neuron.w[0].data = 0.5
        neuron.w[1].data = -0.3
        neuron.b.data = 0.1

        print(f"神经元参数: w1={neuron.w[0].data}, w2={neuron.w[1].data}, b={neuron.b.data}")

        # 前向传播
        x = [Node(1.0), Node(2.0)]
        y = neuron(x)

        expected = 0.5 * 1.0 + (-0.3) * 2.0 + 0.1  # 0.5 - 0.6 + 0.1 = 0.0
        print(f"前向传播: y = {y.data} (期望: {expected})")

        if abs(y.data - expected) > 1e-8:
            print("❌ 前向传播计算错误!")
            return

        # 反向传播
        y.backward()

        print(f"反向传播梯度:")
        print(f"  dx1 = {x[0].grad} (期望: {neuron.w[0].data})")
        print(f"  dx2 = {x[1].grad} (期望: {neuron.w[1].data})")
        print(f"  dw1 = {neuron.w[0].grad} (期望: {x[0].data})")
        print(f"  dw2 = {neuron.w[1].grad} (期望: {x[1].data})")
        print(f"  db  = {neuron.b.grad} (期望: 1.0)")

        # 验证梯度
        errors = []
        if abs(x[0].grad - neuron.w[0].data) > 1e-8:
            errors.append("dx1")
        if abs(x[1].grad - neuron.w[1].data) > 1e-8:
            errors.append("dx2")
        if abs(neuron.w[0].grad - x[0].data) > 1e-8:
            errors.append("dw1")
        if abs(neuron.w[1].grad - x[1].data) > 1e-8:
            errors.append("dw2")
        if abs(neuron.b.grad - 1.0) > 1e-8:
            errors.append("db")

        if errors:
            print(f"❌ 梯度错误: {errors}")
        else:
            print("✅ 单神经元梯度正确")

    def debug_relu_neuron(self):
        """调试ReLU神经元"""
        print("\n" + "=" * 30)
        print("ReLU神经元调试")
        print("=" * 30)

        neuron = Neuron(2, nonlin=True)
        neuron.w[0].data = 1.0
        neuron.w[1].data = -2.0
        neuron.b.data = 0.5

        # 测试激活情况 (输出 > 0)
        print("\n激活情况测试 (输出 > 0):")
        x = [Node(1.0), Node(0.0)]  # 1*1 + (-2)*0 + 0.5 = 1.5 > 0
        y = neuron(x)
        y.backward()

        print(f"输入: x1={x[0].data}, x2={x[1].data}")
        print(f"输出: y={y.data} (期望: 1.5)")
        print(f"梯度: dx1={x[0].grad}, dx2={x[1].grad}")

        # 测试未激活情况 (输出 = 0)
        print("\n未激活情况测试 (输出 = 0):")
        x = [Node(0.0), Node(1.0)]  # 1*0 + (-2)*1 + 0.5 = -1.5 < 0 -> 0
        neuron.zero_grad()
        for xi in x:
            xi.grad = 0

        y = neuron(x)
        y.backward()

        print(f"输入: x1={x[0].data}, x2={x[1].data}")
        print(f"输出: y={y.data} (期望: 0.0)")
        print(f"梯度: dx1={x[0].grad}, dx2={x[1].grad} (期望: 0.0, 0.0)")

    def compare_with_pytorch_step_by_step(self):
        """与PyTorch逐步对比"""
        print("\n" + "=" * 50)
        print("与PyTorch逐步对比")
        print("=" * 50)

        # 创建简单网络进行对比
        print("\n创建简单网络: 2输入 -> 1输出(线性)")

        # 自制引擎网络
        custom_net = MLP(2, [1])  # 2输入 -> 1输出

        # 设置固定权重
        custom_net.layers[0].neurons[0].w[0].data = 0.5
        custom_net.layers[0].neurons[0].w[1].data = -0.3
        custom_net.layers[0].neurons[0].b.data = 0.1

        # PyTorch网络
        torch_net = nn.Linear(2, 1)
        torch_net.weight.data = torch.tensor([[0.5, -0.3]])
        torch_net.bias.data = torch.tensor([0.1])

        print("网络权重已同步")

        # 测试数据
        test_x = [1.0, -2.0]

        # 自制引擎前向传播
        x_custom = [Node(test_x[0]), Node(test_x[1])]
        y_custom = custom_net(x_custom)

        # PyTorch前向传播
        x_torch = torch.tensor([test_x], requires_grad=True, dtype=torch.float32)
        y_torch = torch_net(x_torch)

        print(f"\n前向传播对比:")
        print(f"  输入: {test_x}")
        print(f"  自制引擎输出: {y_custom.data}")
        print(f"  PyTorch输出:  {y_torch.item()}")
        print(f"  差异: {abs(y_custom.data - y_torch.item())}")

        # 反向传播
        y_custom.backward()
        y_torch.backward()

        print(f"\n反向传播对比:")
        print(f"  自制引擎梯度: dx1={x_custom[0].grad:.8f}, dx2={x_custom[1].grad:.8f}")

        x_torch_grad = torch.autograd.grad(y_torch, x_torch, create_graph=True)[0]
        print(f"  PyTorch梯度:   dx1={x_torch_grad[0][0].item():.8f}, dx2={x_torch_grad[0][1].item():.8f}")

        grad_diff1 = abs(x_custom[0].grad - x_torch_grad[0][0].item())
        grad_diff2 = abs(x_custom[1].grad - x_torch_grad[0][1].item())
        print(f"  梯度差异: {grad_diff1:.8f}, {grad_diff2:.8f}")

        if max(grad_diff1, grad_diff2) < 1e-6:
            print("  ✅ 梯度计算一致")
        else:
            print("  ❌ 梯度计算不一致")

    def debug_loss_computation(self):
        """调试损失函数计算"""
        print("\n" + "=" * 50)
        print("损失函数计算调试")
        print("=" * 50)

        # 简单的MSE损失测试
        y_pred = Node(2.0)
        y_true = Node(1.0)

        diff = y_pred - y_true
        loss = diff * diff

        print(f"MSE计算: pred={y_pred.data}, true={y_true.data}")
        print(f"差值: {diff.data}")
        print(f"损失: {loss.data} (期望: 1.0)")

        # 反向传播
        loss.backward()
        print(f"梯度: d_pred={y_pred.grad} (期望: 2.0)")

        if abs(loss.data - 1.0) > 1e-8:
            print("❌ 损失计算错误")
        elif abs(y_pred.grad - 2.0) > 1e-8:
            print("❌ 损失梯度错误")
        else:
            print("✅ 损失计算正确")

    def identify_specific_bugs(self):
        """识别具体的bug"""
        print("\n" + "=" * 50)
        print("潜在Bug分析")
        print("=" * 50)

        print("基于你的测试结果，可能的问题:")
        print("\n1. 反向传播累积问题:")
        print("   - 梯度差异0.23是巨大的，说明反向传播逻辑有根本问题")
        print("   - 检查 Node.__add__ 和 Node.__mul__ 的 _backward 函数")
        print("   - 检查梯度累积是否正确 (+=)")

        print("\n2. ReLU实现问题:")
        print("   - 检查 relu() 函数的梯度计算")
        print("   - 确保 (out.data > 0) 的逻辑正确")

        print("\n3. 拓扑排序问题:")
        print("   - 检查 backward() 中的拓扑排序")
        print("   - 确保按正确顺序处理节点")

        print("\n4. 梯度清零问题:")
        print("   - 检查每次反向传播前是否正确清零")
        print("   - zero_grad() 实现是否正确")

        print("\n5. 数据类型问题:")
        print("   - 检查是否有整数/浮点数混合导致的精度问题")

    def suggest_fixes(self):
        """建议修复方案"""
        print("\n" + "=" * 50)
        print("修复建议")
        print("=" * 50)

        print("建议按以下顺序调试:")
        print("\n1. 首先运行基础运算测试:")
        print("   debugger.debug_basic_operations()")

        print("\n2. 然后测试单神经元:")
        print("   debugger.debug_neuron()")

        print("\n3. 测试ReLU神经元:")
        print("   debugger.debug_relu_neuron()")

        print("\n4. 与PyTorch逐步对比:")
        print("   debugger.compare_with_pytorch_step_by_step()")

        print("\n5. 检查你的Node类实现:")

        # 显示可能的修复
        print("""
可能需要修复的代码位置:

1. Node.__add__ 方法:
   def _backward():
       self.grad += out.grad    # 确保使用 +=
       other.grad += out.grad   # 确保使用 +=

2. Node.__mul__ 方法:
   def _backward():
       self.grad += other.data * out.grad   # 确保使用 +=
       other.grad += self.data * out.grad   # 确保使用 +=

3. Node.relu 方法:
   def _backward():
       self.grad += (out.data > 0) * out.grad  # 确保条件正确

4. 梯度清零:
   def zero_grad(self):
       for p in self.parameters():
           p.grad = 0  # 确保清零
        """)

    def run_complete_diagnosis(self):
        """运行完整诊断"""
        print("开始完整的引擎诊断...")

        self.debug_basic_operations()
        self.debug_neuron()
        self.debug_relu_neuron()
        self.compare_with_pytorch_step_by_step()
        self.debug_loss_computation()
        self.identify_specific_bugs()
        self.suggest_fixes()


# 简化的测试函数
def quick_gradient_test():
    """快速梯度测试"""
    print("快速梯度一致性测试:")
    print("=" * 30)

    # 创建最简单的计算图
    x1 = Node(1.0)
    x2 = Node(2.0)

    # f = x1 * x2 + x1
    y = x1 * x2 + x1
    y.backward()

    print(f"f = x1 * x2 + x1")
    print(f"f = {x1.data} * {x2.data} + {x1.data} = {y.data}")
    print(f"df/dx1 = {x1.grad} (期望: x2 + 1 = {x2.data + 1})")
    print(f"df/dx2 = {x2.grad} (期望: x1 = {x1.data})")

    # 验证
    expected_grad_x1 = x2.data + 1  # 3.0
    expected_grad_x2 = x1.data  # 1.0

    error1 = abs(x1.grad - expected_grad_x1)
    error2 = abs(x2.grad - expected_grad_x2)

    print(f"误差: {error1:.8f}, {error2:.8f}")

    if max(error1, error2) < 1e-6:
        print("✅ 基础梯度计算正确")
        return True
    else:
        print("❌ 基础梯度计算有问题")
        return False


if __name__ == "__main__":
    # 运行快速测试
    if quick_gradient_test():
        print("\n基础功能正常，进行完整诊断:")
        debugger = EngineDebugger()
        debugger.run_complete_diagnosis()
    else:
        print("\n基础功能有问题，建议首先修复Node类的实现")
