"""
深度学习框架完整使用示例
展示多维数组、矩阵运算、神经网络训练等功能
"""

import numpy as np
import matplotlib.pyplot as plt
from core.tensor import Tensor, tensor, zeros, ones, randn, eye
from core.nn.tensor_nn import Sequential, Linear, ReLU, Sigmoid, Tanh, BatchNorm1d, Dropout
from core.nn.tensor_nn import MSELoss, CrossEntropyLoss, SGD, Adam, init_weights
from core.utils.memory_utils import memory_context, memory_summary, profile_memory_usage, get_memory_monitor

# 设置随机种子
np.random.seed(42)


def demo_basic_tensor_operations():
    """演示基础张量操作"""
    print("=" * 60)
    print("1. 基础张量操作演示")
    print("=" * 60)

    # 创建张量
    a = tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = tensor([[2, 3], [4, 5], [6, 7]], requires_grad=True)

    print(f"张量 a:\n{a}")
    print(f"张量 b:\n{b}")

    # 矩阵乘法
    c = a @ b
    print(f"\na @ b (矩阵乘法):\n{c}")

    # 元素级运算
    d = a * 2
    e = a + 1
    print(f"\na * 2:\n{d}")
    print(f"a + 1:\n{e}")

    # 形状操作
    f = a.reshape(3, 2)
    g = a.transpose()
    print(f"\na.reshape(3, 2):\n{f}")
    print(f"a.transpose():\n{g}")

    # 聚合操作
    h = a.sum()
    i = a.mean(axis=1)
    print(f"\na.sum(): {h.data}")
    print(f"a.mean(axis=1): {i.data}")

    # 反向传播
    loss = c.sum()
    loss.backward()

    print(f"\n反向传播后的梯度:")
    print(f"a.grad:\n{a.grad.data}")
    print(f"b.grad:\n{b.grad.data}")


def demo_activation_functions():
    """演示激活函数"""
    print("=" * 60)
    print("2. 激活函数演示")
    print("=" * 60)

    x = tensor(np.linspace(-5, 5, 11), requires_grad=True)
    print(f"输入 x: {x.data}")

    # ReLU
    relu_out = x.relu()
    print(f"ReLU(x): {relu_out.data}")

    # Sigmoid
    sigmoid_out = x.sigmoid()
    print(f"Sigmoid(x): {sigmoid_out.data}")

    # Tanh
    tanh_out = x.tanh()
    print(f"Tanh(x): {tanh_out.data}")

    # 可视化激活函数
    try:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(x.data, relu_out.data, 'r-', linewidth=2)
        plt.title('ReLU')
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(x.data, sigmoid_out.data, 'g-', linewidth=2)
        plt.title('Sigmoid')
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(x.data, tanh_out.data, 'b-', linewidth=2)
        plt.title('Tanh')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('activation_functions.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("激活函数图像已保存为 activation_functions.png")
    except Exception as e:
        print(f"无法绘制图像: {e}")


@profile_memory_usage
def demo_neural_network():
    """演示神经网络训练"""
    print("=" * 60)
    print("3. 神经网络训练演示")
    print("=" * 60)

    # 生成样本数据（XOR问题）
    def generate_xor_data(n_samples=1000):
        X = np.random.randn(n_samples, 2)
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.float32)
        return X, y.reshape(-1, 1)

    X_train, y_train = generate_xor_data(1000)
    X_test, y_test = generate_xor_data(200)

    print(f"训练数据形状: X={X_train.shape}, y={y_train.shape}")
    print(f"测试数据形状: X={X_test.shape}, y={y_test.shape}")

    # 创建神经网络
    model = Sequential(
        Linear(2, 8),
        ReLU(),
        BatchNorm1d(8),
        Dropout(0.2),
        Linear(8, 4),
        ReLU(),
        Linear(4, 1),
        Sigmoid()
    )

    # 初始化权重
    for layer in model.layers:
        if isinstance(layer, Linear):
            init_weights(layer, 'xavier')

    print(f"模型结构:\n{model}")
    print(f"模型参数数量: {len(model.parameters())}")

    # 定义损失函数和优化器
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=0.01)

    # 训练循环
    epochs = 100
    batch_size = 32
    n_batches = len(X_train) // batch_size

    train_losses = []
    test_losses = []

    print(f"\n开始训练 ({epochs} epochs, batch_size={batch_size})...")

    model.train()  # 设置为训练模式

    for epoch in range(epochs):
        epoch_loss = 0.0

        # 随机打乱数据
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            # 获取批次数据
            X_batch = tensor(X_train_shuffled[start_idx:end_idx], requires_grad=False)
            y_batch = tensor(y_train_shuffled[start_idx:end_idx], requires_grad=False)

            # 前向传播
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.item()

        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)

        # 计算测试损失
        if epoch % 10 == 0:
            model.eval()  # 设置为评估模式
            X_test_tensor = tensor(X_test, requires_grad=False)
            y_test_tensor = tensor(y_test, requires_grad=False)

            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)
            test_losses.append(test_loss.data.item())

            # 计算准确率
            predictions = (test_outputs.data > 0.5).astype(np.float32)
            accuracy = (predictions == y_test).mean()

            print(f"Epoch {epoch:3d}: 训练损失={avg_train_loss:.4f}, "
                  f"测试损失={test_loss.data.item():.4f}, 准确率={accuracy:.4f}")

            model.train()  # 回到训练模式

    print(f"\n训练完成！最终训练损失: {train_losses[-1]:.4f}")

    # 最终评估
    model.eval()
    X_test_tensor = tensor(X_test, requires_grad=False)
    final_outputs = model(X_test_tensor)
    final_predictions = (final_outputs.data > 0.5).astype(np.float32)
    final_accuracy = (final_predictions == y_test).mean()

    print(f"最终测试准确率: {final_accuracy:.4f}")

    # 绘制训练曲线
    try:
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('训练损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        test_epochs = list(range(0, epochs, 10))
        plt.plot(test_epochs, test_losses)
        plt.title('测试损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("训练曲线已保存为 training_curves.png")
    except Exception as e:
        print(f"无法绘制训练曲线: {e}")

    return model, final_accuracy


def demo_batch_operations():
    """演示批量操作"""
    print("=" * 60)
    print("4. 批量操作演示")
    print("=" * 60)

    batch_size = 4
    input_dim = 3
    output_dim = 2

    # 创建批量数据
    X = randn(batch_size, input_dim, requires_grad=False)
    W = randn(input_dim, output_dim, requires_grad=True)
    b = zeros(output_dim, requires_grad=True)

    print(f"批量输入 X 形状: {X.shape}")
    print(f"权重 W 形状: {W.shape}")
    print(f"偏置 b 形状: {b.shape}")

    # 批量线性变换
    Y = X @ W + b
    print(f"输出 Y 形状: {Y.shape}")
    print(f"输出 Y:\n{Y.data}")

    # 批量激活函数
    Y_activated = Y.relu()
    print(f"ReLU后的输出:\n{Y_activated.data}")

    # 批量损失计算
    target = randn(batch_size, output_dim)
    loss = ((Y_activated - target) ** 2).mean()
    print(f"批量MSE损失: {loss.data}")

    # 反向传播
    loss.backward()
    print(f"权重梯度形状: {W.grad.shape}")
    print(f"偏置梯度形状: {b.grad.shape}")


def demo_gpu_operations():
    """演示GPU操作（如果可用）"""
    print("=" * 60)
    print("5. GPU操作演示")
    print("=" * 60)

    try:
        import cupy as cp

        # 创建CPU张量
        cpu_tensor = randn(100, 100, requires_grad=True, device='cpu')
        print(f"CPU张量设备: {cpu_tensor.device}")
        print(f"CPU张量形状: {cpu_tensor.shape}")

        # 移动到GPU
        gpu_tensor = cpu_tensor.cuda()
        print(f"GPU张量设备: {gpu_tensor.device}")
        print(f"GPU张量形状: {gpu_tensor.shape}")

        # GPU上的运算
        gpu_result = gpu_tensor @ gpu_tensor.T
        print(f"GPU矩阵乘法结果形状: {gpu_result.shape}")

        # 移回CPU
        cpu_result = gpu_result.cpu()
        print(f"结果移回CPU，设备: {cpu_result.device}")

        print("GPU操作成功！")

    except ImportError:
        print("CuPy未安装，跳过GPU演示")
    except Exception as e:
        print(f"GPU操作失败: {e}")


def demo_memory_management():
    """演示内存管理"""
    print("=" * 60)
    print("6. 内存管理演示")
    print("=" * 60)

    print("创建大张量前的内存状态:")
    memory_summary()

    with memory_context(monitor=True, clear_on_exit=True):
        print("\n在内存上下文中创建大张量...")
        big_tensors = []

        for i in range(5):
            tensor_data = randn(500, 500)
            big_tensors.append(tensor_data)
            print(f"创建张量 {i + 1}/5, 大小: {tensor_data.data.nbytes / 1024 ** 2:.2f} MB")

        print("\n创建大张量后的内存状态:")
        memory_summary()

    print("\n退出内存上下文后的内存状态:")
    memory_summary()


def benchmark_performance():
    """性能基准测试"""
    print("=" * 60)
    print("7. 性能基准测试")
    print("=" * 60)

    import time

    sizes = [100, 200, 500, 1000]
    times = {'numpy': [], 'tensor': []}

    for size in sizes:
        print(f"\n测试矩阵大小: {size}x{size}")

        # NumPy基准测试
        np_a = np.random.randn(size, size).astype(np.float32)
        np_b = np.random.randn(size, size).astype(np.float32)

        start_time = time.time()
        np_result = np_a @ np_b
        np_time = time.time() - start_time
        times['numpy'].append(np_time)

        print(f"NumPy时间: {np_time:.4f}s")

        # 张量基准测试
        tensor_a = tensor(np_a, requires_grad=True)
        tensor_b = tensor(np_b, requires_grad=True)

        start_time = time.time()
        tensor_result = tensor_a @ tensor_b
        tensor_time = time.time() - start_time
        times['tensor'].append(tensor_time)

        print(f"Tensor时间: {tensor_time:.4f}s")
        print(f"速度比: {tensor_time / np_time:.2f}x")

    # 绘制性能对比
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, times['numpy'], 'o-', label='NumPy', linewidth=2)
        plt.plot(sizes, times['tensor'], 's-', label='Tensor', linewidth=2)
        plt.xlabel('矩阵大小')
        plt.ylabel('时间 (秒)')
        plt.title('矩阵乘法性能对比')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.savefig('performance_benchmark.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("性能对比图已保存为 performance_benchmark.png")
    except Exception as e:
        print(f"无法绘制性能对比图: {e}")


def main():
    """主函数"""
    print("深度学习框架完整功能演示")
    print("==" * 30)

    # 记录开始时的内存状态
    monitor = get_memory_monitor()
    initial_memory = monitor.record_memory()

    try:
        # 1. 基础张量操作
        demo_basic_tensor_operations()

        # 2. 激活函数演示
        demo_activation_functions()

        # 3. 神经网络训练
        model, accuracy = demo_neural_network()

        # 4. 批量操作
        demo_batch_operations()

        # 5. GPU操作
        demo_gpu_operations()

        # 6. 内存管理
        demo_memory_management()

        # 7. 性能基准测试
        benchmark_performance()

        print("\n" + "=" * 60)
        print("演示完成总结:")
        print("=" * 60)
        print(f"✓ 基础张量操作: 支持多维数组、矩阵运算、自动微分")
        print(f"✓ 激活函数: ReLU, Sigmoid, Tanh")
        print(f"✓ 神经网络: 全连接层、批归一化、Dropout")
        print(f"✓ 优化器: SGD, Adam")
        print(f"✓ 损失函数: MSE, CrossEntropy")
        print(f"✓ 训练示例: XOR问题求解，准确率: {accuracy:.3f}")
        print(f"✓ 内存管理: 自动清理、使用监控")
        print(f"✓ GPU支持: CUDA加速（如果可用）")

    except KeyboardInterrupt:
        print("\n用户中断了演示")
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 记录最终内存状态
        final_memory = monitor.record_memory()
        memory_diff = final_memory['cpu']['rss'] - initial_memory['cpu']['rss']

        print("\n" + "=" * 60)
        print("最终内存摘要:")
        print("=" * 60)
        memory_summary()

        print(f"\n总内存变化: {memory_diff / 1024 ** 2:+.2f} MB")
        print("演示结束。")


if __name__ == "__main__":
    main()