import numpy as np
import matplotlib.pyplot as plt
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from core.v1.nn import MLP
from core.v1.engine import Node

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class SimpleEngineComparator:
    """简化版引擎对比器 - 避免复杂的权重同步"""

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def set_seeds(self, seed):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def create_torch_network(self, architecture):
        """创建PyTorch网络"""
        nin, hidden_layers, nout = architecture
        layers = []

        # 构建层
        prev_size = nin
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        # 输出层
        layers.append(nn.Linear(prev_size, nout))

        return nn.Sequential(*layers)

    def train_custom_engine(self, architecture, learning_rate=0.01, epochs=100, seed=42):
        """训练自制引擎"""
        self.set_seeds(seed)

        nin, hidden_layers, nout = architecture
        net = MLP(nin, hidden_layers + [nout])

        losses = []
        start_time = time.time()

        for epoch in range(epochs):
            total_loss = Node(0.0)

            for i in range(len(self.X_train)):
                x = [Node(float(self.X_train[i, 0])), Node(float(self.X_train[i, 1]))]
                y_true = Node(float(self.y_train[i]))

                y_pred = net(x)
                diff = y_pred - y_true
                loss = diff * diff
                total_loss = total_loss + loss

            avg_loss = total_loss * Node(1.0 / len(self.X_train))
            losses.append(avg_loss.data)

            # 反向传播和参数更新
            net.zero_grad()
            avg_loss.backward()

            for param in net.parameters():
                param.data -= learning_rate * param.grad

        training_time = time.time() - start_time

        # 测试集预测
        test_predictions = []
        for i in range(len(self.X_test)):
            x = [Node(float(self.X_test[i, 0])), Node(float(self.X_test[i, 1]))]
            pred = net(x)
            test_predictions.append(pred.data)

        test_mse = np.mean((np.array(test_predictions) - self.y_test) ** 2)

        return {
            'network': net,
            'losses': losses,
            'training_time': training_time,
            'test_mse': test_mse,
            'test_predictions': np.array(test_predictions)
        }

    def train_torch_engine(self, architecture, learning_rate=0.01, epochs=100, seed=42):
        """训练PyTorch引擎"""
        self.set_seeds(seed)

        net = self.create_torch_network(architecture)

        X_torch = torch.FloatTensor(self.X_train)
        y_torch = torch.FloatTensor(self.y_train).reshape(-1, 1)
        X_test_torch = torch.FloatTensor(self.X_test)

        optimizer = optim.SGD(net.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        losses = []
        start_time = time.time()

        for epoch in range(epochs):
            y_pred = net(X_torch)
            loss = criterion(y_pred, y_torch)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        training_time = time.time() - start_time

        # 测试集预测
        with torch.no_grad():
            test_predictions = net(X_test_torch).numpy().flatten()

        test_mse = np.mean((test_predictions - self.y_test) ** 2)

        return {
            'network': net,
            'losses': losses,
            'training_time': training_time,
            'test_mse': test_mse,
            'test_predictions': test_predictions
        }

    def gradient_consistency_test(self, custom_net, torch_net, test_point=[1.0, -0.5]):
        """梯度一致性测试"""
        # 自制引擎梯度
        x_custom = [Node(test_point[0]), Node(test_point[1])]
        for param in custom_net.parameters():
            param.grad = 0

        y_custom = custom_net(x_custom)
        y_custom.backward()

        custom_grads = [x_custom[0].grad, x_custom[1].grad]

        # PyTorch梯度
        x_torch = torch.tensor([test_point], requires_grad=True, dtype=torch.float32)
        y_torch = torch_net(x_torch)
        y_torch.sum().backward()  # 使用sum()确保标量

        torch_grads = x_torch.grad[0].tolist()

        grad_diff = [abs(custom_grads[i] - torch_grads[i]) for i in range(2)]

        return {
            'custom_grads': custom_grads,
            'torch_grads': torch_grads,
            'differences': grad_diff,
            'max_diff': max(grad_diff)
        }

    def comprehensive_comparison(self, architecture=[2, [8, 8], 1],
                                 learning_rate=0.01, epochs=50, n_trials=3):
        """综合对比分析"""

        print("=" * 60)
        print("自制微分引擎 vs PyTorch 综合对比分析")
        print("=" * 60)
        print(f"网络架构: {architecture[0]}输入 -> {architecture[1]} -> {architecture[2]}输出")
        print(f"训练参数: 学习率={learning_rate}, 轮数={epochs}, 试验次数={n_trials}")
        print()

        all_results = {
            'custom': {'losses': [], 'times': [], 'mse': [], 'final_losses': []},
            'torch': {'losses': [], 'times': [], 'mse': [], 'final_losses': []},
            'gradient_tests': []
        }

        for trial in range(n_trials):
            print(f">>> 试验 {trial + 1}/{n_trials}")

            # 使用不同种子进行训练
            seed = 42 + trial

            # 训练自制引擎
            print("  训练自制引擎...")
            custom_result = self.train_custom_engine(architecture, learning_rate, epochs, seed)

            # 训练PyTorch引擎
            print("  训练PyTorch引擎...")
            torch_result = self.train_torch_engine(architecture, learning_rate, epochs, seed)

            # 梯度一致性测试
            print("  梯度一致性测试...")
            grad_result = self.gradient_consistency_test(
                custom_result['network'], torch_result['network']
            )

            # 收集结果
            all_results['custom']['losses'].append(custom_result['losses'])
            all_results['custom']['times'].append(custom_result['training_time'])
            all_results['custom']['mse'].append(custom_result['test_mse'])
            all_results['custom']['final_losses'].append(custom_result['losses'][-1])

            all_results['torch']['losses'].append(torch_result['losses'])
            all_results['torch']['times'].append(torch_result['training_time'])
            all_results['torch']['mse'].append(torch_result['test_mse'])
            all_results['torch']['final_losses'].append(torch_result['losses'][-1])

            all_results['gradient_tests'].append(grad_result)

            # 打印单次结果
            print(f"    最终损失: 自制={custom_result['losses'][-1]:.6f}, "
                  f"PyTorch={torch_result['losses'][-1]:.6f}")
            print(f"    测试MSE:  自制={custom_result['test_mse']:.6f}, "
                  f"PyTorch={torch_result['test_mse']:.6f}")
            print(f"    训练时间: 自制={custom_result['training_time']:.3f}s, "
                  f"PyTorch={torch_result['training_time']:.3f}s")
            print(f"    梯度差异: {grad_result['max_diff']:.8f}")
            print()

        # 统计分析
        self.statistical_analysis(all_results)

        # 可视化结果
        self.visualize_results(all_results, epochs)

        return all_results

    def statistical_analysis(self, results):
        """统计分析结果"""
        print("=" * 40 + " 统计分析 " + "=" * 40)

        # 1. 收敛性分析
        print("\n1. 收敛性分析:")
        custom_final_losses = results['custom']['final_losses']
        torch_final_losses = results['torch']['final_losses']

        custom_mean = np.mean(custom_final_losses)
        custom_std = np.std(custom_final_losses)
        torch_mean = np.mean(torch_final_losses)
        torch_std = np.std(torch_final_losses)

        print(f"   最终损失统计:")
        print(f"     自制引擎: {custom_mean:.6f} ± {custom_std:.6f}")
        print(f"     PyTorch:  {torch_mean:.6f} ± {torch_std:.6f}")
        print(f"     平均差异: {abs(custom_mean - torch_mean):.6f}")

        # 2. 预测性能分析
        print("\n2. 预测性能分析:")
        custom_mse = results['custom']['mse']
        torch_mse = results['torch']['mse']

        custom_mse_mean = np.mean(custom_mse)
        custom_mse_std = np.std(custom_mse)
        torch_mse_mean = np.mean(torch_mse)
        torch_mse_std = np.std(torch_mse)

        print(f"   测试MSE统计:")
        print(f"     自制引擎: {custom_mse_mean:.6f} ± {custom_mse_std:.6f}")
        print(f"     PyTorch:  {torch_mse_mean:.6f} ± {torch_mse_std:.6f}")
        print(f"     相对差异: {abs(custom_mse_mean - torch_mse_mean) / torch_mse_mean * 100:.2f}%")

        # 3. 计算效率分析
        print("\n3. 计算效率分析:")
        custom_times = results['custom']['times']
        torch_times = results['torch']['times']

        custom_time_mean = np.mean(custom_times)
        torch_time_mean = np.mean(torch_times)
        speed_ratio = custom_time_mean / torch_time_mean

        print(f"   平均训练时间:")
        print(f"     自制引擎: {custom_time_mean:.3f}s")
        print(f"     PyTorch:  {torch_time_mean:.3f}s")
        print(f"     速度比率: {speed_ratio:.2f}x (自制/PyTorch)")

        # 4. 梯度一致性分析
        print("\n4. 梯度一致性分析:")
        grad_diffs = [g['max_diff'] for g in results['gradient_tests']]
        grad_mean = np.mean(grad_diffs)
        grad_std = np.std(grad_diffs)

        print(f"   最大梯度差异: {grad_mean:.8f} ± {grad_std:.8f}")

        consistency_rate = sum(1 for d in grad_diffs if d < 1e-6) / len(grad_diffs)
        print(f"   高一致性比率: {consistency_rate * 100:.1f}% (差异 < 1e-6)")

        # 5. 综合评估
        print("\n" + "=" * 20 + " 综合评估 " + "=" * 20)

        # 评估标准
        loss_consistent = abs(custom_mean - torch_mean) < 1e-3
        gradient_consistent = grad_mean < 1e-6
        performance_reasonable = abs(custom_mse_mean - torch_mse_mean) / torch_mse_mean < 0.1

        print(f"✓ 损失一致性: {'通过' if loss_consistent else '未通过'}")
        print(f"✓ 梯度一致性: {'通过' if gradient_consistent else '未通过'}")
        print(f"✓ 性能合理性: {'通过' if performance_reasonable else '未通过'}")

        if loss_consistent and gradient_consistent and performance_reasonable:
            print("\n🎉 结论: 自制引擎实现正确，与PyTorch表现高度一致！")
            print("推荐使用场景:")
            print("  - 学习深度学习原理和自动微分机制")
            print("  - 理解神经网络的底层实现细节")
            print("  - 原型验证和算法研究")
        else:
            print("\n⚠️  结论: 存在一定差异，建议进一步调试")
            if not loss_consistent:
                print("  - 检查损失函数计算")
            if not gradient_consistent:
                print("  - 检查反向传播实现")
            if not performance_reasonable:
                print("  - 检查网络结构对应关系")

    def visualize_results(self, results, epochs):
        """可视化结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 训练损失曲线对比
        ax1 = axes[0, 0]
        for i, (custom_losses, torch_losses) in enumerate(zip(
                results['custom']['losses'], results['torch']['losses']
        )):
            alpha = 0.7 if len(results['custom']['losses']) > 1 else 1.0
            ax1.plot(custom_losses, label=f'自制引擎 试验{i + 1}' if i == 0 else '',
                     color='blue', alpha=alpha, linewidth=1)
            ax1.plot(torch_losses, label=f'PyTorch 试验{i + 1}' if i == 0 else '',
                     color='red', alpha=alpha, linewidth=1)

        ax1.set_xlabel('训练轮数')
        ax1.set_ylabel('损失值')
        ax1.set_title('训练损失曲线对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. 最终损失对比
        ax2 = axes[0, 1]
        custom_final = results['custom']['final_losses']
        torch_final = results['torch']['final_losses']

        x = np.arange(len(custom_final))
        width = 0.35

        ax2.bar(x - width / 2, custom_final, width, label='自制引擎', alpha=0.7)
        ax2.bar(x + width / 2, torch_final, width, label='PyTorch', alpha=0.7)

        ax2.set_xlabel('试验编号')
        ax2.set_ylabel('最终损失值')
        ax2.set_title('最终损失对比')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 测试MSE对比
        ax3 = axes[1, 0]
        custom_mse = results['custom']['mse']
        torch_mse = results['torch']['mse']

        ax3.bar(x - width / 2, custom_mse, width, label='自制引擎', alpha=0.7)
        ax3.bar(x + width / 2, torch_mse, width, label='PyTorch', alpha=0.7)

        ax3.set_xlabel('试验编号')
        ax3.set_ylabel('测试MSE')
        ax3.set_title('测试性能对比')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 梯度差异分析
        ax4 = axes[1, 1]
        grad_diffs = [g['max_diff'] for g in results['gradient_tests']]

        ax4.bar(x, grad_diffs, alpha=0.7, color='green')
        ax4.axhline(y=1e-6, color='red', linestyle='--', label='一致性阈值 (1e-6)')

        ax4.set_xlabel('试验编号')
        ax4.set_ylabel('最大梯度差异')
        ax4.set_title('梯度一致性分析')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')

        plt.tight_layout()
        plt.show()


# 使用示例
def run_comparison_example():
    """运行对比示例"""

    # 生成测试数据
    def target_function(x1, x2):
        return np.sin(x1) * np.cos(x2) + 0.1 * x1 * x2

    # 训练数据
    np.random.seed(123)  # 固定数据生成种子
    X_train = np.random.uniform(-2, 2, (100, 2))
    y_train = target_function(X_train[:, 0], X_train[:, 1])

    # 测试数据
    X_test = np.random.uniform(-2, 2, (50, 2))
    y_test = target_function(X_test[:, 0], X_test[:, 1])

    print("数据准备完成:")
    print(f"  训练集大小: {X_train.shape}")
    print(f"  测试集大小: {X_test.shape}")
    print(f"  目标函数: sin(x1) * cos(x2) + 0.1 * x1 * x2")
    print()

    # 创建对比器
    comparator = SimpleEngineComparator(X_train, y_train, X_test, y_test)

    # 运行对比分析
    results = comparator.comprehensive_comparison(
        architecture=[2, [8, 8], 1],  # 2输入 -> 8隐藏 -> 8隐藏 -> 1输出
        learning_rate=0.01,
        epochs=100,
        n_trials=3
    )

    return results


if __name__ == "__main__":
    results = run_comparison_example()
