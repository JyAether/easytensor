import random as python_random
from core.v1.engine import Node
from core.v1.nn2.module.batchnorm import BatchNorm
from core.v1.nn2.module.lr_scheduler import StepLR


class MLPWithBatchNorm:
    """带BatchNorm的多层感知机"""

    def __init__(self, layer_sizes, use_bn=True, activation='relu'):
        """
        参数:
            layer_sizes: 各层大小，例如 [784, 256, 128, 10]
            use_bn: 是否使用BatchNorm
            activation: 激活函数类型
        """
        self.layers = []
        self.batch_norms = []
        self.use_bn = use_bn
        self.activation = activation

        # 构建网络层
        for i in range(len(layer_sizes) - 1):
            # 创建线性层
            layer = self._create_linear_layer(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)

            # 为隐藏层添加BatchNorm（不在输出层添加）
            if use_bn and i < len(layer_sizes) - 2:
                bn = BatchNorm(layer_sizes[i + 1])
                self.batch_norms.append(bn)
            else:
                self.batch_norms.append(None)

    def _create_linear_layer(self, input_size, output_size):
        """创建线性层"""
        # Xavier初始化
        std = (2.0 / input_size) ** 0.5

        weights = []
        for _ in range(output_size):
            row = []
            for _ in range(input_size):
                w = python_random.gauss(0, std)
                row.append(Node(w))
            weights.append(row)

        bias = [Node(0.0) for _ in range(output_size)]

        return {'weights': weights, 'bias': bias}

    def __call__(self, x):
        """前向传播"""
        # 确保输入是批次格式
        if isinstance(x[0], Node):
            x = [x]  # 单个样本转为批次
            single_sample = True
        else:
            single_sample = False

        # 逐层前向传播
        current_input = x

        for i, (layer, bn) in enumerate(zip(self.layers, self.batch_norms)):
            # 线性变换
            linear_output = self._apply_linear(current_input, layer)

            # 批归一化（如果有）
            if bn is not None:
                linear_output = bn(linear_output)

            # 激活函数（除了最后一层）
            if i < len(self.layers) - 1:
                if self.activation == 'relu':
                    current_input = self._apply_relu(linear_output)
                elif self.activation == 'tanh':
                    current_input = self._apply_tanh(linear_output)
                else:
                    current_input = linear_output
            else:
                current_input = linear_output

        return current_input[0] if single_sample else current_input

    def _apply_linear(self, x, layer):
        """应用线性变换"""
        output = []
        for sample in x:
            sample_output = []
            for i, (weight_row, bias_val) in enumerate(zip(layer['weights'], layer['bias'])):
                # 计算 w·x + b
                weighted_sum = bias_val
                for w, xi in zip(weight_row, sample):
                    weighted_sum = weighted_sum + w * xi
                sample_output.append(weighted_sum)
            output.append(sample_output)
        return output

    def _apply_relu(self, x):
        """应用ReLU激活函数"""
        return [[node.relu() for node in sample] for sample in x]

    def _apply_tanh(self, x):
        """应用Tanh激活函数"""
        return [[node.tanh() for node in sample] for sample in x]

    def parameters(self):
        """返回所有可学习参数"""
        params = []

        # 添加线性层参数
        for layer in self.layers:
            for weight_row in layer['weights']:
                params.extend(weight_row)
            params.extend(layer['bias'])

        # 添加BatchNorm参数
        for bn in self.batch_norms:
            if bn is not None:
                params.extend(bn.parameters())

        return params

    def train(self):
        """设置为训练模式"""
        for bn in self.batch_norms:
            if bn is not None:
                bn.train()

    def eval(self):
        """设置为评估模式"""
        for bn in self.batch_norms:
            if bn is not None:
                bn.eval()

    def zero_grad(self):
        """清零所有参数的梯度"""
        for param in self.parameters():
            param.grad = 0


def generate_classification_data(n_samples=100, n_features=10, n_classes=3):
    """生成分类数据用于测试"""
    data = []
    labels = []

    for _ in range(n_samples):
        # 生成随机特征
        features = [Node(python_random.gauss(0, 1)) for _ in range(n_features)]
        data.append(features)

        # 生成随机标签（one-hot编码）
        label_idx = python_random.randint(0, n_classes - 1)
        label = [Node(1.0 if i == label_idx else 0.0) for i in range(n_classes)]
        labels.append(label)

    return data, labels


def compute_mse_loss(predictions, targets):
    """计算均方误差损失"""
    total_loss = Node(0.0)

    for pred_batch, target_batch in zip(predictions, targets):
        for pred, target in zip(pred_batch, target_batch):
            diff = pred - target
            total_loss = total_loss + diff * diff

    return total_loss / len(predictions)


def compute_accuracy(predictions, targets):
    """计算分类准确率"""
    correct = 0
    total = 0

    for pred_batch, target_batch in zip(predictions, targets):
        # 找到预测的最大值索引
        pred_idx = max(range(len(pred_batch)), key=lambda i: pred_batch[i].data)
        # 找到真实标签索引
        target_idx = max(range(len(target_batch)), key=lambda i: target_batch[i].data)

        if pred_idx == target_idx:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def train_with_step_lr_and_bn():
    """使用StepLR和BatchNorm的完整训练示例"""
    print("=== 使用StepLR和BatchNorm的训练示例 ===")

    # 1. 创建模型
    model = MLPWithBatchNorm(
        layer_sizes=[10, 32, 16, 3],  # 输入10维，输出3类
        use_bn=True,
        activation='relu'
    )

    print(f"模型参数数量: {len(model.parameters())}")

    # 2. 创建优化器（简化的Adam）
    class SimpleOptimizer:
        def __init__(self, parameters, lr=0.01):
            self.parameters = parameters
            self.lr = lr

        def step(self):
            # 简单的梯度下降更新
            for param in self.parameters:
                param.data -= self.lr * param.grad

        def zero_grad(self):
            for param in self.parameters:
                param.grad = 0

    optimizer = SimpleOptimizer(model.parameters(), lr=0.01)

    # 3. 创建学习率调度器
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # 4. 生成训练和测试数据
    print("\n生成数据...")
    train_x, train_y = generate_classification_data(200, 10, 3)
    test_x, test_y = generate_classification_data(50, 10, 3)

    print(f"训练数据: {len(train_x)} 样本")
    print(f"测试数据: {len(test_x)} 样本")

    # 5. 训练循环
    print("\n开始训练...")
    print("Epoch | Train Loss | Test Acc | Learning Rate | BN Stats")
    print("-" * 65)

    batch_size = 20
    num_epochs = 30

    for epoch in range(num_epochs):
        model.train()

        # 训练阶段
        total_train_loss = 0
        num_batches = 0

        # 分批训练
        for i in range(0, len(train_x), batch_size):
            batch_x = train_x[i:i + batch_size]
            batch_y = train_y[i:i + batch_size]

            # 前向传播
            pred = model(batch_x)
            loss = compute_mse_loss([pred], [batch_y])

            total_train_loss += loss.data
            num_batches += 1

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / num_batches

        # 测试阶段
        model.eval()
        test_predictions = []
        for test_sample in test_x:
            pred = model(test_sample)
            test_predictions.append(pred)

        test_acc = compute_accuracy(test_predictions, test_y)

        # 更新学习率
        current_lr = scheduler.step()

        # 获取BatchNorm统计信息
        first_bn = next((bn for bn in model.batch_norms if bn is not None), None)
        bn_mean = first_bn.running_mean[0] if first_bn else 0
        bn_var = first_bn.running_var[0] if first_bn else 0

        # 打印训练信息
        if epoch % 5 == 0 or epoch < 3:
            print(f"{epoch:5d} | {avg_train_loss:10.4f} | {test_acc:8.2%} | "
                  f"{current_lr:11.6f} | μ={bn_mean:.3f},σ²={bn_var:.3f}")

    print(f"\n训练完成！最终测试准确率: {test_acc:.2%}")

    return model, scheduler


def compare_with_without_bn():
    """比较使用和不使用BatchNorm的效果"""
    print("\n=== BatchNorm效果对比 ===")

    # 生成相同的数据
    train_x, train_y = generate_classification_data(150, 8, 2)
    test_x, test_y = generate_classification_data(30, 8, 2)

    configs = [
        {"name": "不使用BatchNorm", "use_bn": False},
        {"name": "使用BatchNorm", "use_bn": True}
    ]

    results = {}

    for config in configs:
        print(f"\n训练 {config['name']}...")

        # 创建模型
        model = MLPWithBatchNorm(
            layer_sizes=[8, 16, 8, 2],
            use_bn=config['use_bn'],
            activation='relu'
        )

        # 简单优化器
        class SimpleOptimizer:
            def __init__(self, parameters, lr=0.02):
                self.parameters = parameters
                self.lr = lr

            def step(self):
                for param in self.parameters:
                    param.data -= self.lr * param.grad

            def zero_grad(self):
                for param in self.parameters:
                    param.grad = 0

        optimizer = SimpleOptimizer(model.parameters(), lr=0.02)
        scheduler = StepLR(optimizer, step_size=8, gamma=0.8)

        # 训练
        train_losses = []
        test_accs = []

        for epoch in range(15):
            model.train()

            # 训练一个epoch
            total_loss = 0
            for i in range(0, len(train_x), 10):
                batch_x = train_x[i:i + 10]
                batch_y = train_y[i:i + 10]

                pred = model(batch_x)
                loss = compute_mse_loss([pred], [batch_y])

                total_loss += loss.data

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / (len(train_x) // 10)
            train_losses.append(avg_loss)

            # 测试
            model.eval()
            test_preds = [model(sample) for sample in test_x]
            test_acc = compute_accuracy(test_preds, test_y)
            test_accs.append(test_acc)

            scheduler.step()

        results[config['name']] = {
            'final_loss': train_losses[-1],
            'final_acc': test_accs[-1],
            'best_acc': max(test_accs)
        }

        print(f"最终损失: {train_losses[-1]:.4f}")
        print(f"最终准确率: {test_accs[-1]:.2%}")
        print(f"最佳准确率: {max(test_accs):.2%}")

    # 比较结果
    print(f"\n=== 对比结果 ===")
    for name, result in results.items():
        print(f"{name:15s}: 损失={result['final_loss']:.4f}, "
              f"准确率={result['final_acc']:.2%}, 最佳={result['best_acc']:.2%}")


def demonstrate_lr_scheduling():
    """演示学习率调度的效果"""
    print("\n=== 学习率调度效果演示 ===")

    # 创建简单模型
    model = MLPWithBatchNorm([5, 10, 3], use_bn=True)

    class SimpleOptimizer:
        def __init__(self, parameters, lr=0.1):
            self.parameters = parameters
            self.lr = lr

        def step(self):
            for param in self.parameters:
                param.data -= self.lr * param.grad

        def zero_grad(self):
            for param in self.parameters:
                param.grad = 0

    # 比较不同调度策略
    schedulers = [
        ("无调度", None),
        ("StepLR(5,0.5)", StepLR),
        ("StepLR(3,0.8)", StepLR)
    ]

    data_x, data_y = generate_classification_data(80, 5, 3)

    for name, scheduler_class in schedulers:
        print(f"\n{name}:")

        # 重新初始化模型参数
        model = MLPWithBatchNorm([5, 10, 3], use_bn=True)
        optimizer = SimpleOptimizer(model.parameters(), lr=0.1)

        if scheduler_class:
            if "5,0.5" in name:
                scheduler = scheduler_class(optimizer, step_size=5, gamma=0.5)
            else:
                scheduler = scheduler_class(optimizer, step_size=3, gamma=0.8)
        else:
            scheduler = None

        print("Epoch | Loss     | LR")
        print("-" * 25)

        for epoch in range(12):
            model.train()

            # 训练
            pred = model(data_x[:20])  # 使用部分数据
            loss = compute_mse_loss([pred], [data_y[:20]])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新学习率
            current_lr = optimizer.lr
            if scheduler:
                scheduler.step()
                current_lr = optimizer.lr

            if epoch % 3 == 0:
                print(f"{epoch:5d} | {loss.data:8.4f} | {current_lr:.6f}")


if __name__ == "__main__":
    # 运行完整训练示例
    model, scheduler = train_with_step_lr_and_bn()

    # 比较BatchNorm效果
    compare_with_without_bn()

    # 演示学习率调度
    demonstrate_lr_scheduling()
