from core.v1.nn2.module.lr_scheduler import StepLR
def test_step_lr():
    """测试StepLR调度器"""
    print("=== StepLR调度器测试 ===")

    # 创建一个模拟的优化器
    class MockOptimizer:
        def __init__(self, lr):
            self.lr = lr

        def __repr__(self):
            return f"MockOptimizer(lr={self.lr})"

    # 测试基本功能
    print("1. 基本功能测试:")
    optimizer = MockOptimizer(lr=0.1)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

    print("Epoch | Learning Rate | 说明")
    print("-" * 40)

    for epoch in range(12):
        lr = scheduler.step()
        status = ""
        if epoch % 3 == 0 and epoch > 0:
            status = "← 衰减点"
        print(f"{epoch:5d} | {lr:11.6f} | {status}")

    # 测试不同参数
    print("\n2. 不同参数测试:")
    configs = [
        {"step_size": 2, "gamma": 0.8, "name": "每2个epoch，衰减20%"},
        {"step_size": 5, "gamma": 0.1, "name": "每5个epoch，衰减90%"},
        {"step_size": 1, "gamma": 0.95, "name": "每个epoch，衰减5%"}
    ]

    for config in configs:
        print(f"\n{config['name']}:")
        optimizer = MockOptimizer(lr=0.1)
        scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])

        lrs = []
        for epoch in range(8):
            lr = scheduler.step()
            lrs.append(lr)

        print("Epochs 0-7:", [f"{lr:.4f}" for lr in lrs])

    # 测试状态保存和加载
    print("\n3. 状态保存/加载测试:")
    optimizer = MockOptimizer(lr=0.1)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

    # 运行几个epoch
    for epoch in range(5):
        scheduler.step()

    print(f"第5个epoch后的学习率: {scheduler.get_last_lr():.6f}")

    # 保存状态
    state = scheduler.state_dict()
    print(f"保存的状态: {state}")

    # 创建新的调度器并加载状态
    new_optimizer = MockOptimizer(lr=0.5)  # 不同的初始lr
    new_scheduler = StepLR(new_optimizer, step_size=1, gamma=0.9)  # 不同的参数
    new_scheduler.load_state_dict(state)

    print(f"加载状态后的学习率: {new_scheduler.get_last_lr():.6f}")
    print(f"继续训练第6个epoch: {new_scheduler.step():.6f}")


def usage_example():
    """使用示例"""
    print("\n=== 使用示例 ===")

    # 模拟Adam优化器
    class SimpleAdam:
        def __init__(self, parameters, lr=0.001):
            self.parameters = parameters
            self.lr = lr

        def step(self):
            # 这里应该是实际的参数更新逻辑
            pass

        def zero_grad(self):
            # 这里应该是梯度清零逻辑
            pass

    # 创建优化器和调度器
    model_params = []  # 这里应该是模型参数
    optimizer = SimpleAdam(model_params, lr=0.01)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    print("训练循环示例:")
    print("Epoch | LR")
    print("-" * 15)

    # 模拟训练循环
    num_epochs = 25
    for epoch in range(num_epochs):
        # 训练代码...
        # for batch in dataloader:
        #     optimizer.zero_grad()
        #     loss = model(batch)
        #     loss.backward()
        #     optimizer.step()

        # 更新学习率
        current_lr = scheduler.step()

        if epoch % 5 == 0:
            print(f"{epoch:5d} | {current_lr:.6f}")

    print("\n关键点:")
    print("1. 在每个epoch结束时调用 scheduler.step()")
    print("2. StepLR会自动管理epoch计数")
    print("3. 可以通过 scheduler.get_last_lr() 获取当前学习率")
    print("4. 支持保存和加载状态以便恢复训练")


if __name__ == "__main__":
    test_step_lr()
    usage_example()
