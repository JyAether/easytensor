class StepLR:
    """等间隔学习率调度器

    每隔step_size个epoch，学习率乘以gamma
    例如：初始lr=0.1, step_size=3, gamma=0.5
    - Epoch 0-2: lr = 0.1
    - Epoch 3-5: lr = 0.05
    - Epoch 6-8: lr = 0.025
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        """
        参数:
            optimizer: 优化器对象，需要有lr属性
            step_size: 学习率衰减的步长（每多少个epoch衰减一次）
            gamma: 学习率衰减系数
            last_epoch: 上一个epoch的索引，用于恢复训练
        """
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = last_epoch

        # 保存初始学习率
        self.base_lr = optimizer.lr

        # 如果不是从-1开始，立即设置学习率
        if last_epoch != -1:
            self._set_lr()

    def step(self, epoch=None):
        """更新学习率

        参数:
            epoch: 当前epoch，如果为None则自动递增

        返回:
            更新后的学习率
        """
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch

        # 设置新的学习率
        self._set_lr()

        return self.optimizer.lr

    def _set_lr(self):
        """根据当前epoch计算并设置学习率"""
        # 计算当前应该衰减的次数
        decay_times = self.last_epoch // self.step_size

        # 计算新的学习率
        new_lr = self.base_lr * (self.gamma ** decay_times)

        # 更新优化器的学习率
        self.optimizer.lr = new_lr

    def get_lr(self):
        """获取当前学习率（不更新epoch）"""
        decay_times = self.last_epoch // self.step_size
        return self.base_lr * (self.gamma ** decay_times)

    def get_last_lr(self):
        """获取上一次设置的学习率"""
        return self.optimizer.lr

    def state_dict(self):
        """返回调度器状态，用于保存模型"""
        return {
            'step_size': self.step_size,
            'gamma': self.gamma,
            'base_lr': self.base_lr,
            'last_epoch': self.last_epoch
        }

    def load_state_dict(self, state_dict):
        """加载调度器状态，用于恢复模型"""
        self.step_size = state_dict['step_size']
        self.gamma = state_dict['gamma']
        self.base_lr = state_dict['base_lr']
        self.last_epoch = state_dict['last_epoch']
        self._set_lr()