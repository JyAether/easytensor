import numpy as np
import math
from abc import ABC, abstractmethod


class LRScheduler(ABC):
    """学习率调度器基类"""

    def __init__(self, optimizer, last_epoch=-1):
        """
        Args:
            optimizer: 优化器对象
            last_epoch: 上一个epoch的索引，用于恢复训练
        """
        self.optimizer = optimizer
        self.last_epoch = last_epoch

        # 保存初始学习率
        if hasattr(optimizer, 'lr'):
            self.base_lr = optimizer.lr
        else:
            # 如果优化器没有lr属性，尝试从参数组中获取
            self.base_lr = getattr(optimizer, 'lr', 0.01)

        # 如果从checkpoint恢复，需要更新学习率
        if last_epoch != -1:
            self.step()

    @abstractmethod
    def get_lr(self):
        """计算当前epoch的学习率"""
        pass

    def step(self, epoch=None):
        """更新学习率"""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        # 获取新的学习率
        new_lr = self.get_lr()

        # 更新优化器中的学习率
        self.optimizer.lr = new_lr

        return new_lr

    def state_dict(self):
        """返回调度器状态"""
        return {
            'last_epoch': self.last_epoch,
            'base_lr': self.base_lr
        }

    def load_state_dict(self, state_dict):
        """加载调度器状态"""
        self.last_epoch = state_dict['last_epoch']
        self.base_lr = state_dict['base_lr']


class ExponentialLR(LRScheduler):
    """指数衰减学习率调度器（重点实现）

    学习率按照 lr = base_lr * gamma^epoch 进行衰减
    """

    def __init__(self, optimizer, gamma, last_epoch=-1):
        """
        Args:
            optimizer: 优化器
            gamma: 衰减因子，应该 < 1.0，通常在 0.9-0.99 之间
            last_epoch: 上一个epoch的索引
        """
        if gamma >= 1.0:
            raise ValueError(f"ExponentialLR gamma should be < 1.0, got {gamma}")

        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """计算指数衰减后的学习率"""
        if self.last_epoch == 0:
            # 如果指数为0，直接返回，不处理
            return self.base_lr
        return self.base_lr * (self.gamma ** self.last_epoch)

    def __repr__(self):
        return f"ExponentialLR(gamma={self.gamma}, last_epoch={self.last_epoch})"


class StepLR(LRScheduler):
    """固定步长学习率衰减调度器

    每隔step_size个epoch，学习率乘以gamma
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        """
        Args:
            optimizer: 优化器
            step_size: 学习率衰减的步长
            gamma: 衰减因子
            last_epoch: 上一个epoch的索引
        """
        if step_size <= 0:
            raise ValueError(f"StepLR step_size should be > 0, got {step_size}")

        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """计算步长衰减后的学习率"""
        return self.base_lr * (self.gamma ** (self.last_epoch // self.step_size))

    def __repr__(self):
        return f"StepLR(step_size={self.step_size}, gamma={self.gamma}, last_epoch={self.last_epoch})"


class MultiStepLR(LRScheduler):
    """多步长学习率衰减调度器

    在指定的epoch处，学习率乘以gamma
    """

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        """
        Args:
            optimizer: 优化器
            milestones: 学习率衰减的epoch列表
            gamma: 衰减因子
            last_epoch: 上一个epoch的索引
        """
        if not isinstance(milestones, (list, tuple)):
            milestones = [milestones]

        self.milestones = sorted(list(milestones))
        self.gamma = gamma

        # 检查milestones是否有序且为正数
        if len(self.milestones) != len(set(self.milestones)):
            raise ValueError("milestones should be a list of unique values in increasing order")

        for i in range(len(self.milestones) - 1):
            if self.milestones[i] >= self.milestones[i + 1]:
                raise ValueError("milestones should be a list of unique values in increasing order")

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """计算多步长衰减后的学习率"""
        # 计算当前epoch之前有多少个milestone
        decay_count = sum(1 for milestone in self.milestones if milestone <= self.last_epoch)
        return self.base_lr * (self.gamma ** decay_count)

    def __repr__(self):
        return f"MultiStepLR(milestones={self.milestones}, gamma={self.gamma}, last_epoch={self.last_epoch})"


class CosineAnnealingLR(LRScheduler):
    """余弦退火学习率调度器

    学习率按照余弦函数进行衰减，在T_max个epoch后重新开始
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        """
        Args:
            optimizer: 优化器
            T_max: 一个周期的最大epoch数
            eta_min: 最小学习率
            last_epoch: 上一个epoch的索引
        """
        if T_max <= 0:
            raise ValueError(f"T_max should be > 0, got {T_max}")

        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """计算余弦退火后的学习率"""
        if self.last_epoch == 0:
            return self.base_lr
        elif self.last_epoch % self.T_max == 0:
            return self.base_lr

        # 余弦退火公式
        return self.eta_min + (self.base_lr - self.eta_min) * \
            (1 + math.cos(math.pi * (self.last_epoch % self.T_max) / self.T_max)) / 2

    def __repr__(self):
        return f"CosineAnnealingLR(T_max={self.T_max}, eta_min={self.eta_min}, last_epoch={self.last_epoch})"


class ReduceLROnPlateau(LRScheduler):
    """基于验证指标的学习率衰减调度器

    当验证指标停止改善时，降低学习率
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8):
        """
        Args:
            optimizer: 优化器
            mode: 'min' 或 'max'，指标是越小越好还是越大越好
            factor: 学习率衰减因子
            patience: 容忍多少个epoch指标不改善
            threshold: 改善的阈值
            threshold_mode: 'rel' 或 'abs'，相对阈值或绝对阈值
            cooldown: 衰减后等待多少个epoch再开始监控
            min_lr: 最小学习率
            eps: 学习率的最小衰减
        """
        if factor >= 1.0:
            raise ValueError(f"Factor should be < 1.0, got {factor}")

        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps

        # 内部状态
        self.best = None
        self.num_bad_epochs = 0
        self.mode_worse = None
        self.cooldown_counter = 0

        self._init_is_better()
        super().__init__(optimizer, -1)  # ReduceLROnPlateau不使用epoch

    def _init_is_better(self):
        """初始化比较函数"""
        if self.mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max'
            self.mode_worse = -float('inf')

    def _is_better(self, current, best):
        """判断当前指标是否比最佳指标更好"""
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return current < best * rel_epsilon
        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return current < best - self.threshold
        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return current > best * rel_epsilon
        else:  # mode == 'max' and threshold_mode == 'abs'
            return current > best + self.threshold

    def step(self, metrics):
        """根据验证指标更新学习率

        Args:
            metrics: 当前epoch的验证指标值
        """
        current = float(metrics)

        if self.best is None:
            self.best = current
        else:
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0
            elif self._is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self):
        """降低学习率"""
        old_lr = self.optimizer.lr
        new_lr = max(old_lr * self.factor, self.min_lr)

        if old_lr - new_lr > self.eps:
            self.optimizer.lr = new_lr
            print(f"学习率从 {old_lr:.6f} 降低到 {new_lr:.6f}")

    def get_lr(self):
        """获取当前学习率"""
        return self.optimizer.lr

    def __repr__(self):
        return (f"ReduceLROnPlateau(mode={self.mode}, factor={self.factor}, "
                f"patience={self.patience}, threshold={self.threshold})")


class WarmupLR(LRScheduler):
    """学习率预热调度器

    在训练初期逐渐增加学习率，然后保持不变
    """

    def __init__(self, optimizer, warmup_epochs, last_epoch=-1):
        """
        Args:
            optimizer: 优化器
            warmup_epochs: 预热的epoch数
            last_epoch: 上一个epoch的索引
        """
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """计算预热后的学习率"""
        if self.last_epoch < self.warmup_epochs:
            # 线性预热
            return self.base_lr * (self.last_epoch + 1) / self.warmup_epochs
        else:
            return self.base_lr

    def __repr__(self):
        return f"WarmupLR(warmup_epochs={self.warmup_epochs}, last_epoch={self.last_epoch})"


class CyclicLR(LRScheduler):
    """循环学习率调度器

    学习率在base_lr和max_lr之间循环变化
    """

    def __init__(self, optimizer, base_lr, max_lr, step_size_up=2000,
                 step_size_down=None, mode='triangular', gamma=1.0,
                 scale_fn=None, scale_mode='cycle', last_epoch=-1):
        """
        Args:
            optimizer: 优化器
            base_lr: 最小学习率
            max_lr: 最大学习率
            step_size_up: 上升阶段的步数
            step_size_down: 下降阶段的步数，默认等于step_size_up
            mode: 'triangular', 'triangular2', 'exp_range'
            gamma: 指数模式的衰减因子
            scale_fn: 自定义缩放函数
            scale_mode: 'cycle' 或 'iterations'
        """
        self.base_lr_val = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.mode = mode
        self.gamma = gamma
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode

        self.total_size = self.step_size_up + self.step_size_down
        self.step_ratio = self.step_size_up / self.total_size

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """计算循环学习率"""
        cycle = math.floor(1 + self.last_epoch / self.total_size)
        x = 1. + self.last_epoch / self.total_size - cycle

        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)

        base_height = (self.max_lr - self.base_lr_val) * scale_factor

        if self.scale_fn is None:
            if self.mode == 'triangular':
                lr = self.base_lr_val + base_height
            elif self.mode == 'triangular2':
                lr = self.base_lr_val + base_height / (2. ** (cycle - 1))
            elif self.mode == 'exp_range':
                lr = self.base_lr_val + base_height * (self.gamma ** self.last_epoch)
        else:
            if self.scale_mode == 'cycle':
                lr = self.base_lr_val + base_height * self.scale_fn(cycle)
            else:
                lr = self.base_lr_val + base_height * self.scale_fn(self.last_epoch)

        return lr

    def __repr__(self):
        return (f"CyclicLR(base_lr={self.base_lr_val}, max_lr={self.max_lr}, "
                f"step_size_up={self.step_size_up}, mode={self.mode})")


# ==================== 组合调度器 ====================

class SequentialLR(LRScheduler):
    """顺序学习率调度器

    按顺序应用多个调度器
    """

    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1):
        """
        Args:
            optimizer: 优化器
            schedulers: 调度器列表
            milestones: 切换调度器的epoch列表
            last_epoch: 上一个epoch的索引
        """
        if len(schedulers) != len(milestones) + 1:
            raise ValueError("Number of schedulers should be one more than milestones")

        self.schedulers = schedulers
        self.milestones = milestones
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """根据当前epoch选择对应的调度器"""
        for i, milestone in enumerate(self.milestones):
            if self.last_epoch < milestone:
                return self.schedulers[i].get_lr()
        return self.schedulers[-1].get_lr()

    def step(self, epoch=None):
        """更新所有调度器"""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        # 更新所有调度器
        for scheduler in self.schedulers:
            scheduler.last_epoch = epoch

        new_lr = self.get_lr()
        self.optimizer.lr = new_lr
        return new_lr


# ==================== 实用函数 ====================

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    创建带预热的线性衰减调度器

    Args:
        optimizer: 优化器
        num_warmup_steps: 预热步数
        num_training_steps: 总训练步数
        last_epoch: 上一个epoch
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) /
                   float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class LambdaLR(LRScheduler):
    """Lambda学习率调度器

    使用用户定义的lambda函数来计算学习率
    """

    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        """
        Args:
            optimizer: 优化器
            lr_lambda: 接受epoch参数并返回学习率倍数的函数
            last_epoch: 上一个epoch的索引
        """
        if not callable(lr_lambda):
            raise TypeError("lr_lambda should be callable")

        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """使用lambda函数计算学习率"""
        return self.base_lr * self.lr_lambda(self.last_epoch)

    def __repr__(self):
        return f"LambdaLR(lr_lambda={self.lr_lambda}, last_epoch={self.last_epoch})"