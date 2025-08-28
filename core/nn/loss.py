from core.nn.tensor_nn import Loss
from core.tensor import Tensor
import numpy as np

class BCEWithLogitsLoss(Loss):
    """二元交叉熵损失函数（带logits）

    适用于二元分类问题，自动应用sigmoid + BCE
    计算公式: BCE = -[y*log(sigmoid(x)) + (1-y)*log(1-sigmoid(x))]

    为了数值稳定性，使用log-sum-exp技巧：
    BCE = max(x,0) - x*y + log(1 + exp(-abs(x)))

    Args:
        reduction: 'mean', 'sum', 'none'
        pos_weight: 正样本权重，用于处理类别不平衡（可选）
    """

    def __init__(self, reduction='mean', pos_weight=None):
        super().__init__()
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, predictions, targets):
        """
        predictions: 预测logits张量 (batch_size,) 或 (batch_size, 1) 或任意形状
        targets: 目标标签张量 (batch_size,) 或 (batch_size, 1) 或与predictions相同形状
                值应该在 [0, 1] 之间，通常是0或1
        """
        # 确保targets是Tensor
        if not isinstance(targets, Tensor):
            if isinstance(targets, (list, tuple)):
                targets = Tensor(np.array(targets), device=predictions.device)
            elif isinstance(targets, np.ndarray):
                targets = Tensor(targets, device=predictions.device)

        # 确保形状匹配
        if predictions.shape != targets.shape:
            # 尝试广播或reshape
            if predictions.ndim == 2 and predictions.shape[1] == 1 and targets.ndim == 1:
                targets = targets.reshape(-1, 1)
            elif predictions.ndim == 1 and targets.ndim == 2 and targets.shape[1] == 1:
                targets = targets.reshape(-1)
            elif predictions.ndim == targets.ndim + 1 and predictions.shape[-1] == 1:
                predictions = predictions.squeeze(-1)
            else:
                raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")

        # 数值稳定的BCE计算
        # BCE = max(x,0) - x*y + log(1 + exp(-abs(x)))
        abs_logits = predictions.abs()
        max_logits = predictions.maximum(Tensor(np.zeros_like(predictions.data), device=predictions.device))

        # log(1 + exp(-abs(x)))
        neg_abs_logits = -abs_logits
        log_exp_term = (neg_abs_logits.exp() + 1).log()

        # BCE loss per sample
        loss_per_sample = max_logits - predictions * targets + log_exp_term

        # 应用正样本权重（如果提供）
        if self.pos_weight is not None:
            if not isinstance(self.pos_weight, Tensor):
                pos_weight = Tensor(np.array([self.pos_weight]), device=predictions.device)
            else:
                pos_weight = self.pos_weight

            # 权重: pos_weight * y + (1 - y)
            weight = targets * pos_weight + (1 - targets)
            loss_per_sample = loss_per_sample * weight

        # 根据reduction参数返回结果
        if self.reduction == 'mean':
            return loss_per_sample.mean()
        elif self.reduction == 'sum':
            return loss_per_sample.sum()
        else:  # 'none'
            return loss_per_sample

    def __repr__(self):
        if self.pos_weight is not None:
            return f"BCEWithLogitsLoss(reduction='{self.reduction}', pos_weight={self.pos_weight})"
        else:
            return f"BCEWithLogitsLoss(reduction='{self.reduction}')"


# 使用示例和测试
if __name__ == "__main__":
    import numpy as np

    # 创建损失函数
    bce_loss = BCEWithLogitsLoss(reduction='mean')
    bce_loss_weighted = BCEWithLogitsLoss(reduction='mean', pos_weight=2.0)

    print("BCEWithLogitsLoss implementation completed!")
    print(f"Basic BCE: {bce_loss}")
    print(f"Weighted BCE: {bce_loss_weighted}")

    logits = Tensor(np.array([0.5, -1.2, 2.1, -0.8]))
    targets = Tensor(np.array([1, 0, 1, 0]))
    loss = bce_loss(logits, targets)
    print(f"Loss: {loss}")