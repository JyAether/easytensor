from core.v1.engine import Node
import numpy as np


class Loss:
    """损失函数基类"""

    def __call__(self, predictions, targets):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + "()"


class MSELoss(Loss):
    """均方误差损失函数

    计算公式: MSE = (1/n) * Σ(pred_i - target_i)^2
    其中 n 是样本数量

    Args:
        reduction: 'mean', 'sum', 'none'
            - 'mean': 返回平均损失 (默认)
            - 'sum': 返回损失总和
            - 'none': 返回每个样本的损失
    """

    def __init__(self, reduction='mean'):
        assert reduction in ['mean', 'sum', 'none'], \
            f"reduction必须是 'mean', 'sum' 或 'none', 得到 '{reduction}'"
        self.reduction = reduction

    def __call__(self, predictions, targets):
        """
        Args:
            predictions: 预测值，可以是单个Node或Node列表
            targets: 目标值，可以是单个Node或Node列表

        Returns:
            Node: 损失值
        """
        # 统一处理单个值和列表的情况
        if not isinstance(predictions, list):
            predictions = [predictions]
        if not isinstance(targets, list):
            targets = [targets]

        assert len(predictions) == len(targets), \
            f"预测值数量({len(predictions)})与目标值数量({len(targets)})不匹配"

        # 计算每个样本的平方误差
        squared_errors = []
        for pred, target in zip(predictions, targets):
            diff = pred - target
            squared_error = diff * diff
            squared_errors.append(squared_error)

        # 根据reduction参数处理结果
        if self.reduction == 'none':
            return squared_errors
        elif self.reduction == 'sum':
            total_loss = squared_errors[0]
            for se in squared_errors[1:]:
                total_loss = total_loss + se
            return total_loss
        else:  # reduction == 'mean'
            total_loss = squared_errors[0]
            for se in squared_errors[1:]:
                total_loss = total_loss + se
            # 除以样本数量得到平均损失
            return total_loss * Node(1.0 / len(squared_errors))


class CrossEntropyLoss(Loss):
    """多分类交叉熵损失函数

    适用于多分类问题，自动应用softmax + logist
    计算公式: CE = -log(softmax(logits)[target_class])

    Args:
        reduction: 'mean', 'sum', 'none'
        ignore_index: 忽略的类别索引（可选）
    """

    def __init__(self, reduction='mean', ignore_index=None):
        assert reduction in ['mean', 'sum', 'none'], \
            f"reduction必须是 'mean', 'sum' 或 'none', 得到 '{reduction}'"
        self.reduction = reduction
        self.ignore_index = ignore_index

    def _softmax(self, logits):
        """计算softmax，包含数值稳定性处理"""
        # 数值稳定性：减去最大值
        max_logit = max(logit.data for logit in logits)

        # 计算 exp(x - max)
        exp_logits = []
        for logit in logits:
            shifted_logit = logit + Node(-max_logit)

            # exp_logits.append(shifted_logit.exp())
            exp_logits.append(np.exp(shifted_logit.data))

        # 计算分母：所有exp值的和
        sum_exp = exp_logits[0]
        for exp_logit in exp_logits[1:]:
            sum_exp = sum_exp + exp_logit

        # 计算softmax概率
        probabilities = []
        for exp_logit in exp_logits:
            prob = exp_logit * (Node(1) / sum_exp)
            probabilities.append(prob)

        return probabilities

    def _log_softmax(self, logits):
        """计算log_softmax，数值稳定版本"""
        # 数值稳定性：减去最大值
        max_logit = max(logit.data for logit in logits)

        # 计算 log_sum_exp
        shifted_logits = [logit + Node(-max_logit) for logit in logits]
        # exp_sum = shifted_logits[0].exp()
        exp_sum = np.exp(shifted_logits[0].data)
        for shifted_logit in shifted_logits[1:]:
            # exp_sum = exp_sum + shifted_logit.exp()
            exp_sum = exp_sum + np.exp(shifted_logit.data)

        # log_sum_exp = exp_sum.log() + Node(max_logit)
        log_sum_exp = np.log(exp_sum) + Node(max_logit)

        # 计算 log_softmax = logit - log_sum_exp
        log_probs = []
        for logit in logits:
            log_prob = logit - log_sum_exp
            log_probs.append(log_prob)

        return log_probs

    def __call__(self, predictions, targets):
        """
        Args:
            predictions: 预测logits，形状为 [batch_size, num_classes] 的嵌套列表
                       或单个样本的 [num_classes] 列表
            targets: 目标类别索引，整数或整数列表

        Returns:
            Node: 损失值
        """
        # 处理单个样本的情况
        if isinstance(targets, int):
            targets = [targets]
            predictions = [predictions]

        losses = []

        for pred_logits, target_idx in zip(predictions, targets):
            if self.ignore_index is not None and target_idx == self.ignore_index:
                continue

            # 计算log_softmax
            log_probs = self._log_softmax(pred_logits)

            # 选择目标类别的log概率
            target_log_prob = log_probs[target_idx]

            # 负对数似然损失: -log(p_target)
            nll_loss = Node(0) - target_log_prob
            losses.append(nll_loss)

        if len(losses) == 0:
            return Node(0.0)

        # 根据reduction参数处理结果
        if self.reduction == 'none':
            return losses
        elif self.reduction == 'sum':
            total_loss = losses[0]
            for loss in losses[1:]:
                total_loss = total_loss + loss
            return total_loss
        else:  # reduction == 'mean'
            total_loss = losses[0]
            for loss in losses[1:]:
                total_loss = total_loss + loss
            return total_loss * Node(1.0 / len(losses))


class BCELoss(Loss):
    """二分类交叉熵损失函数（Binary Cross Entropy）

    适用于二分类问题，假设输入已经过sigmoid激活
    计算公式: BCE = -[y*log(p) + (1-y)*log(1-p)]
    """

    def __init__(self, reduction='mean'):
        assert reduction in ['mean', 'sum', 'none'], \
            f"reduction必须是 'mean', 'sum' 或 'none', 得到 '{reduction}'"
        self.reduction = reduction

    def __call__(self, predictions, targets):
        if not isinstance(predictions, list):
            predictions = [predictions]
        if not isinstance(targets, list):
            targets = [targets]

        losses = []
        for pred, target in zip(predictions, targets):
            # 数值稳定性处理
            eps = 1e-7
            pred_clamped = Node(max(eps, min(1 - eps, pred.data)))

            # BCE = -[y*log(p) + (1-y)*log(1-p)]
            # log_pred = pred_clamped.log()
            log_pred = np.log(pred_clamped.data)

            # log_one_minus_pred = (Node(1) - pred_clamped).log()
            log_one_minus_pred = np.log(Node(1).data - pred_clamped.data)

            bce = Node(0) - (target * log_pred + (Node(1) - target) * log_one_minus_pred)
            losses.append(bce)

        if self.reduction == 'none':
            return losses
        elif self.reduction == 'sum':
            total = losses[0]
            for loss in losses[1:]:
                total = total + loss
            return total
        else:  # mean
            total = losses[0]
            for loss in losses[1:]:
                total = total + loss
            return total * Node(1.0 / len(losses))
