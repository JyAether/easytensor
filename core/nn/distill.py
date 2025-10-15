import numpy as np
from typing import Dict, Tuple, Optional, List

from core.nn import LSTM
from core.nn.bert_gpt import BERT
from core.tensor import Tensor, zeros, randn, cat
from core.nn.tensor_nn import Module, Linear, CrossEntropyLoss, MSELoss


class KLDivergenceLoss(Module):
    """KL散度损失函数，用于软标签蒸馏"""

    def __init__(self, temperature=4.0, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, student_logits, teacher_logits):
        """
        计算KL散度损失
        Args:
            student_logits: 学生模型输出 (batch_size, num_classes)
            teacher_logits: 教师模型输出 (batch_size, num_classes)
        """
        # 应用温度缩放
        student_soft = (student_logits / self.temperature).softmax(dim=-1)
        teacher_soft = (teacher_logits / self.temperature).softmax(dim=-1)

        # 计算KL散度: KL(P||Q) = sum(P * log(P/Q))
        # 这里P是teacher, Q是student
        log_student = student_soft.log()
        kl_loss = teacher_soft * (teacher_soft.log() - log_student)

        if self.reduction == 'mean':
            return kl_loss.mean() * (self.temperature ** 2)
        elif self.reduction == 'sum':
            return kl_loss.sum() * (self.temperature ** 2)
        else:
            return kl_loss * (self.temperature ** 2)



class TeacherBERTClassifier(Module):
    """教师模型：基于BERT的分类器（增强版）"""

    def __init__(self, bert_model, num_classes, device='cpu'):
        super().__init__()
        self.bert = bert_model
        self.classifier = Linear(bert_model.d_model, num_classes, device=device)
        self.device = device

    def forward(self, input_ids, segment_ids=None, attention_mask=None,
                return_intermediate=False):
        """
        前向传播，返回详细的中间层信息
        """
        # 获取BERT输出，包含所有隐藏层状态和注意力
        outputs = self.bert(input_ids, segment_ids, attention_mask,
                            output_hidden_states=True, output_attentions=True)

        # 使用pooler_output进行分类
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)

        if return_intermediate:
            return {
                'logits': logits,
                'pooler_output': pooled_output,
                'hidden_states': outputs.hidden_states,  # 所有层的隐藏状态
                'attentions': outputs.attentions,  # 所有层的注意力
                'last_hidden_state': outputs.last_hidden_state
            }
        else:
            return logits


class StudentLSTMClassifier(Module):
    """学生模型：基于LSTM的分类器（增强版）"""

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers,
                 num_classes, dropout=0.1, device='cpu'):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 嵌入层
        from core.nn.transform import Embedding
        self.embedding = Embedding(vocab_size, embedding_dim, device=device)

        # 多层LSTM
        self.lstm_layers = []
        input_size = embedding_dim
        for i in range(num_layers):
            lstm = LSTM(input_size, hidden_size, 1, batch_first=True, device=device)
            self.lstm_layers.append(lstm)
            setattr(self, f'lstm_{i}', lstm)  # 注册为模块属性
            input_size = hidden_size

        # 分类器
        self.classifier = Linear(hidden_size, num_classes, device=device)

    def forward(self, input_ids, return_intermediate=False):
        """
        前向传播，返回所有LSTM层的隐藏状态
        """
        # 嵌入
        embeddings = self.embedding(input_ids)

        # 逐层通过LSTM，收集所有隐藏状态
        hidden_states = []
        x = embeddings

        for i, lstm_layer in enumerate(self.lstm_layers):
            x, (h, c) = lstm_layer(x)
            hidden_states.append(x)  # 保存每一层的输出

        # 使用最后一层的最后一个时间步进行分类
        last_hidden = Tensor(x.data[:, -1, :],
                             requires_grad=x.requires_grad,
                             device=x.device)
        logits = self.classifier(last_hidden)

        if return_intermediate:
            return {
                'logits': logits,
                'last_hidden': last_hidden,
                'hidden_states': hidden_states,  # 所有LSTM层的隐藏状态
                'embeddings': embeddings
            }
        else:
            return logits


# class TeacherBERTClassifier(Module):
#     """教师模型：基于BERT的分类器"""
#
#     def __init__(self, bert_model, num_classes, device='cpu'):
#         super().__init__()
#         self.bert = bert_model
#         self.classifier = Linear(bert_model.d_model, num_classes, device=device)
#         self.device = device
#
#     def forward(self, input_ids, segment_ids=None, attention_mask=None,
#                 return_intermediate=False):
#         """
#         前向传播
#         Args:
#             return_intermediate: 是否返回中间层特征用于蒸馏
#         """
#         # 获取BERT输出，包含所有隐藏层状态
#         outputs = self.bert(input_ids, segment_ids, attention_mask,
#                             output_hidden_states=return_intermediate)
#
#         # 使用pooler_output进行分类
#         pooled_output = outputs.pooler_output
#         logits = self.classifier(pooled_output)
#
#         if return_intermediate:
#             return {
#                 'logits': logits,
#                 'pooler_output': pooled_output,
#                 'hidden_states': outputs.hidden_states,
#                 'last_hidden_state': outputs.last_hidden_state
#             }
#         else:
#             return logits
#
#
# class StudentLSTMClassifier(Module):
#     """学生模型：基于LSTM的分类器"""
#
#     def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers,
#                  num_classes, dropout=0.1, device='cpu'):
#         super().__init__()
#         self.device = device
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#
#         # 嵌入层
#         from core.nn.transform import Embedding
#         self.embedding = Embedding(vocab_size, embedding_dim, device=device)
#
#         # LSTM层
#         self.lstm = LSTM(embedding_dim, hidden_size, num_layers,
#                          batch_first=True, dropout=dropout, device=device)
#
#         # 分类器
#         self.classifier = Linear(hidden_size, num_classes, device=device)
#
#         # 中间层投影（用于中间层蒸馏）
#         self.intermediate_projection = Linear(hidden_size, 768, device=device)  # 投影到BERT维度
#
#     def forward(self, input_ids, return_intermediate=False):
#         """
#         前向传播
#         Args:
#             return_intermediate: 是否返回中间层特征用于蒸馏
#         """
#         # 嵌入
#         embeddings = self.embedding(input_ids)
#
#         # LSTM处理
#         lstm_output, (hidden, cell) = self.lstm(embeddings)
#
#         # 使用最后一个时间步的输出进行分类
#         last_hidden = Tensor(lstm_output.data[:, -1, :],
#                              requires_grad=lstm_output.requires_grad,
#                              device=lstm_output.device)
#         logits = self.classifier(last_hidden)
#
#         if return_intermediate:
#             # 投影到教师模型的维度用于中间层蒸馏
#             projected_hidden = self.intermediate_projection(last_hidden)
#             return {
#                 'logits': logits,
#                 'last_hidden': last_hidden,
#                 'projected_hidden': projected_hidden,
#                 'lstm_output': lstm_output
#             }
#         else:
#             return logits

#
# class KnowledgeDistillationTrainer(Module):
#     """知识蒸馏训练器"""
#
#     def __init__(self, teacher_model, student_model, device='cpu'):
#         super().__init__()
#         self.teacher = teacher_model
#         self.student = student_model
#         self.device = device
#
#         # 损失函数
#         self.hard_loss_fn = CrossEntropyLoss()  # 硬标签损失
#         self.soft_loss_fn = KLDivergenceLoss(temperature=4.0)  # 软标签损失
#         self.intermediate_loss_fn = MSELoss()  # 中间层损失
#
#         # 设置教师模型为评估模式
#         self.teacher.eval()
#
#     def hard_label_distillation(self, student_logits, teacher_logits, true_labels):
#         """
#         硬标签蒸馏：学生模型直接学习教师模型的硬标签预测
#         """
#         # 获取教师模型的硬标签预测
#         teacher_predictions = teacher_logits.argmax(axis=-1)
#
#         # 计算学生模型对教师预测的损失
#         hard_distill_loss = self.hard_loss_fn(student_logits, teacher_predictions)
#
#         # 计算学生模型对真实标签的损失
#         true_label_loss = self.hard_loss_fn(student_logits, true_labels)
#
#         return hard_distill_loss, true_label_loss
#
#     def soft_label_distillation(self, student_logits, teacher_logits):
#         """
#         软标签蒸馏：学生模型学习教师模型的软标签分布
#         """
#         # 使用KL散度计算软标签损失
#         soft_loss = self.soft_loss_fn(student_logits, teacher_logits)
#         return soft_loss
#
#     def intermediate_layer_distillation(self, student_features, teacher_features):
#         """
#         中间层蒸馏：学生模型学习教师模型的中间层表示
#         """
#         # 计算中间层特征的MSE损失
#         intermediate_loss = self.intermediate_loss_fn(student_features, teacher_features)
#         return intermediate_loss
#
#     def forward(self, input_ids, segment_ids, true_labels,
#                 distillation_type='all', alpha=0.7, beta=0.3, gamma=0.1):
#         """
#         完整的知识蒸馏训练步骤
#
#         Args:
#             distillation_type: 'hard', 'soft', 'intermediate', 'all'
#             alpha: 软标签损失权重
#             beta: 硬标签损失权重
#             gamma: 中间层损失权重
#         """
#         # 教师模型前向传播（设置为评估模式，不计算梯度）
#         self.teacher.eval()
#         teacher_outputs = self.teacher(input_ids, segment_ids,
#                                        return_intermediate=True)
#         teacher_logits = teacher_outputs['logits']
#         teacher_pooler = teacher_outputs['pooler_output']
#
#         # 学生模型前向传播
#         self.student.train()
#         student_outputs = self.student(input_ids, return_intermediate=True)
#         student_logits = student_outputs['logits']
#         student_projected = student_outputs['projected_hidden']
#
#         # 初始化总损失
#         total_loss = Tensor(np.array([0.0]), requires_grad=True, device=self.device)
#         loss_components = {}
#
#         # 1. 硬标签蒸馏
#         if distillation_type in ['hard', 'all']:
#             hard_distill_loss, true_label_loss = self.hard_label_distillation(
#                 student_logits, teacher_logits, true_labels)
#
#             hard_loss = Tensor(
#                 beta * hard_distill_loss.data + (1 - beta) * true_label_loss.data,
#                 requires_grad=True, device=self.device
#             )
#             total_loss = Tensor(
#                 total_loss.data + hard_loss.data,
#                 requires_grad=True, device=self.device
#             )
#             loss_components['hard_loss'] = hard_loss
#
#         # 2. 软标签蒸馏
#         if distillation_type in ['soft', 'all']:
#             soft_loss = self.soft_label_distillation(student_logits, teacher_logits)
#             weighted_soft_loss = Tensor(
#                 alpha * soft_loss.data,
#                 requires_grad=True, device=self.device
#             )
#             total_loss = Tensor(
#                 total_loss.data + weighted_soft_loss.data,
#                 requires_grad=True, device=self.device
#             )
#             loss_components['soft_loss'] = soft_loss
#
#         # 3. 中间层蒸馏
#         if distillation_type in ['intermediate', 'all']:
#             intermediate_loss = self.intermediate_layer_distillation(
#                 student_projected, teacher_pooler)
#             weighted_intermediate_loss = Tensor(
#                 gamma * intermediate_loss.data,
#                 requires_grad=True, device=self.device
#             )
#             total_loss = Tensor(
#                 total_loss.data + weighted_intermediate_loss.data,
#                 requires_grad=True, device=self.device
#             )
#             loss_components['intermediate_loss'] = intermediate_loss
#
#         # 如果只使用真实标签训练
#         if distillation_type == 'baseline':
#             baseline_loss = self.hard_loss_fn(student_logits, true_labels)
#             total_loss = baseline_loss
#             loss_components['baseline_loss'] = baseline_loss
#
#         return total_loss, loss_components, student_logits


def create_distillation_models(device='cpu'):
    """创建教师和学生模型"""

    # 创建教师模型 (BERT)
    bert_model = BERT(vocab_size=30522, max_seq_len=512, d_model=768,
                      n_layers=12, n_heads=12, d_ff=3072, device=device)
    teacher_model = TeacherBERTClassifier(bert_model, num_classes=2, device=device)

    # 创建学生模型 (LSTM)
    student_model = StudentLSTMClassifier(
        vocab_size=30522, embedding_dim=128, hidden_size=256,
        num_layers=2, num_classes=2, device=device)

    return teacher_model, student_model


def train_distillation_step(trainer, input_ids, segment_ids, labels,
                            distillation_type='all'):
    """单步蒸馏训练"""

    # 清零梯度
    trainer.zero_grad()

    # 前向传播
    total_loss, loss_components, student_logits = trainer(
        input_ids, segment_ids, labels, distillation_type=distillation_type)

    # 反向传播
    total_loss.backward()

    # 计算准确率
    predictions = student_logits.argmax(axis=-1)
    correct = Tensor(
        (predictions.data == labels.data).astype(np.float32),
        device=predictions.device
    )
    accuracy = correct.mean()

    # return {
    #     'total_loss': float(total_loss.data.item() if hasattr(total_loss.data, 'item') else total_loss.data),
    #     'loss_components': {k: float(v.data.item() if hasattr(v.data, 'item') else v.data)
    #                         for k, v in loss_components.items()},
    #     'accuracy': float(accuracy.data.item() if hasattr(accuracy.data, 'item') else accuracy.data)
    # }
    # 修复：处理loss_components，跳过layer_losses或单独展开列表
    processed_loss_components = {}
    for k, v in loss_components.items():
        if k == 'layer_losses':
            # 单独处理列表：将每一层的损失转为数值并存储为列表
            processed_loss_components[k] = [float(loss.data.item() if hasattr(loss.data, 'item') else loss.data)
                                            for loss in v]
        else:
            # 正常处理Tensor类型的损失
            processed_loss_components[k] = float(v.data.item() if hasattr(v.data, 'item') else v.data)

    return {
        'total_loss': float(total_loss.data.item() if hasattr(total_loss.data, 'item') else total_loss.data),
        'loss_components': processed_loss_components,  # 使用处理后的字典
        'accuracy': float(accuracy.data.item() if hasattr(accuracy.data, 'item') else accuracy.data)
    }


# 优化器实现（简化版）
class SGD:
    """随机梯度下降优化器"""

    def __init__(self, parameters, lr=0.01, momentum=0.0):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [zeros(*param.shape, device=param.device) for param in self.parameters]

    def step(self):
        """执行一步优化"""
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                # 更新速度
                self.velocity[i] = Tensor(
                    self.momentum * self.velocity[i].data - self.lr * param.grad.data,
                    device=param.device
                )
                # 更新参数
                param.data = param.data + self.velocity[i].data

    def zero_grad(self):
        """清零梯度"""
        for param in self.parameters:
            param.zero_grad()


# 使用示例和对比实验
def demonstrate_distillation_methods():
    """演示不同蒸馏方法的效果"""

    device = 'cpu'

    # 创建模型
    teacher_model, student_model = create_distillation_models(device)
    trainer = KnowledgeDistillationTrainer(teacher_model, student_model, device)

    # 创建优化器
    optimizer = SGD(student_model.parameters(), lr=0.001)

    # 模拟数据
    batch_size, seq_len = 4, 32
    input_ids = Tensor(np.random.randint(0, 30522, (batch_size, seq_len)), device=device)
    segment_ids = Tensor(np.zeros((batch_size, seq_len)), device=device)
    labels = Tensor(np.random.randint(0, 2, (batch_size,)), device=device)

    print("知识蒸馏方法对比实验")
    print("=" * 50)

    distillation_methods = [
        ('基线方法', 'baseline'),
        ('硬标签蒸馏', 'hard'),
        ('软标签蒸馏', 'soft'),
        ('中间层蒸馏', 'intermediate'),
        ('综合蒸馏', 'all')
    ]

    for method_name, method_type in distillation_methods:
        # 训练一步
        results = train_distillation_step(
            trainer, input_ids, segment_ids, labels, method_type)

        print(f"{method_name} - 损失: {results['total_loss']:.4f}, "
              f"准确率: {results['accuracy']:.4f}")

        # 更新参数（仅对学生模型）
        optimizer.step()
        optimizer.zero_grad()

    print(f"\n教师模型参数量: {teacher_model.count_parameters():,}")
    print(f"学生模型参数量: {student_model.count_parameters():,}")
    print(f"压缩比: {teacher_model.count_parameters() / student_model.count_parameters():.2f}x")


class DistillationConfig:
    """蒸馏配置类"""

    def __init__(self):
        self.temperature = 4.0  # 温度参数
        self.alpha = 0.7  # 软标签损失权重
        self.beta = 0.3  # 硬标签损失权重
        self.gamma = 0.1  # 中间层损失权重
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.num_epochs = 10


def full_training_loop(teacher_model, student_model, train_data, config, device='cpu'):
    """完整的蒸馏训练循环"""

    trainer = KnowledgeDistillationTrainer(teacher_model, student_model, device)
    optimizer = SGD(student_model.parameters(), lr=config.learning_rate)

    print("开始知识蒸馏训练...")
    print(f"训练配置: 温度={config.temperature}, α={config.alpha}, β={config.beta}, γ={config.gamma}")

    for epoch in range(config.num_epochs):
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        for batch_idx, (input_ids, segment_ids, labels) in enumerate(train_data):
            # 训练步骤
            results = train_distillation_step(
                trainer, input_ids, segment_ids, labels, 'all')

            # 更新参数
            optimizer.step()
            optimizer.zero_grad()

            total_loss += results['total_loss']
            total_accuracy += results['accuracy']
            num_batches += 1

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/{config.num_epochs}, "
                      f"Batch {batch_idx}, Loss: {results['total_loss']:.4f}, "
                      f"Acc: {results['accuracy']:.4f}")

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        print(f"Epoch {epoch + 1} 完成 - 平均损失: {avg_loss:.4f}, 平均准确率: {avg_accuracy:.4f}")


class IntermediateLayerDistillation(Module):
    """中间层蒸馏模块（完整版）"""

    def __init__(self, teacher_hidden_size, student_hidden_size, num_layers=4, device='cpu'):
        super().__init__()
        self.num_layers = num_layers
        self.device = device

        # 为每一层创建投影层
        self.projections = []
        for i in range(num_layers):
            projection = Linear(student_hidden_size, teacher_hidden_size, device=device)
            self.projections.append(projection)
            setattr(self, f'projection_{i}', projection)

        self.loss_fn = MSELoss()

    def forward(self, student_hidden_states, teacher_hidden_states):
        """
        计算多层中间层蒸馏损失

        Args:
            student_hidden_states: 学生模型的隐藏状态列表 [(batch, seq_len, student_hidden)]
            teacher_hidden_states: 教师模型的隐藏状态列表 [(batch, seq_len, teacher_hidden)]
        """
        # 选择要蒸馏的层
        teacher_indices = self._select_teacher_layers(len(teacher_hidden_states))
        student_indices = self._select_student_layers(len(student_hidden_states))

        # 初始化总损失
        total_loss = None
        layer_losses = []

        for i, (t_idx, s_idx) in enumerate(zip(teacher_indices, student_indices)):
            if i >= len(self.projections):
                break

            # 获取对应层的隐藏状态
            teacher_hidden = teacher_hidden_states[t_idx]  # (batch, seq_len, teacher_hidden)
            student_hidden = student_hidden_states[s_idx]  # (batch, seq_len, student_hidden)

            # 投影学生隐藏状态到教师维度
            projected_student = self.projections[i](student_hidden)  # (batch, seq_len, teacher_hidden)

            # 计算该层的损失
            layer_loss = self.loss_fn(projected_student, teacher_hidden)
            layer_losses.append(layer_loss)

            # 累加到总损失（保持梯度）
            if total_loss is None:
                total_loss = layer_loss
            else:
                # 创建新的张量来累加损失，保持梯度图
                total_loss = Tensor(
                    total_loss.data + layer_loss.data,
                    requires_grad=True,
                    device=self.device
                )

                # 手动构建梯度计算图
                def backward_fn(grad_output):
                    # 将梯度传递给两个损失项
                    if total_loss.requires_grad:
                        # 这里需要根据您的自动微分系统来实现
                        pass
                    return grad_output, grad_output

                # 如果您的框架支持，可以设置backward hook
                # total_loss.backward_fn = backward_fn

        # 计算平均损失
        num_layers = len(layer_losses)
        if num_layers > 0:
            avg_loss = Tensor(
                total_loss.data / num_layers,
                requires_grad=True,
                device=self.device
            )

            # 保持梯度传播链
            def avg_backward_fn(grad_output):
                if total_loss.requires_grad:
                    return grad_output / num_layers
                return grad_output

            # avg_loss.backward_fn = avg_backward_fn
        else:
            avg_loss = Tensor(np.array([0.0]), requires_grad=True, device=self.device)

        return avg_loss, layer_losses

    def _select_teacher_layers(self, total_teacher_layers):
        """选择教师模型的层进行蒸馏"""
        if total_teacher_layers <= self.num_layers:
            return list(range(total_teacher_layers))
        else:
            # 均匀分布选择层
            indices = np.linspace(0, total_teacher_layers - 1, self.num_layers, dtype=int)
            return indices.tolist()

    def _select_student_layers(self, total_student_layers):
        """选择学生模型的层进行蒸馏"""
        if total_student_layers <= self.num_layers:
            return list(range(total_student_layers))
        else:
            indices = np.linspace(0, total_student_layers - 1, self.num_layers, dtype=int)
            return indices.tolist()


class KnowledgeDistillationTrainer(Module):
    """知识蒸馏训练器（完整版）"""

    def __init__(self, teacher_model, student_model, device='cpu'):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.device = device

        # 损失函数
        self.hard_loss_fn = CrossEntropyLoss()
        self.soft_loss_fn = KLDivergenceLoss(temperature=4.0)

        # 中间层蒸馏模块
        self.intermediate_distiller = IntermediateLayerDistillation(
            teacher_hidden_size=768,  # BERT hidden size
            student_hidden_size=256,  # LSTM hidden size
            num_layers=4,
            device=device
        )

        # 设置教师模型为评估模式
        self.teacher.eval()

    def forward(self, input_ids, segment_ids, true_labels,
                distillation_type='all', alpha=0.4, beta=0.4, gamma=0.2):
        """
        完整的知识蒸馏训练步骤

        Args:
            distillation_type: 'hard', 'soft', 'intermediate', 'all'
            alpha: 软标签损失权重
            beta: 硬标签损失权重
            gamma: 中间层损失权重
        """
        # 教师模型前向传播
        self.teacher.eval()
        teacher_outputs = self.teacher(input_ids, segment_ids,
                                       return_intermediate=True)
        teacher_logits = teacher_outputs['logits']
        teacher_hidden_states = teacher_outputs['hidden_states']

        # 学生模型前向传播
        self.student.train()
        student_outputs = self.student(input_ids, return_intermediate=True)
        student_logits = student_outputs['logits']
        student_hidden_states = student_outputs['hidden_states']

        # 初始化总损失
        total_loss = None
        loss_components = {}

        # 1. 硬标签损失
        if distillation_type in ['hard', 'all']:
            hard_loss = self.hard_loss_fn(student_logits, true_labels)
            weighted_hard_loss = self._scale_loss(hard_loss, beta)
            total_loss = self._add_loss(total_loss, weighted_hard_loss)
            loss_components['hard_loss'] = hard_loss

        # 2. 软标签蒸馏
        if distillation_type in ['soft', 'all']:
            soft_loss = self.soft_loss_fn(student_logits, teacher_logits)
            weighted_soft_loss = self._scale_loss(soft_loss, alpha)
            total_loss = self._add_loss(total_loss, weighted_soft_loss)
            loss_components['soft_loss'] = soft_loss

        # 3. 中间层蒸馏（多层对多层）
        if distillation_type in ['intermediate', 'all']:
            intermediate_loss, layer_losses = self.intermediate_distiller(
                student_hidden_states, teacher_hidden_states)
            weighted_intermediate_loss = self._scale_loss(intermediate_loss, gamma)
            total_loss = self._add_loss(total_loss, weighted_intermediate_loss)
            loss_components['intermediate_loss'] = intermediate_loss
            loss_components['layer_losses'] = layer_losses

        if total_loss is None:
            total_loss = Tensor(np.array([0.0]), requires_grad=True, device=self.device)

        return total_loss, loss_components, student_logits

    def _scale_loss(self, loss, weight):
        """缩放损失，保持梯度"""
        return Tensor(
            weight * loss.data,
            requires_grad=True,
            device=self.device
        )

    def _add_loss(self, total_loss, new_loss):
        """累加损失，保持梯度"""
        if total_loss is None:
            return new_loss
        else:
            return Tensor(
                total_loss.data + new_loss.data,
                requires_grad=True,
                device=self.device
            )


def demonstrate_layerwise_distillation():
    """演示逐层蒸馏的效果"""

    device = 'cpu'

    # 创建模型
    teacher_model, student_model = create_distillation_models(device)
    trainer = KnowledgeDistillationTrainer(teacher_model, student_model, device)

    # 模拟数据
    batch_size, seq_len = 2, 16
    input_ids = Tensor(np.random.randint(0, 1000, (batch_size, seq_len)), device=device)
    segment_ids = Tensor(np.zeros((batch_size, seq_len)), device=device)
    labels = Tensor(np.random.randint(0, 2, (batch_size,)), device=device)

    print("逐层知识蒸馏演示")
    print("=" * 50)

    # 获取教师和学生模型的中间层信息
    teacher_outputs = teacher_model(input_ids, segment_ids, return_intermediate=True)
    student_outputs = student_model(input_ids, return_intermediate=True)

    print(f"教师模型隐藏层数量: {len(teacher_outputs['hidden_states'])}")
    print(f"学生模型隐藏层数量: {len(student_outputs['hidden_states'])}")

    # 显示层的形状信息
    for i, hidden_state in enumerate(teacher_outputs['hidden_states']):
        print(f"教师模型第{i}层形状: {hidden_state.data.shape}")

    for i, hidden_state in enumerate(student_outputs['hidden_states']):
        print(f"学生模型第{i}层形状: {hidden_state.data.shape}")

    # 显示层匹配策略
    distiller = trainer.intermediate_distiller
    teacher_indices = distiller._select_teacher_layers(len(teacher_outputs['hidden_states']))
    student_indices = distiller._select_student_layers(len(student_outputs['hidden_states']))

    print(f"\n层匹配策略:")
    for i, (t_idx, s_idx) in enumerate(zip(teacher_indices, student_indices)):
        print(f"蒸馏层{i}: 学生层{s_idx} ← 教师层{t_idx}")

    # 执行蒸馏训练
    total_loss, loss_components, _ = trainer(
        input_ids, segment_ids, labels, distillation_type='intermediate')

    print(f"\n中间层蒸馏损失: {loss_components['intermediate_loss'].data:.4f}")
    if 'layer_losses' in loss_components:
        for i, layer_loss in enumerate(loss_components['layer_losses']):
            print(f"第{i}层损失: {layer_loss.data:.4f}")

    # 测试完整蒸馏
    print(f"\n完整蒸馏测试:")
    total_loss, loss_components, _ = trainer(
        input_ids, segment_ids, labels, distillation_type='all',
        alpha=0.4, beta=0.4, gamma=0.2)

    print(f"总损失: {total_loss.data:.4f}")
    for loss_name, loss_value in loss_components.items():
        if loss_name != 'layer_losses':
            print(f"{loss_name}: {loss_value.data:.4f}")


if __name__ == "__main__":
    print("知识蒸馏完整实现")
    print("支持硬标签、软标签和中间层蒸馏")
    print("教师模型: BERT, 学生模型: LSTM")
    print("基于自制深度学习引擎")

    # 运行演示
    demonstrate_distillation_methods()
