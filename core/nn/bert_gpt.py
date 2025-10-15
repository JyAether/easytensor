from typing import List, NamedTuple

import numpy as np
import math
from core.nn import Linear
from core.nn.transform import Embedding, LayerNorm, PositionalEncoding, TransformerDecoderLayer, TransformerEncoderLayer
from core.tensor import Tensor
from core.nn import Module

class BERTOutput:
    """BERT输出包装类（增强版）"""
    def __init__(self, last_hidden_state, pooler_output, hidden_states=None, attentions=None):
        self.last_hidden_state = last_hidden_state
        self.pooler_output = pooler_output
        self.hidden_states = hidden_states
        self.attentions = attentions

class BERTPooler(Module):
    """
    BERT的池化层
    将CLS token的hidden state转换为用于分类任务的向量表示
    """

    def __init__(self, d_model, device='cpu'):
        super().__init__()
        # 线性变换层 + tanh激活
        self.dense = Linear(d_model, d_model, device=device)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch_size, seq_len, d_model)
        Returns:
            pooled_output: (batch_size, d_model) - 池化后的CLS表示
        """
        # 取CLS token的hidden state (第一个位置)
        first_token_tensor = Tensor(
            hidden_states.data[:, 0],
            requires_grad=hidden_states.requires_grad,
            device=hidden_states.device
        )

        # 通过线性层
        pooled_output = self.dense(first_token_tensor)

        # 应用tanh激活函数
        pooled_output = pooled_output.tanh()

        return pooled_output


class BERT(Module):
    """BERT模型实现（包含pooler_output）"""

    def __init__(self, vocab_size=30522, max_seq_len=512, d_model=768,
                 n_layers=12, n_heads=12, d_ff=3072, device='cpu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_layers = n_layers  # 添加这个属性
        self.device = device

        # 嵌入层
        self.token_embedding = Embedding(vocab_size, d_model, device)
        self.position_embedding = Embedding(max_seq_len, d_model, device)
        self.segment_embedding = Embedding(2, d_model, device)  # 句子A和句子B
        self.norm = LayerNorm(d_model, device=device)

        # Transformer编码器层
        self.encoder_layers = []
        for _ in range(n_layers):
            layer = TransformerEncoderLayer(d_model, n_heads, d_ff, device)
            self.encoder_layers.append(layer)

        # Pooler层 - 这是关键添加
        self.pooler = BERTPooler(d_model, device)

        # 预训练任务头
        self.mlm_head = Linear(d_model, vocab_size, device=device)  # 掩码语言模型
        self.nsp_head = Linear(d_model, 2, device=device)  # 下一句预测

    def create_position_ids(self, input_ids):
        """创建位置ID"""
        batch_size, seq_len = input_ids.shape
        position_ids = np.tile(np.arange(seq_len), (batch_size, 1))
        return Tensor(position_ids, device=self.device)

    def forward(self, input_ids, segment_ids=None, attention_mask=None,
                output_hidden_states=False, output_attentions=False, return_dict=True):
        batch_size, seq_len = input_ids.shape

        # 创建位置ID
        position_ids = self.create_position_ids(input_ids)

        # 如果没有提供段落ID，默认全为0
        if segment_ids is None:
            segment_ids = Tensor(np.zeros_like(input_ids.data), device=self.device)

        # 嵌入
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        segment_embeddings = self.segment_embedding(segment_ids)

        # 合并嵌入
        embeddings = Tensor(
            token_embeddings.data + position_embeddings.data + segment_embeddings.data,
            requires_grad=token_embeddings.requires_grad,
            device=self.device
        )
        embeddings = self.norm(embeddings)

        # 通过编码器层
        hidden_states = embeddings
        all_hidden_states = []
        all_attentions = []

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        for layer in self.encoder_layers:
            if output_attentions:
                # 如果需要注意力权重，调用支持返回注意力的forward方法
                hidden_states, attention_weights = layer(hidden_states, attention_mask, return_attention=True)
                all_attentions.append(attention_weights)
            else:
                hidden_states = layer(hidden_states, attention_mask)

            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        # 计算pooler_output
        pooler_output = self.pooler(hidden_states)

        if return_dict:
            return BERTOutput(
                last_hidden_state=hidden_states,
                pooler_output=pooler_output,
                hidden_states=all_hidden_states if output_hidden_states else None,
                attentions=all_attentions if output_attentions else None
            )
        else:
            return hidden_states, pooler_output

    def create_position_ids(self, input_ids):
        """创建位置ID"""
        batch_size, seq_len = input_ids.shape
        position_ids = np.tile(np.arange(seq_len), (batch_size, 1))
        return Tensor(position_ids, device=self.device)

    def forward_mlm(self, input_ids, segment_ids=None, attention_mask=None):
        """前向传播用于掩码语言模型"""
        outputs = self.forward(input_ids, segment_ids, attention_mask)
        mlm_scores = self.mlm_head(outputs.last_hidden_state)
        return mlm_scores

    def forward_nsp(self, input_ids, segment_ids=None, attention_mask=None):
        """前向传播用于下一句预测"""
        outputs = self.forward(input_ids, segment_ids, attention_mask)
        # 使用pooler_output而不是直接使用CLS hidden state
        nsp_scores = self.nsp_head(outputs.pooler_output)
        return nsp_scores


class BERTForSequenceClassification(Module):
    """BERT用于序列分类（使用pooler_output）"""

    def __init__(self, bert_model, num_classes, dropout_rate=0.1, device='cpu'):
        super().__init__()
        self.bert = bert_model
        self.num_classes = num_classes

        # 可选的dropout层
        self.dropout_rate = dropout_rate

        # 分类器
        self.classifier = Linear(bert_model.d_model, num_classes, device=device)

    def forward(self, input_ids, segment_ids=None, attention_mask=None, labels=None):
        # 获取BERT输出
        outputs = self.bert(input_ids, segment_ids, attention_mask)

        # 使用pooler_output进行分类
        pooled_output = outputs.pooler_output

        # 可以在这里添加dropout（简化版本跳过）

        # 分类
        logits = self.classifier(pooled_output)

        return logits

# 使用示例和对比
def demonstrate_cls_vs_pooler():
    """演示CLS hidden state vs pooler output的区别"""

    device = 'cpu'
    bert = BERT(vocab_size=1000, max_seq_len=128, d_model=256,
                n_layers=6, n_heads=8, d_ff=1024, device=device)

    # 模拟输入
    batch_size, seq_len = 2, 10
    input_ids = Tensor(np.random.randint(0, 1000, (batch_size, seq_len)), device=device)

    # 获取输出
    outputs = bert(input_ids)

    # CLS位置的hidden state (原始)
    cls_hidden = outputs.last_hidden_state.data[:, 0, :]  # (batch_size, d_model)

    # Pooler output (经过变换的)
    pooler_out = outputs.pooler_output.data  # (batch_size, d_model)

    print("CLS Hidden State shape:", cls_hidden.shape)
    print("Pooler Output shape:", pooler_out.shape)
    print("Are they the same?", np.allclose(cls_hidden, pooler_out))
    print("Pooler output is typically different due to Linear+Tanh transformation")

    return cls_hidden, pooler_out


class GPT1(Module):
    """GPT-1模型实现"""

    def __init__(self, vocab_size=40478, max_seq_len=512, d_model=768,
                 n_layers=12, n_heads=12, d_ff=3072, device='cpu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.device = device

        # 嵌入层
        self.token_embedding = Embedding(vocab_size, d_model, device)
        self.position_encoding = PositionalEncoding(max_seq_len, d_model, device)

        # Transformer解码器层
        self.decoder_layers = []
        for _ in range(n_layers):
            layer = TransformerDecoderLayer(d_model, n_heads, d_ff, device)
            self.decoder_layers.append(layer)

        # 输出层
        self.ln_f = LayerNorm(d_model, device=device)
        self.lm_head = Linear(d_model, vocab_size, device=device)

    def create_causal_mask(self, seq_len):
        """创建因果掩码（下三角矩阵）"""
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        return Tensor(mask, device=self.device)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape

        # 嵌入
        token_embeddings = self.token_embedding(input_ids)
        embeddings = self.position_encoding(token_embeddings)

        # 创建因果掩码
        causal_mask = self.create_causal_mask(seq_len)

        # 通过解码器层
        hidden_states = embeddings
        for layer in self.decoder_layers:
            hidden_states = layer(hidden_states, causal_mask)

        # 最终归一化和输出投影
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits


# 使用示例
def create_bert_base(device='cpu'):
    """创建BERT-Base模型"""
    return BERT(
        vocab_size=30522,
        max_seq_len=512,
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_ff=3072,
        device=device
    )


def create_bert_large(device='cpu'):
    """创建BERT-Large模型"""
    return BERT(
        vocab_size=30522,
        max_seq_len=512,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        d_ff=4096,
        device=device
    )


def create_gpt1(device='cpu'):
    """创建GPT-1模型"""
    return GPT1(
        vocab_size=40478,
        max_seq_len=512,
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_ff=3072,
        device=device
    )


# 预训练示例
def bert_pretraining_step(model, input_ids, segment_ids, masked_positions, masked_labels,
                          nsp_labels, mlm_loss_fn, nsp_loss_fn):
    """BERT预训练步骤"""
    # MLM任务
    mlm_scores = model.forward_mlm(input_ids, segment_ids)

    # 只计算被掩码位置的损失
    masked_scores = Tensor(
        mlm_scores.data[masked_positions[:, 0], masked_positions[:, 1], :],
        requires_grad=mlm_scores.requires_grad,
        device=mlm_scores.device
    )
    mlm_loss = mlm_loss_fn(masked_scores, masked_labels)

    # NSP任务
    nsp_scores = model.forward_nsp(input_ids, segment_ids)
    nsp_loss = nsp_loss_fn(nsp_scores, nsp_labels)

    # 总损失
    total_loss = Tensor(
        mlm_loss.data + nsp_loss.data,
        requires_grad=True,
        device=model.device
    )

    return total_loss, mlm_loss, nsp_loss


def gpt_training_step(model, input_ids, target_ids, loss_fn):
    """GPT训练步骤"""
    # 前向传播
    logits = model(input_ids)

    # 移位：输入是x[0:n-1]，目标是x[1:n]
    shift_logits = Tensor(logits.data[:, :-1, :], requires_grad=logits.requires_grad, device=logits.device)
    shift_labels = Tensor(target_ids.data[:, 1:], requires_grad=False, device=target_ids.device)

    # 计算损失
    loss = loss_fn(shift_logits, shift_labels)

    return loss


# 微调任务示例
class BERTForSequenceClassification(Module):
    """BERT用于序列分类"""

    def __init__(self, bert_model, num_classes, device='cpu'):
        super().__init__()
        self.bert = bert_model
        self.classifier = Linear(bert_model.d_model, num_classes, device=device)

    def forward(self, input_ids, segment_ids=None, attention_mask=None):
        hidden_states = self.bert(input_ids, segment_ids, attention_mask)
        # 使用[CLS]标记的输出
        cls_output = Tensor(hidden_states.data[:, 0, :], requires_grad=hidden_states.requires_grad,
                            device=hidden_states.device)
        logits = self.classifier(cls_output)
        return logits


class BERTForQuestionAnswering(Module):
    """BERT用于问答任务"""

    def __init__(self, bert_model, device='cpu'):
        super().__init__()
        self.bert = bert_model
        self.qa_outputs = Linear(bert_model.d_model, 2, device=device)  # start和end位置

    def forward(self, input_ids, segment_ids=None, attention_mask=None):
        hidden_states = self.bert(input_ids, segment_ids, attention_mask)
        logits = self.qa_outputs(hidden_states)

        # 分离start和end logits
        start_logits = Tensor(logits.data[:, :, 0], requires_grad=logits.requires_grad, device=logits.device)
        end_logits = Tensor(logits.data[:, :, 1], requires_grad=logits.requires_grad, device=logits.device)

        return start_logits, end_logits


if __name__ == "__main__":
    print("BERT和GPT-1模型测试！")

    # 创建模型示例
    device = 'cpu'  # 或 'cuda' 如果有GPU支持

    # BERT模型
    bert_base = create_bert_base(device)
    print(f"BERT-Base参数量: {bert_base.count_parameters():,}")

    bert_large = create_bert_large(device)
    print(f"BERT-Large参数量: {bert_large.count_parameters():,}")

    # GPT-1模型
    gpt1 = create_gpt1(device)
    print(f"GPT-1参数量: {gpt1.count_parameters():,}")

    # 微调模型示例
    bert_classifier = BERTForSequenceClassification(bert_base, num_classes=2, device=device)
    bert_qa = BERTForQuestionAnswering(bert_base, device=device)

    print("\n模型创建成功，可以开始训练和推理！")

    print("BERT with Pooler Output Implementation")
    print("=====================================")

    # 演示使用
    cls_hidden, pooler_out = demonstrate_cls_vs_pooler()

    # 创建分类模型
    device = 'cpu'
    bert_base = BERT(vocab_size=30522, max_seq_len=512, d_model=768,
                     n_layers=12, n_heads=12, d_ff=3072, device=device)

    classifier = BERTForSequenceClassification(bert_base, num_classes=2, device=device)

    print(f"\nBERT模型参数量: {bert_base.count_parameters():,}")
    print("模型包含完整的pooler_output功能！")
