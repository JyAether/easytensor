import numpy as np
import math
from typing import Optional, Tuple

from core.nn import Module, Linear
from core.nn.transform import PositionalEncoding, TransformerEncoderLayer, MultiHeadAttention
from core.tensor import Tensor
import core.nn as nn


class Attention(Module):
    """
    基础注意力机制
    实现 Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """

    def __init__(self, scale=None):
        super().__init__()
        self.scale = scale

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch_size, seq_len_q, d_k)
            key: (batch_size, seq_len_k, d_k)
            value: (batch_size, seq_len_v, d_v)
            mask: (batch_size, seq_len_q, seq_len_k) or None

        Returns:
            output: (batch_size, seq_len_q, d_v)
            attention_weights: (batch_size, seq_len_q, seq_len_k)
        """
        batch_size, seq_len_q, d_k = query.shape
        seq_len_k = key.shape[1]

        # 计算缩放因子
        if self.scale is None:
            scale = 1.0 / math.sqrt(d_k)
        else:
            scale = self.scale

        # 计算注意力分数: Q @ K^T
        scores = query.matmul(key.transpose(-2, -1))  # (batch, seq_q, seq_k)
        scores = scores * scale

        # 应用掩码（如果有）
        if mask is not None:
            # 将掩码为0的位置设为很大的负数
            mask_value = -1e9
            masked_scores_data = scores.data.copy()
            masked_scores_data[mask.data == 0] = mask_value
            scores = Tensor(masked_scores_data, requires_grad=scores.requires_grad, device=scores.device)

        # 计算注意力权重
        attention_weights = self.softmax(scores, dim=-1)

        # 计算输出: attention_weights @ V
        # value  [batch_size, seq_len_v, d_v] 2,8,64
        # attention_weights  [batch, seq_q, seq_k] 2,8,8
        # output [2 ,8 , 64]
        output = attention_weights.matmul(value)

        return output, attention_weights

    def softmax(self, x, dim=-1):
        """沿指定维度计算softmax"""
        # 为了数值稳定性，减去最大值
        x_max = x.max(axis=dim, keepdims=True)
        x_shifted = x - x_max
        exp_x = x_shifted.exp()
        sum_exp = exp_x.sum(axis=dim, keepdims=True)
        return exp_x / sum_exp


class SelfAttention(Module):
    """
    自注意力机制
    输入相同的序列作为 Q, K, V
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0.0, device='cpu'):
        super().__init__()
        self.multihead_attn = MultiHeadAttention(embed_dim, num_heads, device)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            mask: 掩码

        Returns:
            output: (batch_size, seq_len, embed_dim)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        return self.multihead_attn(x, x, x, mask=mask)


class CrossAttention(Module):
    """
    交叉注意力机制
    用于 Encoder-Decoder 架构
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0.0, device='cpu'):
        super().__init__()
        self.multihead_attn = MultiHeadAttention(
            d_model=embed_dim,
            n_heads=num_heads,
            device=device
        )

    def forward(self, query, key_value, key_padding_mask=None):
        """
        Args:
            query: decoder输入 (batch_size, tgt_len, embed_dim)
            key_value: encoder输出 (batch_size, src_len, embed_dim)
            key_padding_mask: (batch_size, src_len)

        Returns:
            output: (batch_size, tgt_len, embed_dim)
            attention_weights: (batch_size, num_heads, tgt_len, src_len)
        """
        return self.multihead_attn(query, key_value, key_value, mask=key_padding_mask)


class ScaledDotProductAttention(Module):
    """
    缩放点积注意力
    直接实现最基础的注意力计算
    """

    def __init__(self, temperature=None, dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = dropout

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: Query (batch_size, n_head, len_q, d_k)
            k: Key   (batch_size, n_head, len_k, d_k)
            v: Value (batch_size, n_head, len_v, d_v)
            mask: (batch_size, n_head, len_q, len_k)

        Returns:
            output: (batch_size, n_head, len_q, d_v)
            attention: (batch_size, n_head, len_q, len_k)
        """
        d_k = q.shape[-1]

        # 计算注意力分数
        scores = q.matmul(k.transpose(-2, -1))

        # 缩放
        if self.temperature is not None:
            scores = scores / self.temperature
        else:
            scores = scores / math.sqrt(d_k)

        # 应用掩码
        if mask is not None:
            mask_value = -1e9
            masked_scores_data = scores.data.copy()
            masked_scores_data[mask.data == 0] = mask_value
            scores = Tensor(masked_scores_data, requires_grad=scores.requires_grad, device=scores.device)

        # Softmax
        attention = self.softmax(scores, dim=-1)

        # 计算输出
        output = attention.matmul(v)

        return output, attention

    def softmax(self, x, dim=-1):
        """沿指定维度计算softmax"""
        x_max = x.max(axis=dim, keepdims=True)
        x_shifted = x - x_max
        exp_x = x_shifted.exp()
        sum_exp = exp_x.sum(axis=dim, keepdims=True)
        return exp_x / sum_exp


# ==================== 工具函数：创建各种掩码 ====================

def create_padding_mask(seq, pad_token=0):
    """创建填充掩码"""
    # seq: (batch_size, seq_len)
    mask_data = (seq.data != pad_token).astype(np.float32)
    return Tensor(mask_data, device=seq.device)


def create_look_ahead_mask(seq_len, device='cpu'):
    """创建前瞻掩码（用于解码器自注意力）"""
    mask_data = np.triu(np.ones((seq_len, seq_len)), k=1)
    return Tensor(mask_data, device=device)


def create_combined_mask(seq, pad_token=0, device='cpu'):
    """创建组合掩码（填充 + 前瞻）"""
    seq_len = seq.shape[-1]

    # 填充掩码
    padding_mask = create_padding_mask(seq, pad_token)
    # 扩展维度用于广播
    padding_mask_data = np.expand_dims(padding_mask.data, axis=1)

    # 前瞻掩码
    look_ahead_mask = create_look_ahead_mask(seq_len, device)
    look_ahead_mask_data = np.expand_dims(look_ahead_mask.data, axis=0)

    # 组合掩码
    combined_mask_data = np.maximum(padding_mask_data, look_ahead_mask_data)

    return Tensor(combined_mask_data, device=device)


# ==================== 使用示例和测试代码 ====================

if __name__ == "__main__":
    # 测试基础注意力
    print("=" * 50)
    print("测试基础注意力机制")
    print("=" * 50)

    batch_size, seq_len, d_model = 2, 8, 64

    # 创建测试数据
    query = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)
    key = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)
    value = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)

    # 基础注意力
    attention = Attention()
    output, weights = attention(query, key, value)

    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")

    # 测试多头注意力
    print("\n" + "=" * 50)
    print("测试多头注意力机制")
    print("=" * 50)

    d_model = 64
    num_heads = 8
    multihead_attn = MultiHeadAttention(d_model, num_heads)

    x = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)
    # output, weights = multihead_attn(x, x, x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Multi-head attention weights shape: {weights.shape}")

    # 测试自注意力
    print("\n" + "=" * 50)
    print("测试自注意力机制")
    print("=" * 50)

    self_attn = SelfAttention(d_model, num_heads)
    output, weights = self_attn(x)

    print(f"Self-attention output shape: {output.shape}")
    print(f"Self-attention weights shape: {weights.shape}")

    # 测试位置编码
    print("\n" + "=" * 50)
    print("测试位置编码")
    print("=" * 50)

    pos_encoding = PositionalEncoding(d_model)
    encoded_x = pos_encoding(x)

    print(f"Original shape: {x.shape}")
    print(f"After positional encoding: {encoded_x.shape}")

    # 测试掩码
    print("\n" + "=" * 50)
    print("测试掩码功能")
    print("=" * 50)

    # 创建带填充的序列 (假设0是填充token)
    seq_with_padding = Tensor([[1, 2, 3, 0, 0],
                               [1, 2, 3, 4, 5]])

    padding_mask = create_padding_mask(seq_with_padding, pad_token=0)
    look_ahead_mask = create_look_ahead_mask(5)

    print(f"Sequence with padding: {seq_with_padding.shape}")
    print(f"Padding mask shape: {padding_mask.shape}")
    print(f"Look-ahead mask shape: {look_ahead_mask.shape}")
    print(f"Padding mask:\n{padding_mask.data}")
    print(f"Look-ahead mask:\n{look_ahead_mask.data}")

    # 测试Transformer编码器层
    print("\n" + "=" * 50)
    print("测试Transformer编码器层")
    print("=" * 50)

    encoder_layer = TransformerEncoderLayer(d_model=64, n_heads=8, d_ff=256)
    encoder_output = encoder_layer(x)

    print(f"Encoder input shape: {x.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")

    # 测试带掩码的注意力
    print("\n" + "=" * 50)
    print("测试带掩码的注意力")
    print("=" * 50)

    # 创建一个简单的掩码（忽略最后两个位置）
    mask = Tensor(np.array([[1, 1, 1, 1, 1, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1]]))  # (batch_size, seq_len)

    masked_output, masked_weights = self_attn(x, mask=mask)

    print(f"Masked attention output shape: {masked_output.shape}")
    print(f"Masked attention weights shape: {masked_weights.shape}")

    # 测试交叉注意力
    print("\n" + "=" * 50)
    print("测试交叉注意力")
    print("=" * 50)

    cross_attn = CrossAttention(d_model, num_heads=8)

    # 模拟encoder-decoder场景
    encoder_output = Tensor(np.random.randn(batch_size, 10, d_model), requires_grad=True)  # 源序列长度=10
    decoder_input = Tensor(np.random.randn(batch_size, 6, d_model), requires_grad=True)  # 目标序列长度=6

    cross_output, cross_weights = cross_attn(decoder_input, encoder_output)

    print(f"Decoder input shape: {decoder_input.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Cross attention output shape: {cross_output.shape}")
    print(f"Cross attention weights shape: {cross_weights.shape}")

    # 测试缩放点积注意力
    print("\n" + "=" * 50)
    print("测试缩放点积注意力")
    print("=" * 50)

    scaled_attn = ScaledDotProductAttention()

    # 创建多头格式的输入
    batch_size, num_heads, seq_len, head_dim = 2, 8, 10, 8
    q = Tensor(np.random.randn(batch_size, num_heads, seq_len, head_dim), requires_grad=True)
    k = Tensor(np.random.randn(batch_size, num_heads, seq_len, head_dim), requires_grad=True)
    v = Tensor(np.random.randn(batch_size, num_heads, seq_len, head_dim), requires_grad=True)

    scaled_output, scaled_weights = scaled_attn(q, k, v)

    print(f"Scaled attention Q shape: {q.shape}")
    print(f"Scaled attention K shape: {k.shape}")
    print(f"Scaled attention V shape: {v.shape}")
    print(f"Scaled attention output shape: {scaled_output.shape}")
    print(f"Scaled attention weights shape: {scaled_weights.shape}")

    print("\n" + "=" * 50)
    print("测试完成！")


# ==================== 高级使用示例 ====================

class SimpleTransformer(Module):
    """简单的Transformer模型示例"""

    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len=512, device='cpu'):
        super().__init__()
        self.d_model = d_model
        self.device = device

        # 词嵌入层
        self.embedding = Linear(vocab_size, d_model, device=device)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, device)

        # Transformer编码器层
        self.encoder_layers = []
        for _ in range(num_layers):
            layer = TransformerEncoderLayer(
                d_model=d_model,
                n_heads=num_heads,
                d_ff=d_model * 4,
                device=device
            )
            self.encoder_layers.append(layer)

        # 输出层
        self.output_linear = Linear(d_model, vocab_size, device=device)

    def forward(self, input_ids, padding_mask=None):
        """
        Args:
            input_ids: (batch_size, seq_len) - 输入token ids
            padding_mask: (batch_size, seq_len) - 填充掩码
        """
        # 词嵌入
        # 注意：这里假设input_ids是one-hot编码或需要转换
        # 实际使用中可能需要embedding lookup
        x = self.embedding(input_ids.data.astype(np.float32))

        # 位置编码
        x = self.pos_encoding(x)

        # 通过编码器层
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, padding_mask)

        # 输出投影
        output = self.output_linear(x)

        return output


def example_usage():
    # 模型参数
    vocab_size = 1000
    d_model = 128
    num_heads = 8
    num_layers = 2
    batch_size = 2
    seq_len = 16

    # 创建模型
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers
    )

    # 创建输入数据（随机token ids）
    # 注意：token ids 应该是整数类型
    input_ids = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)).astype(np.int32))

    # 创建填充掩码 - 修复维度匹配问题
    # 形状应该是 (batch_size, seq_len)
    # 1 表示真实token，0 表示填充token
    padding_mask = Tensor(np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # 第一个序列：前12个是真实token，后4个是padding
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  # 第二个序列：前10个是真实token，后6个是padding
    ]).astype(np.float32))

    print(f"input_ids shape: {input_ids.data.shape}")
    print(f"padding_mask shape: {padding_mask.data.shape}")

    # 前向传播
    try:
        output = model(input_ids, padding_mask)
        print(f"Output shape: {output.data.shape}")
        print("Forward pass successful!")
    except Exception as e:
        print(f"Error: {e}")

        # 如果还有 embedding 错误，检查以下几点：
        print("\n调试信息:")
        print(f"vocab_size: {vocab_size}")
        print(f"d_model: {d_model}")
        print(f"input_ids min/max: {input_ids.data.min()}/{input_ids.data.max()}")
        print(f"input_ids dtype: {input_ids.data.dtype}")

        # 检查 embedding 权重
        if hasattr(model.embedding, 'weight'):
            print(f"embedding weight shape: {model.embedding.weight.shape}")

    # 或者如果你想要简单的全1掩码（表示没有padding）
    simple_mask = Tensor(np.ones((batch_size, seq_len)).astype(np.float32))
    print(f"\nSimple mask shape: {simple_mask.data.shape}")

    try:
        output_simple = model(input_ids, simple_mask)
        print(f"Output with simple mask shape: {output_simple.data.shape}")
        print("Forward pass with simple mask successful!")
    except Exception as e:
        print(f"Error with simple mask: {e}")


def attention_visualization_example():
    """注意力可视化示例"""
    print("\n" + "=" * 60)
    print("注意力权重可视化示例")
    print("=" * 60)

    # 创建一个小的示例
    batch_size, seq_len, d_model = 1, 5, 32
    num_heads = 4

    # 创建输入序列（模拟句子："我 爱 深度 学习"）
    x = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)

    # 创建多头自注意力
    self_attn = SelfAttention(d_model, num_heads)

    # 计算注意力
    output, attention_weights = self_attn(x)

    print(f"输入序列长度: {seq_len}")
    print(f"注意力头数: {num_heads}")
    print(f"注意力权重shape: {attention_weights.shape}")

    # 打印第一个头的注意力权重
    first_head_weights = attention_weights.data[0, 0, :, :]  # [seq_len, seq_len]

    print("\n第一个注意力头的权重矩阵:")
    print("(行=query位置, 列=key位置)")
    print("-" * 40)
    for i in range(seq_len):
        row_str = " ".join([f"{first_head_weights[i, j]:.3f}" for j in range(seq_len)])
        print(f"位置 {i}: {row_str}")

    # 分析注意力模式
    print("\n注意力分析:")
    for i in range(seq_len):
        max_attention_pos = np.argmax(first_head_weights[i, :])
        max_attention_score = first_head_weights[i, max_attention_pos]
        print(f"位置 {i} 最关注位置 {max_attention_pos} (权重: {max_attention_score:.3f})")


if __name__ == "__main__":
    # 运行完整示例
    example_usage()
    attention_visualization_example()
