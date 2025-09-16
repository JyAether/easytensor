import numpy as np
import math
from typing import Optional, Tuple, List

from core.nn import Module, Linear, CrossEntropyLoss
from core.tensor import Tensor


class MultiHeadAttention(Module):
    """多头注意力机制"""

    def __init__(self, d_model, n_heads, device='cpu'):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.device = device

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Q, K, V 投影矩阵
        self.W_q = Linear(d_model, d_model, device=device)
        self.W_k = Linear(d_model, d_model, device=device)
        self.W_v = Linear(d_model, d_model, device=device)
        self.W_o = Linear(d_model, d_model, device=device)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """缩放点积注意力"""
        xp = Q._get_array_module() if hasattr(Q, '_get_array_module') else np

        # 计算注意力分数: Q @ K.T / sqrt(d_k)
        # Q: (batch_size, n_heads, seq_len_q, d_k)
        # K: (batch_size, n_heads, seq_len_k, d_k)
        # V: (batch_size, n_heads, seq_len_v, d_k)
        scores = Q.data @ K.data.transpose(0, 1, 3, 2) / math.sqrt(self.d_k)

        # 应用掩码（如果提供）
        if mask is not None:
            # 将掩码位置设为很小的负数，softmax后接近0
            scores = scores + (mask.data * -1e9)

        # Softmax归一化
        exp_scores = xp.exp(scores - xp.max(scores, axis=-1, keepdims=True))
        attention_weights = exp_scores / xp.sum(exp_scores, axis=-1, keepdims=True)

        # 应用注意力权重到V
        output = attention_weights @ V.data

        return Tensor(output, requires_grad=Q.requires_grad, device=Q.device), \
            Tensor(attention_weights, requires_grad=False, device=Q.device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        seq_len_q = query.shape[1]
        seq_len_k = key.shape[1]
        seq_len_v = value.shape[1]

        # 线性投影得到Q, K, V
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 重塑为多头格式: (batch_size, n_heads, seq_len, d_k)
        Q_reshaped = Q.data.reshape(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K_reshaped = K.data.reshape(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V_reshaped = V.data.reshape(batch_size, seq_len_v, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        Q_tensor = Tensor(Q_reshaped, requires_grad=Q.requires_grad, device=Q.device)
        K_tensor = Tensor(K_reshaped, requires_grad=K.requires_grad, device=K.device)
        V_tensor = Tensor(V_reshaped, requires_grad=V.requires_grad, device=V.device)

        # 计算多头注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(
            Q_tensor, K_tensor, V_tensor, mask
        )

        # 合并多头结果: (batch_size, seq_len_q, d_model)
        attention_output_concat = attention_output.data.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len_q, self.d_model
        )

        # 最终线性投影
        output = self.W_o(Tensor(attention_output_concat, requires_grad=attention_output.requires_grad,
                                 device=attention_output.device))

        return output, attention_weights


class LayerNorm(Module):
    """层归一化"""

    def __init__(self, d_model, eps=1e-5, device='cpu'):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device

        # 可学习参数
        self.gamma = Tensor(np.ones(d_model), requires_grad=True, device=device)
        self.beta = Tensor(np.zeros(d_model), requires_grad=True, device=device)

    def forward(self, x):
        xp = x._get_array_module() if hasattr(x, '_get_array_module') else np

        # 计算均值和方差
        mean = xp.mean(x.data, axis=-1, keepdims=True)
        var = xp.var(x.data, axis=-1, keepdims=True)

        # 归一化
        normalized = (x.data - mean) / xp.sqrt(var + self.eps)

        # 应用可学习参数
        output_data = normalized * self.gamma.data + self.beta.data

        return Tensor(output_data, requires_grad=x.requires_grad, device=x.device)


class FeedForward(Module):
    """位置相关的前馈网络"""

    def __init__(self, d_model, d_ff, device='cpu'):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff, device=device)
        self.linear2 = Linear(d_ff, d_model, device=device)

    def forward(self, x):
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        hidden = self.linear1(x).relu()
        output = self.linear2(hidden)
        return output


class PositionalEncoding(Module):
    """位置编码"""

    def __init__(self, d_model, max_seq_len=5000, device='cpu'):
        super().__init__()
        self.d_model = d_model
        self.device = device

        # 创建位置编码矩阵
        pe = np.zeros((max_seq_len, d_model))
        position = np.arange(0, max_seq_len).reshape(-1, 1)

        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = np.cos(position * div_term)  # 奇数位置

        self.register_buffer('pe', pe)
        self.pe_tensor = Tensor(pe, requires_grad=False, device=device)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape

        # 添加位置编码
        pos_encoding = self.pe_tensor.data[:seq_len, :]
        output_data = x.data + pos_encoding

        return Tensor(output_data, requires_grad=x.requires_grad, device=x.device)

    def register_buffer(self, name, tensor):
        """简单的缓冲区注册"""
        pass


class Embedding(Module):
    """词嵌入层"""

    def __init__(self, vocab_size, d_model, device='cpu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.device = device

        # 嵌入权重初始化
        self.weight = Tensor(
            np.random.normal(0, 1 / math.sqrt(d_model), (vocab_size, d_model)),
            requires_grad=True,
            device=device
        )

    def forward(self, input_ids):
        # 简化的嵌入查找
        batch_size, seq_len = input_ids.shape
        xp = input_ids._get_array_module() if hasattr(input_ids, '_get_array_module') else np

        output_data = xp.zeros((batch_size, seq_len, self.d_model))

        for i in range(batch_size):
            for j in range(seq_len):
                token_id = int(input_ids.data[i, j])
                if 0 <= token_id < self.vocab_size:
                    output_data[i, j] = self.weight.data[token_id]

        # 按照Transformer论文，嵌入乘以sqrt(d_model)
        output_data *= math.sqrt(self.d_model)

        return Tensor(output_data, requires_grad=self.weight.requires_grad, device=self.device)


class TransformerEncoderLayer(Module):
    """Transformer编码器层"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, device='cpu'):
        super().__init__()
        self.d_model = d_model
        self.device = device

        # 多头自注意力
        self.self_attn = MultiHeadAttention(d_model, n_heads, device)
        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, device)
        # 层归一化
        self.norm1 = LayerNorm(d_model, device=device)
        self.norm2 = LayerNorm(d_model, device=device)
        # Dropout
        self.dropout = dropout

    def forward(self, src, src_mask=None):
        """
        src: (batch_size, src_len, d_model)
        src_mask: (batch_size, n_heads, src_len, src_len) 或 None
        """
        # 多头自注意力 + 残差连接 + 层归一化
        attn_output, _ = self.self_attn(src, src, src, src_mask)
        src2 = self.norm1(Tensor(src.data + attn_output.data, requires_grad=src.requires_grad, device=src.device))

        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(src2)
        output = self.norm2(Tensor(src2.data + ff_output.data, requires_grad=src2.requires_grad, device=src2.device))

        return output


class TransformerDecoderLayer(Module):
    """Transformer解码器层"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, device='cpu'):
        super().__init__()
        self.d_model = d_model
        self.device = device

        # 掩码多头自注意力
        self.self_attn = MultiHeadAttention(d_model, n_heads, device)
        # 编码器-解码器注意力
        self.cross_attn = MultiHeadAttention(d_model, n_heads, device)
        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, device)
        # 层归一化
        self.norm1 = LayerNorm(d_model, device=device)
        self.norm2 = LayerNorm(d_model, device=device)
        self.norm3 = LayerNorm(d_model, device=device)
        # Dropout
        self.dropout = dropout

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        tgt: (batch_size, tgt_len, d_model) - 目标序列
        memory: (batch_size, src_len, d_model) - 编码器输出
        tgt_mask: (batch_size, n_heads, tgt_len, tgt_len) - 目标掩码（因果掩码）
        memory_mask: (batch_size, n_heads, tgt_len, src_len) - 源掩码
        """
        # 1. 掩码多头自注意力 + 残差 + 层归一化
        self_attn_output, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt2 = self.norm1(Tensor(tgt.data + self_attn_output.data, requires_grad=tgt.requires_grad, device=tgt.device))

        # 2. 编码器-解码器注意力 + 残差 + 层归一化
        cross_attn_output, _ = self.cross_attn(tgt2, memory, memory, memory_mask)
        tgt3 = self.norm2(
            Tensor(tgt2.data + cross_attn_output.data, requires_grad=tgt2.requires_grad, device=tgt2.device))

        # 3. 前馈网络 + 残差 + 层归一化
        ff_output = self.feed_forward(tgt3)
        output = self.norm3(Tensor(tgt3.data + ff_output.data, requires_grad=tgt3.requires_grad, device=tgt3.device))

        return output


class TransformerEncoder(Module):
    """Transformer编码器"""

    def __init__(self, encoder_layer, num_layers, device='cpu'):
        super().__init__()
        # 创建多个独立的编码器层
        self.layers = [TransformerEncoderLayer(
            encoder_layer.d_model,
            encoder_layer.self_attn.n_heads,
            encoder_layer.feed_forward.linear1.out_features,
            encoder_layer.dropout,
            device
        ) for _ in range(num_layers)]
        self.num_layers = num_layers
        self.device = device
        self.norm = LayerNorm(encoder_layer.d_model, device=device)

    def forward(self, src, src_mask=None):
        """
        src: (batch_size, src_len, d_model)
        src_mask: 源序列掩码
        """
        output = src

        # 通过所有编码器层
        for layer in self.layers:
            output = layer(output, src_mask)

        # 最终层归一化
        return self.norm(output)


class TransformerDecoder(Module):
    """Transformer解码器"""

    def __init__(self, decoder_layer, num_layers, device='cpu'):
        super().__init__()
        # 创建多个独立的解码器层
        self.layers = [TransformerDecoderLayer(
            decoder_layer.d_model,
            decoder_layer.self_attn.n_heads,
            decoder_layer.feed_forward.linear1.out_features,
            decoder_layer.dropout,
            device
        ) for _ in range(num_layers)]
        self.num_layers = num_layers
        self.device = device
        self.norm = LayerNorm(decoder_layer.d_model, device=device)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        tgt: (batch_size, tgt_len, d_model)
        memory: (batch_size, src_len, d_model)
        """
        output = tgt

        # 通过所有解码器层
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask)

        # 最终层归一化
        return self.norm(output)


class Transformer(Module):
    """完整的Transformer模型"""

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048,
                 max_seq_len=5000, dropout=0.1, device='cpu'):
        super().__init__()
        self.d_model = d_model
        self.device = device

        # 嵌入层
        self.src_embedding = Embedding(src_vocab_size, d_model, device)
        self.tgt_embedding = Embedding(tgt_vocab_size, d_model, device)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, device)

        # 编码器
        encoder_layer = TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, device)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, device)

        # 解码器
        decoder_layer = TransformerDecoderLayer(d_model, n_heads, d_ff, dropout, device)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, device)

        # 输出投影
        self.generator = Linear(d_model, tgt_vocab_size, device=device)

        self.dropout = dropout

    def encode(self, src, src_mask=None):
        """编码源序列"""
        # 嵌入 + 位置编码
        src_emb = self.src_embedding(src)
        src_emb = self.pos_encoding(src_emb)

        # 通过编码器
        memory = self.encoder(src_emb, src_mask)
        return memory

    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """解码目标序列"""
        # 嵌入 + 位置编码
        tgt_emb = self.tgt_embedding(tgt)
        tgt_emb = self.pos_encoding(tgt_emb)

        # 通过解码器
        output = self.decoder(tgt_emb, memory, tgt_mask, memory_mask)
        return output

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """完整的前向传播"""
        # 编码
        memory = self.encode(src, src_mask)

        # 解码
        output = self.decode(tgt, memory, tgt_mask, memory_mask)

        # 输出投影
        return self.generator(output)

    def create_pad_mask(self, seq, pad_idx=0):
        """创建填充掩码"""
        # seq: (batch_size, seq_len)
        batch_size, seq_len = seq.shape
        mask = (seq.data == pad_idx)
        # 扩展维度以匹配注意力形状: (batch_size, 1, 1, seq_len)
        mask = mask[:, np.newaxis, np.newaxis, :]
        return Tensor(mask, device=seq.device)

    def create_causal_mask(self, size, device='cpu'):
        """创建因果掩码（下三角掩码）"""
        mask = np.triu(np.ones((size, size)), k=1).astype(bool)
        return Tensor(mask, device=device)

    def create_look_ahead_mask(self, tgt_len, src_len, device='cpu'):
        """为解码器创建适当维度的掩码"""
        # 对于交叉注意力，通常不需要特殊掩码，除非有填充
        mask = np.zeros((tgt_len, src_len), dtype=bool)
        return Tensor(mask, device=device)

    def generate(self, src, src_mask, max_len=50, start_token=1, end_token=2):
        """贪心生成"""
        # 编码源序列
        memory = self.encode(src, src_mask)

        batch_size = src.shape[0]

        # 初始化目标序列
        tgt = Tensor(np.full((batch_size, 1), start_token), device=self.device)

        for _ in range(max_len):
            # 创建因果掩码
            tgt_len = tgt.shape[1]
            tgt_mask = self.create_causal_mask(tgt_len, self.device)

            # 扩展掩码维度以匹配多头注意力
            tgt_mask_expanded = np.broadcast_to(
                tgt_mask.data[np.newaxis, np.newaxis, :, :],
                (batch_size, 1, tgt_len, tgt_len)
            )
            tgt_mask = Tensor(tgt_mask_expanded, device=self.device)

            # 解码
            output = self.decode(tgt, memory, tgt_mask)

            # 预测下一个token
            next_token_logits = self.generator(Tensor(
                output.data[:, -1:, :],
                requires_grad=output.requires_grad,
                device=output.device
            ))

            # 贪心选择
            next_token = np.argmax(next_token_logits.data, axis=-1)

            # 添加到目标序列
            tgt_data = np.concatenate([tgt.data, next_token], axis=1)
            tgt = Tensor(tgt_data, device=self.device)

            # 检查是否生成结束标记
            if np.all(next_token == end_token):
                break

        return tgt


# 训练器类
class TransformerTrainer:
    """Transformer训练器"""

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.criterion = CrossEntropyLoss()

    def train_step(self, src, tgt, src_mask=None, tgt_mask=None):
        """单步训练"""
        batch_size = src.shape[0]

        # 准备输入和标签
        tgt_input = Tensor(tgt.data[:, :-1], device=self.device)  # 去掉最后一个token
        tgt_output = Tensor(tgt.data[:, 1:], device=self.device)  # 去掉第一个token

        # 创建目标掩码
        if tgt_mask is None:
            tgt_len = tgt_input.shape[1]
            causal_mask = self.model.create_causal_mask(tgt_len, self.device)
            # 扩展掩码维度
            tgt_mask = np.broadcast_to(
                causal_mask.data[np.newaxis, np.newaxis, :, :],
                (batch_size, 1, tgt_len, tgt_len)
            )
            tgt_mask = Tensor(tgt_mask, device=self.device)

        # 前向传播
        output = self.model(src, tgt_input, src_mask, tgt_mask)

        # 计算损失
        output_flat = Tensor(
            output.data.reshape(-1, output.shape[-1]),
            requires_grad=output.requires_grad,
            device=output.device
        )
        tgt_flat = Tensor(
            tgt_output.data.reshape(-1),
            requires_grad=False,
            device=tgt_output.device
        )

        loss = self.criterion(output_flat, tgt_flat)
        return loss, output


def create_toy_dataset():
    """创建玩具数据集进行测试"""
    # 简单的数字序列转换任务：将输入序列每个数字+1
    np.random.seed(42)

    batch_size = 4
    seq_len = 10
    vocab_size = 20

    # 源序列：随机数字序列
    src_data = np.random.randint(3, vocab_size - 3, (batch_size, seq_len))

    # 目标序列：每个数字+1，添加开始和结束标记
    tgt_input = np.ones((batch_size, seq_len + 1), dtype=int)  # 开始标记为1
    tgt_input[:, 1:] = src_data + 1

    tgt_output = np.ones((batch_size, seq_len + 1), dtype=int)
    tgt_output[:, :-1] = src_data + 1
    tgt_output[:, -1] = 2  # 结束标记为2

    return src_data, tgt_input, tgt_output


def test_transformer_complete_pipeline():
    """测试完整的Transformer编码-解码流程"""
    print("=== Transformer 完整编码-解码流程测试 ===\n")

    # 模型配置
    config = {
        'src_vocab_size': 25,
        'tgt_vocab_size': 25,
        'd_model': 128,
        'n_heads': 8,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'd_ff': 256,
        'max_seq_len': 100,
        'dropout': 0.1,
        'device': 'cpu'
    }

    print(f"模型配置: {config}")
    print()

    # 创建模型
    model = Transformer(**config)
    trainer = TransformerTrainer(model)

    print(f"模型参数量: {model.count_parameters():,}")
    print()

    # 创建测试数据
    src_data, tgt_input, tgt_output = create_toy_dataset()

    src = Tensor(src_data, device=config['device'])
    tgt_in = Tensor(tgt_input, device=config['device'])
    tgt_out = Tensor(tgt_output, device=config['device'])

    print("=== 数据示例 ===")
    print(f"源序列: {src_data[0]}")
    print(f"目标输入: {tgt_input[0]}")
    print(f"目标输出: {tgt_output[0]}")
    print()

    # 1. 编码阶段测试
    print("=== 1. 编码阶段 ===")
    src_mask = model.create_pad_mask(src, pad_idx=0)
    print(f"源序列形状: {src.shape}")
    print(f"源掩码形状: {src_mask.shape}")

    # 编码
    memory = model.encode(src, src_mask)
    print(f"编码器输出形状: {memory.shape}")
    print(f"编码器输出示例 (前3个token的前5维): ")
    print(memory.data[0, :3, :5])
    print()

    # 2. 解码阶段测试
    print("=== 2. 解码阶段 ===")
    batch_size = src.shape[0]

    # 创建因果掩码
    tgt_len = tgt_in.shape[1]
    causal_mask = model.create_causal_mask(tgt_len, config['device'])
    tgt_mask = np.broadcast_to(
        causal_mask.data[np.newaxis, np.newaxis, :, :],
        (batch_size, 1, tgt_len, tgt_len)
    )
    tgt_mask = Tensor(tgt_mask, device=config['device'])

    print(f"目标序列形状: {tgt_in.shape}")
    print(f"因果掩码形状: {tgt_mask.shape}")
    print(f"因果掩码示例 (5x5): ")
    print(causal_mask.data[:5, :5].astype(int))
    print()

    # 解码
    decoder_output = model.decode(tgt_in, memory, tgt_mask)
    print(f"解码器输出形状: {decoder_output.shape}")
    print()

    # 3. 完整前向传播测试
    print("=== 3. 完整前向传播 ===")
    output = model(src, tgt_in, src_mask, tgt_mask)
    print(f"最终输出形状: {output.shape}")
    print(f"最终输出示例 (第1个样本，第1个位置的前10个logits): ")
    print(output.data[0, 0, :10])
    print()

    # 4. 训练步骤测试
    print("=== 4. 训练步骤测试 ===")
    loss, train_output = trainer.train_step(src, tgt_out, src_mask)
    print(f"训练损失: {loss.data:.4f}")
    print(f"训练输出形状: {train_output.shape}")
    print()

    # 5. 生成测试
    print("=== 5. 生成测试 ===")
    print("使用贪心搜索生成序列...")

    # 使用第一个源序列进行生成
    test_src = Tensor(src_data[:1], device=config['device'])  # 只取第一个样本
    test_src_mask = model.create_pad_mask(test_src, pad_idx=0)

    generated = model.generate(
        test_src, test_src_mask,
        max_len=15, start_token=1, end_token=2
    )

    print(f"源序列: {src_data[0]}")
    print(f"生成序列: {generated.data[0]}")
    print(f"期望序列: {tgt_output[0]}")
    print()

    # 6. 注意力权重分析
    print("=== 6. 注意力机制分析 ===")

    # 获取编码器第一层的注意力权重
    src_emb = model.src_embedding(src)
    src_emb = model.pos_encoding(src_emb)

    # 手动通过第一个编码器层以获取注意力权重
    first_encoder_layer = model.encoder.layers[0]
    _, src_attn_weights = first_encoder_layer.self_attn(src_emb, src_emb, src_emb, src_mask)

    print(f"源序列自注意力权重形状: {src_attn_weights.shape}")
    print(f"第一个样本，第一个头，前5x5注意力权重:")
    print(src_attn_weights.data[0, 0, :5, :5])
    print()

    # 7. 模型组件测试
    print("=== 7. 模型组件测试 ===")
    print(f"源词嵌入权重形状: {model.src_embedding.weight.shape}")
    print(f"目标词嵌入权重形状: {model.tgt_embedding.weight.shape}")
    print(f"编码器层数: {model.encoder.num_layers}")
    print(f"解码器层数: {model.decoder.num_layers}")
    print(f"输出投影权重形状: {model.generator.weight.shape}")
    print()

    print("=== Transformer 测试完成 ===")

    return {
        'model': model,
        'loss': loss.data,
        'memory': memory,
        'decoder_output': decoder_output,
        'final_output': output,
        'generated_sequence': generated.data[0],
        'attention_weights': src_attn_weights
    }


def demo_translation_task():
    """演示机器翻译任务"""
    print("\n=== 机器翻译任务演示 ===")

    # 创建简化的英中翻译数据集
    # 词汇表映射（简化版）
    en_vocab = {
        '<pad>': 0, '<sos>': 1, '<eos>': 2,
        'i': 3, 'love': 4, 'you': 5, 'hello': 6, 'world': 7,
        'good': 8, 'morning': 9, 'how': 10, 'are': 11, 'thank': 12, 'very': 13, 'much': 14
    }

    zh_vocab = {
        '<pad>': 0, '<sos>': 1, '<eos>': 2,
        '我': 3, '爱': 4, '你': 5, '你好': 6, '世界': 7,
        '早上': 8, '好': 9, '怎么': 10, '样': 11, '谢谢': 12, '非常': 13, '感谢': 14
    }

    # 训练样例
    en_sentences = [
        [6, 7],  # hello world
        [3, 4, 5],  # i love you
        [8, 9],  # good morning
        [10, 11, 5],  # how are you
        [12, 5, 13, 14]  # thank you very much
    ]

    zh_sentences = [
        [6, 7],  # 你好 世界
        [3, 4, 5],  # 我 爱 你
        [8, 9],  # 早上 好
        [5, 10, 11],  # 你 怎么 样
        [12, 5]  # 谢谢 你
    ]

    # 添加开始和结束标记
    def add_special_tokens(sentences):
        result = []
        for sent in sentences:
            result.append([1] + sent + [2])  # <sos> + sentence + <eos>
        return result

    en_data = add_special_tokens(en_sentences)
    zh_data = add_special_tokens(zh_sentences)

    # 填充到相同长度
    max_len = max(max(len(s) for s in en_data), max(len(s) for s in zh_data))

    def pad_sequences(sequences, max_len):
        padded = []
        for seq in sequences:
            padded_seq = seq + [0] * (max_len - len(seq))
            padded.append(padded_seq)
        return np.array(padded)

    en_padded = pad_sequences(en_data, max_len)
    zh_padded = pad_sequences(zh_data, max_len)

    print(f"英文序列: {en_padded}")
    print(f"中文序列: {zh_padded}")
    print(f"序列长度: {max_len}")
    print()

    # 创建翻译模型
    translation_model = Transformer(
        src_vocab_size=len(en_vocab),
        tgt_vocab_size=len(zh_vocab),
        d_model=64,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        device='cpu'
    )

    print(f"翻译模型参数量: {translation_model.count_parameters():,}")
    print()

    # 转换为张量
    src_tensor = Tensor(en_padded, device='cpu')
    tgt_tensor = Tensor(zh_padded, device='cpu')

    # 创建掩码
    src_mask = translation_model.create_pad_mask(src_tensor, pad_idx=0)

    # 模拟训练
    trainer = TransformerTrainer(translation_model)

    print("开始训练演示（3轮）...")
    for epoch in range(3):
        loss, output = trainer.train_step(src_tensor, tgt_tensor, src_mask)
        print(f"Epoch {epoch + 1}, Loss: {loss.data:.4f}")
    print()

    # 测试翻译
    print("=== 翻译测试 ===")
    test_sentence = np.array([[6, 7, 2]])  # "hello world <eos>"
    test_tensor = Tensor(test_sentence, device='cpu')
    test_mask = translation_model.create_pad_mask(test_tensor, pad_idx=0)

    # 生成翻译
    translated = translation_model.generate(
        test_tensor, test_mask,
        max_len=10, start_token=1, end_token=2
    )

    # 解码结果
    id2word_en = {v: k for k, v in en_vocab.items()}
    id2word_zh = {v: k for k, v in zh_vocab.items()}

    print(f"输入: hello world")
    print(f"输入token IDs: {test_sentence[0]}")
    print(f"翻译token IDs: {translated.data[0]}")

    translated_words = [id2word_zh.get(int(tok), '<unk>') for tok in translated.data[0]]
    print(f"翻译结果: {' '.join(translated_words)}")
    print()

    # 额外测试更多句子
    print("=== 更多翻译测试 ===")
    test_cases = [
        ([3, 4, 5, 2], "i love you"),  # 我爱你
        ([8, 9, 2], "good morning"),  # 早上好
        ([10, 11, 5, 2], "how are you")  # 你怎么样
    ]

    for test_ids, test_text in test_cases:
        test_input = np.array([test_ids])
        test_tensor = Tensor(test_input, device='cpu')
        test_mask = translation_model.create_pad_mask(test_tensor, pad_idx=0)

        translated = translation_model.generate(
            test_tensor, test_mask,
            max_len=10, start_token=1, end_token=2
        )

        translated_words = [id2word_zh.get(int(tok), '<unk>') for tok in translated.data[0]]
        print(f"输入: {test_text}")
        print(f"翻译结果: {''.join([w for w in translated_words if w not in ['<sos>', '<eos>', '<pad>']])}")
        print()


def analyze_attention_patterns():
    """分析注意力模式"""
    print("=== 注意力模式分析 ===")

    # 创建简单测试序列
    test_seq = np.array([[3, 4, 5, 6, 7, 0, 0, 0]])  # 带填充的序列
    test_tensor = Tensor(test_seq, device='cpu')

    # 创建简单的注意力层进行测试
    attention_layer = MultiHeadAttention(d_model=64, n_heads=4, device='cpu')

    # 创建简单嵌入
    embedding = Embedding(vocab_size=20, d_model=64, device='cpu')
    embedded = embedding(test_tensor)

    # 计算自注意力
    attn_output, attn_weights = attention_layer(embedded, embedded, embedded)

    print(f"输入序列: {test_seq[0]}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print(f"第一个头的注意力权重:")
    print(attn_weights.data[0, 0])
    print()

    # 分析注意力模式
    first_head_weights = attn_weights.data[0, 0]  # 第一个样本，第一个头

    print("注意力模式分析:")
    for i, weights in enumerate(first_head_weights):
        if test_seq[0, i] != 0:  # 非填充位置
            max_att_idx = np.argmax(weights)
            max_att_val = np.max(weights)
            print(f"位置 {i} (token {test_seq[0, i]}) 最关注位置 {max_att_idx} (权重: {max_att_val:.3f})")
    print()


def benchmark_model_sizes():
    """基准测试不同模型大小"""
    print("=== 模型大小基准测试 ===")

    configs = [
        {'name': 'Tiny', 'd_model': 64, 'n_heads': 2, 'layers': 2, 'd_ff': 128},
        {'name': 'Small', 'd_model': 128, 'n_heads': 4, 'layers': 3, 'd_ff': 256},
        {'name': 'Base', 'd_model': 256, 'n_heads': 8, 'layers': 4, 'd_ff': 512},
        {'name': 'Large', 'd_model': 512, 'n_heads': 8, 'layers': 6, 'd_ff': 2048}
    ]

    print("| 模型   | d_model | heads | layers | d_ff  | 参数量    |")
    for config in configs:
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            num_encoder_layers=config['layers'],
            num_decoder_layers=config['layers'],
            d_ff=config['d_ff'],
            device='cpu'
        )

        params = model.count_parameters()
        print(f"| {config['name']:<6} | {config['d_model']:<7} | {config['n_heads']:<5} | "
              f"{config['layers']:<6} | {config['d_ff']:<5} | {params:>8,} |")

    print()


def visualize_positional_encoding():
    """可视化位置编码"""
    print("=== 位置编码可视化 ===")

    pos_encoder = PositionalEncoding(d_model=16, max_seq_len=20, device='cpu')

    # 创建虚拟输入以获取位置编码
    dummy_input = Tensor(np.zeros((1, 10, 16)), device='cpu')
    encoded = pos_encoder(dummy_input)

    # 提取位置编码部分
    pos_encodings = pos_encoder.pe_tensor.data[:10, :8]  # 前10个位置，前8维

    print("位置编码矩阵 (前10个位置，前8维):")
    print("位置\\维度  ", end="")
    for d in range(8):
        print(f"{d:>7}", end="")
    print()

    for pos in range(10):
        print(f"pos_{pos:<2}   ", end="")
        for d in range(8):
            print(f"{pos_encodings[pos, d]:>7.3f}", end="")
        print()
    print()

    # 分析位置编码的周期性
    print("位置编码的周期性分析:")
    for dim in [0, 1, 2, 3]:
        encoding_dim = pos_encodings[:, dim]
        print(f"维度 {dim}: {encoding_dim}")
    print()


def test_mask_mechanisms():
    """测试各种掩码机制"""
    print("=== 掩码机制测试 ===")

    # 1. 填充掩码测试
    print("1. 填充掩码 (Padding Mask):")
    seq_with_padding = np.array([
        [3, 4, 5, 0, 0],  # 后两个是填充
        [6, 7, 8, 9, 0]  # 最后一个是填充
    ])
    seq_tensor = Tensor(seq_with_padding, device='cpu')

    model = Transformer(src_vocab_size=10, tgt_vocab_size=10, device='cpu')
    pad_mask = model.create_pad_mask(seq_tensor, pad_idx=0)

    print(f"序列: {seq_with_padding}")
    print(f"填充掩码形状: {pad_mask.shape}")
    print("填充掩码 (True表示需要掩盖):")
    print(pad_mask.data[0, 0, 0])  # 第一个样本的掩码
    print(pad_mask.data[1, 0, 0])  # 第二个样本的掩码
    print()

    # 2. 因果掩码测试
    print("2. 因果掩码 (Causal Mask):")
    causal_mask = model.create_causal_mask(5, device='cpu')
    print("因果掩码 (True表示需要掩盖):")
    print(causal_mask.data.astype(int))
    print()

    # 3. 组合掩码测试
    print("3. 组合掩码 (填充 + 因果):")
    tgt_seq = np.array([[1, 3, 4, 0, 0]])  # 目标序列带填充
    tgt_tensor = Tensor(tgt_seq, device='cpu')

    tgt_pad_mask = model.create_pad_mask(tgt_tensor, pad_idx=0)
    tgt_causal_mask = model.create_causal_mask(5, device='cpu')

    print(f"目标序列: {tgt_seq[0]}")
    print("目标填充掩码:")
    print(tgt_pad_mask.data[0, 0, 0])
    print("因果掩码:")
    print(tgt_causal_mask.data.astype(int))

    # 组合掩码 (任一为True则掩盖)
    # combined_mask = tgt_pad_mask.data[0, 0, 0] | tgt_causal_mask.data
    # print("组合掩码:")
    # print(combined_mask.astype(int))
    print()


# 主执行函数
if __name__ == "__main__":
    print("Transformer 编码器-解码器完整实现")
    print("=" * 50)

    # 1. 完整流程测试
    results = test_transformer_complete_pipeline()

    # 2. 翻译任务演示
    demo_translation_task()

    # 3. 注意力分析
    analyze_attention_patterns()

    # 4. 位置编码可视化
    visualize_positional_encoding()

    # 5. 掩码机制测试
    test_mask_mechanisms()

    # 6. 模型规模基准测试
    benchmark_model_sizes()
