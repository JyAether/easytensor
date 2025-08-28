from core.nn.tensor_nn import Module
from core.tensor import Tensor, zeros
import numpy as np


class Embedding(Module):
    """嵌入层"""

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # 初始化嵌入矩阵
        self.weight = Tensor(
            np.random.normal(0, 1, (num_embeddings, embedding_dim)),
            requires_grad=True
        )

        # 如果指定了padding_idx，将对应行初始化为0且不参与梯度计算
        if padding_idx is not None:
            self.weight.data[padding_idx] = 0

    def forward(self, input_ids):
        """前向传播"""
        # input_ids shape: (batch_size, seq_len)
        # 返回 shape: (batch_size, seq_len, embedding_dim)

        batch_size, seq_len = input_ids.shape
        embedded = []

        for i in range(batch_size):
            sequence_embeddings = []
            for j in range(seq_len):
                idx = int(input_ids.data[i, j])
                sequence_embeddings.append(self.weight.data[idx])
            embedded.append(sequence_embeddings)

        embedded_array = np.array(embedded)

        # 创建结果张量，正确设置计算图信息
        result = Tensor(
            embedded_array,
            requires_grad=self.weight.requires_grad,
            device=input_ids.device,
            _children=(self.weight, input_ids),  # 依赖于权重和输入
            _op='embedding'
        )

        # 设置反向传播函数
        if self.weight.requires_grad:
            def _embedding_backward():
                if result.grad is None:
                    return

                # 初始化权重梯度（如果需要）
                if self.weight.grad is None:
                    self.weight.grad = zeros(*self.weight.shape, device=self.weight.device)

                # 嵌入层的反向传播：将梯度累积到对应的嵌入向量上
                for i in range(batch_size):
                    for j in range(seq_len):
                        idx = int(input_ids.data[i, j])
                        # 跳过padding索引
                        if self.padding_idx is None or idx != self.padding_idx:
                            self.weight.grad.data[idx] += result.grad.data[i, j]

            result._backward = _embedding_backward

        return result

    def __repr__(self):
        return f"Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, padding_idx={self.padding_idx})"
