from core.nn.tensor_nn import Module
from core.tensor import Tensor


class Embedding(Module):
    """嵌入层"""

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # 初始化嵌入矩阵
        import numpy as np
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

        # 简单的索引操作来获取嵌入
        batch_size, seq_len = input_ids.shape
        embedded = []

        for i in range(batch_size):
            sequence_embeddings = []
            for j in range(seq_len):
                idx = int(input_ids.data[i, j])
                sequence_embeddings.append(self.weight.data[idx])
            embedded.append(sequence_embeddings)

        import numpy as np
        embedded_array = np.array(embedded)
        result = Tensor(embedded_array, device=input_ids.device)

        # 设置梯度计算
        if self.weight.requires_grad:
            result.requires_grad = True

            def backward_fn(grad):
                # 嵌入层的反向传播需要将梯度累积到对应的嵌入向量上
                weight_grad = np.zeros_like(self.weight.data)

                for i in range(batch_size):
                    for j in range(seq_len):
                        idx = int(input_ids.data[i, j])
                        if self.padding_idx is None or idx != self.padding_idx:
                            weight_grad[idx] += grad.data[i, j]

                return weight_grad

            result._backward_fn = backward_fn
            result._parents = [self.weight]

        return result

    def __repr__(self):
        return f"Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, padding_idx={self.padding_idx})"
