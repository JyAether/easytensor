# RNN层实现和完整示例
import numpy as np

from core.nn.tensor_nn import Module, MSELoss, Dropout, Linear, CrossEntropyLoss
from core.tensor import randn, zeros, Tensor, cat


class RNNCell(Module):
    """RNN基本单元

    h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
    """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device='cpu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity

        # 输入到隐藏状态的权重
        self.weight_ih = randn(hidden_size, input_size, requires_grad=True, device=device)
        # 隐藏状态到隐藏状态的权重
        self.weight_hh = randn(hidden_size, hidden_size, requires_grad=True, device=device)

        if bias:
            self.bias_ih = zeros(hidden_size, requires_grad=True, device=device)
            self.bias_hh = zeros(hidden_size, requires_grad=True, device=device)
        else:
            self.bias_ih = None
            self.bias_hh = None

        self.device = device
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        std = 1.0 / (self.hidden_size) ** 0.5
        for weight in [self.weight_ih, self.weight_hh]:
            weight.data = (np.random.uniform(-std, std, weight.shape)).astype(np.float32)

    def forward(self, input, hidden):
        """前向传播，单向单层

        Args:
            input: 输入张量 (batch_size, input_size)
            hidden: 隐藏状态 (batch_size, hidden_size)

        Returns:
            new_hidden: 新的隐藏状态 (batch_size, hidden_size)
        """
        # 计算输入到隐藏状态的变换
        gi = input @ self.weight_ih.T
        if self.bias_ih is not None:
            gi = gi + self.bias_ih

        # 计算隐藏状态到隐藏状态的变换
        gh = hidden @ self.weight_hh.T
        if self.bias_hh is not None:
            gh = gh + self.bias_hh

        # 合并并应用激活函数
        i_h = gi + gh

        if self.nonlinearity == 'tanh':
            new_hidden = i_h.tanh()
        elif self.nonlinearity == 'relu':
            new_hidden = i_h.relu()
        else:
            # 如果没有设置线性关系的话，rnn要抛出异常
            raise ValueError(f"Unknown nonlinearity: {self.nonlinearity}")

        return new_hidden


class RNN(Module):
    """多层RNN

    支持多层、双向、dropout等特性
    """

    def __init__(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh',
                 bias=True, batch_first=False, dropout=0., bidirectional=False, device='cpu'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.device = device

        # 创建RNN单元
        self.rnn_cells = []

        # 第一层的输入大小
        layer_input_size = input_size

        for layer in range(num_layers):
            # 前向RNN单元
            cell_fw = RNNCell(layer_input_size, hidden_size, bias, nonlinearity, device)
            self.rnn_cells.append(cell_fw)

            # 双向RNN的后向单元
            if bidirectional:
                cell_bw = RNNCell(layer_input_size, hidden_size, bias, nonlinearity, device)
                self.rnn_cells.append(cell_bw)

            # 下一层的输入大小
            layer_input_size = hidden_size * (2 if bidirectional else 1)

        # Dropout层
        if dropout > 0:
            self.dropout_layer = Dropout(dropout)
        else:
            self.dropout_layer = None

    def forward(self, input, h_0=None):
        """前向传播

        Args:
            input: 输入序列
                - 如果 batch_first=True: (batch_size, seq_len, input_size)
                - 如果 batch_first=False: (seq_len, batch_size, input_size)
            h_0: 初始隐藏状态 (num_layers * num_directions, batch_size, hidden_size)

        Returns:
            output: 输出序列
            h_n: 最终隐藏状态
        """
        # 调整输入格式为 (seq_len, batch_size, input_size)
        if self.batch_first:
            input = input.transpose(1, 0)  # (batch_size, seq_len, input_size) -> (seq_len, batch_size, input_size)

        seq_len, batch_size, _ = input.shape

        # 初始化隐藏状态
        num_directions = 2 if self.bidirectional else 1
        if h_0 is None:
            h_0 = zeros(self.num_layers * num_directions, batch_size, self.hidden_size, device=self.device)

        # 将隐藏状态按层和方向分组
        h_0_layers = []
        for layer in range(self.num_layers):
            if self.bidirectional:
                h_fw = h_0[layer * 2]  # 前向隐藏状态
                h_bw = h_0[layer * 2 + 1]  # 后向隐藏状态
                h_0_layers.append((h_fw, h_bw))
            else:
                h_0_layers.append(h_0[layer])

        # 逐层处理
        layer_input = input
        final_hiddens = []

        for layer in range(self.num_layers):
            if self.bidirectional:
                # 双向处理
                cell_fw = self.rnn_cells[layer * 2]
                cell_bw = self.rnn_cells[layer * 2 + 1]
                h_fw, h_bw = h_0_layers[layer]

                # 前向传播
                fw_outputs = []
                hidden_fw = h_fw
                for t in range(seq_len):
                    hidden_fw = cell_fw.forward(layer_input[t], hidden_fw)
                    fw_outputs.append(hidden_fw)

                # 后向传播
                bw_outputs = []
                hidden_bw = h_bw
                for t in range(seq_len - 1, -1, -1):
                    hidden_bw = cell_bw.forward(layer_input[t], hidden_bw)
                    bw_outputs.insert(0, hidden_bw)  # 插入到开头保持时间顺序

                # 合并前向和后向输出
                layer_outputs = []
                for t in range(seq_len):
                    # 在特征维度上拼接
                    combined = cat([fw_outputs[t], bw_outputs[t]], dim=1)
                    layer_outputs.append(combined)

                final_hiddens.extend([hidden_fw, hidden_bw])

            else:
                # 单向处理
                cell = self.rnn_cells[layer]
                hidden = h_0_layers[layer]

                layer_outputs = []
                for t in range(seq_len):
                    hidden = cell.forward(layer_input[t], hidden)
                    layer_outputs.append(hidden)

                final_hiddens.append(hidden)

            # 构建下一层的输入
            layer_input = Tensor.stack(layer_outputs, dim=0)

            # 应用dropout（除了最后一层）
            if self.dropout_layer is not None and layer < self.num_layers - 1:
                layer_input = self.dropout_layer(layer_input)

        # 最终输出
        output = layer_input
        h_n = Tensor.stack(final_hiddens, dim=0)

        # 调整输出格式
        if self.batch_first:
            output = output.transpose(1, 0)  # (seq_len, batch_size, hidden_size) -> (batch_size, seq_len, hidden_size)

        return output, h_n


# ==================== RNN使用示例 ====================

def rnn_example_1_basic():
    """示例1: 基础RNN使用"""
    print("=== 示例1: 基础RNN使用 ===")

    # 参数设置
    batch_size = 2
    seq_len = 5
    input_size = 10
    hidden_size = 20

    # 创建RNN
    rnn = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

    # 创建输入数据 (batch_size, seq_len, input_size)
    x = randn(batch_size, seq_len, input_size, requires_grad=True)

    print(f"输入形状: {x.shape}，分别对应：[batch_size , seq_len , input_size]")

    # 前向传播
    output, h_n = rnn.forward(x)

    print(f"输出形状: {output.shape}，分别对应：[batch_size , seq_len , num_directions × hidden_size]")  # (batch_size, seq_len, hidden_size)
    print(f"最终隐藏状态形状: {h_n.shape}，分别对应：[num_layers × num_directions , batch_size ,hidden_size]")  # (num_layers, batch_size, hidden_size)

    # 计算损失并反向传播
    target = randn(batch_size, seq_len, hidden_size)
    loss = MSELoss()
    l = loss(output, target)

    print(f"损失值: {l.data}")

    # 反向传播
    l.backward()
    print("反向传播完成")
    print()


def rnn_example_2_multilayer():
    """示例2: 多层RNN"""
    print("=== 示例2: 多层RNN ===")

    batch_size = 3
    seq_len = 8
    input_size = 15
    hidden_size = 25
    num_layers = 2

    # 创建多层RNN
    rnn = RNN(input_size=input_size,
              hidden_size=hidden_size,
              num_layers=num_layers,
              dropout=0.1,
              batch_first=True)

    x = randn(batch_size, seq_len, input_size, requires_grad=True)

    print(f"输入形状: {x.shape}，分别对应：[batch_size , seq_len , input_size]")

    # 初始化隐藏状态
    h_0 = randn(num_layers, batch_size, hidden_size)

    output, h_n = rnn.forward(x, h_0)

    print(f"输出形状: {output.shape}，分别对应：[batch_size , seq_len , num_directions × hidden_size]")
    print(f"最终隐藏状态形状: {h_n.shape}，分别对应：[num_layers × num_directions , batch_size ,hidden_size]")
    print()


def rnn_example_3_bidirectional():
    """示例3: 双向RNN"""
    print("=== 示例3: 双向RNN ===")

    batch_size = 2
    seq_len = 6
    input_size = 8
    hidden_size = 16

    # 创建双向RNN
    rnn = RNN(input_size=input_size,
              hidden_size=hidden_size,
              num_layers=1,
              bidirectional=True,
              batch_first=True)

    x = randn(batch_size, seq_len, input_size, requires_grad=True)

    print(f"输入形状: {x.shape}，分别对应：[batch_size , seq_len , input_size]")

    output, h_n = rnn.forward(x)

    print(f"输出形状: {output.shape}，分别对应：[batch_size , seq_len , num_directions × hidden_size]")  # (batch_size, seq_len, hidden_size * 2)
    print(f"最终隐藏状态形状: {h_n.shape}，分别对应：[num_layers × num_directions , batch_size ,hidden_size]")  # (num_layers * 2, batch_size, hidden_size)
    print()


def rnn_example_4_sequence_classification():
    """示例4: 序列分类任务"""
    print("=== 示例4: 序列分类任务 ===")

    # 模拟文本分类任务
    vocab_size = 1000
    embedding_dim = 50
    hidden_size = 64
    num_classes = 3
    seq_len = 20
    batch_size = 4

    # 简化的模型：Embedding + RNN + Classifier
    class SimpleRNNClassifier(Module):
        def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
            super().__init__()
            # 这里简化为直接使用随机embeddings
            self.embedding_dim = embedding_dim
            self.rnn = RNN(embedding_dim, hidden_size, batch_first=True)
            self.classifier = Linear(hidden_size, num_classes)

        def forward(self, x):
            # 简化的embedding查找（实际应该是embedding层）
            # 这里直接使用随机特征代替
            embedded = randn(x.shape[0], x.shape[1], self.embedding_dim)

            # RNN处理序列
            rnn_output, hidden = self.rnn.forward(embedded)

            # 使用最后一个时间步的输出进行分类
            last_output = rnn_output[:, -1, :]  # (batch_size, hidden_size)

            # 分类
            logits = self.classifier.forward(last_output)
            return logits

    # 创建模型
    model = SimpleRNNClassifier(vocab_size, embedding_dim, hidden_size, num_classes)

    # 模拟输入数据（词汇ID序列）
    input_ids = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))

    # 前向传播
    logits = model.forward(input_ids)

    print(f"输入形状: {input_ids.shape}")
    print(f"分类输出形状: {logits.shape}")

    # 模拟标签和损失计算
    labels = Tensor(np.random.randint(0, num_classes, (batch_size,)))
    loss_fn = CrossEntropyLoss()
    loss = loss_fn.forward(logits, labels)

    print(f"分类损失: {loss.data}")
    print()


def rnn_example_5_sequence_to_sequence():
    """示例5: 序列到序列任务（简化版本）"""
    print("=== 示例5: 序列到序列任务 ===")

    input_size = 10
    hidden_size = 20
    output_size = 15
    seq_len = 8
    batch_size = 2

    class Seq2SeqModel(Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.encoder = RNN(input_size, hidden_size, batch_first=True)
            self.decoder = RNN(output_size, hidden_size, batch_first=True)
            self.output_projection = Linear(hidden_size, output_size)

        def forward(self, encoder_input, decoder_input):
            # 编码器处理输入序列
            encoder_output, encoder_hidden = self.encoder.forward(encoder_input)

            # 解码器使用编码器的最终隐藏状态作为初始状态
            decoder_output, decoder_hidden = self.decoder.forward(decoder_input, encoder_hidden)

            # 投影到输出词汇表
            # 重新调整形状以便应用线性层
            batch_size, seq_len, hidden_size = decoder_output.shape
            decoder_output_reshaped = decoder_output.reshape(batch_size * seq_len, hidden_size)

            output_logits = self.output_projection.forward(decoder_output_reshaped)
            output_logits = output_logits.reshape(batch_size, seq_len, output_size)

            return output_logits

    # 创建模型
    model = Seq2SeqModel(input_size, hidden_size, output_size)

    # 创建输入数据
    encoder_input = randn(batch_size, seq_len, input_size, requires_grad=True)
    decoder_input = randn(batch_size, seq_len, output_size, requires_grad=True)

    # 前向传播
    output = model.forward(encoder_input, decoder_input)

    print(f"编码器输入形状: {encoder_input.shape}")
    print(f"解码器输入形状: {decoder_input.shape}")
    print(f"模型输出形状: {output.shape}")
    print()


def rnn_data_flow_explanation():
    """RNN数据流程详细说明"""
    print("=== RNN数据流程详细说明 ===")
    print("""
    RNN的核心思想是在处理序列数据时维持一个隐藏状态，这个隐藏状态会在时间步之间传递：

    1. 数据流程：
       输入序列: x_1, x_2, x_3, ..., x_T
       隐藏状态: h_0, h_1, h_2, ..., h_T
       输出序列: o_1, o_2, o_3, ..., o_T

    2. 每个时间步的计算：
       h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
       其中：
       - W_ih: 输入到隐藏状态的权重矩阵
       - W_hh: 隐藏状态到隐藏状态的权重矩阵  
       - b: 偏置项

    3. 关键特点：
       - 参数共享：所有时间步使用相同的权重
       - 记忆机制：隐藏状态h_{t-1}携带历史信息
       - 序列处理：逐步处理输入序列的每个元素

    4. 梯度反向传播：
       - 通过时间反向传播(BPTT)
       - 梯度需要沿时间步向前传播
       - 可能出现梯度消失/爆炸问题

    5. 实际应用中的数据形状：
       - 输入: (batch_size, seq_len, input_size) 或 (seq_len, batch_size, input_size)
       - 隐藏状态: (batch_size, hidden_size)
       - 输出: (batch_size, seq_len, hidden_size)
    """)


# 运行所有示例
if __name__ == "__main__":
    rnn_data_flow_explanation()
    rnn_example_1_basic()
    rnn_example_2_multilayer()
    rnn_example_3_bidirectional()
    rnn_example_4_sequence_classification()
    rnn_example_5_sequence_to_sequence()