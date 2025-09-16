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

        # 拼接后的权重矩阵：(hidden_size, input_size + hidden_size)
        # 相当于将 weight_ih 和 weight_hh 水平拼接
        self.weight_combined = randn(hidden_size, input_size + hidden_size,
                                     requires_grad=True, device=device)

        if bias:
            # 拼接后的偏置：原来的 bias_ih + bias_hh
            self.bias_combined = zeros(hidden_size, requires_grad=True, device=device)
        else:
            self.bias_combined = None

        self.device = device
        self._init_weights()

    # def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device='cpu'):
    #     super().__init__()
    #     self.input_size = input_size
    #     self.hidden_size = hidden_size
    #     self.nonlinearity = nonlinearity
    #     # 拼接后的权重矩阵：(hidden_size, input_size + hidden_size)
    #     # 相当于将 weight_ih 和 weight_hh 水平拼接
    #     self.weight_combined = randn(hidden_size, input_size + hidden_size,
    #                                  requires_grad=True, device=device)
    #     # 输入到隐藏状态的权重
    #     self.weight_ih = randn(hidden_size, input_size, requires_grad=True, device=device)
    #     # 隐藏状态到隐藏状态的权重
    #     self.weight_hh = randn(hidden_size, hidden_size, requires_grad=True, device=device)
    #
    #     if bias:
    #         self.bias_ih = zeros(hidden_size, requires_grad=True, device=device)
    #         self.bias_hh = zeros(hidden_size, requires_grad=True, device=device)
    #     else:
    #         self.bias_ih = None
    #         self.bias_hh = None
    #
    #     self.device = device
    #     self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        std = 1.0 / (self.hidden_size) ** 0.5
        self.weight_combined.data = (np.random.uniform(-std, std, self.weight_combined.shape)).astype(np.float32)

    # def _init_weights(self):
    #     """权重初始化"""
    #     std = 1.0 / (self.hidden_size) ** 0.5
    #     for weight in [self.weight_ih, self.weight_hh]:
    #         weight.data = (np.random.uniform(-std, std, weight.shape)).astype(np.float32)

    def forward(self, input, hidden):
        """前向传播 - 拼接方式

        Args:
            input: 输入张量 (batch_size, input_size)
            hidden: 隐藏状态 (batch_size, hidden_size)

        Returns:
            new_hidden: 新的隐藏状态 (batch_size, hidden_size)
        """
        # 在特征维度上拼接输入和隐藏状态，因为拼接不能乱拼接，要计算好维度
        # input: (batch_size, input_size)
        # hidden: (batch_size, hidden_size)
        # combined: (batch_size, input_size + hidden_size)
        combined_input = cat([input, hidden], dim=1)

        # 一次矩阵乘法计算所有变换，这是核心
        # combined_input: (batch_size, input_size + hidden_size)
        # weight_combined.T: (input_size + hidden_size, hidden_size)
        # 结果: (batch_size, hidden_size)
        output = combined_input @ self.weight_combined.T

        # 添加偏置
        if self.bias_combined is not None:
            output = output + self.bias_combined

        # 应用激活函数
        if self.nonlinearity == 'tanh':
            new_hidden = output.tanh()
        elif self.nonlinearity == 'relu':
            new_hidden = output.relu()
        else:
            # 如果没有设置线性关系的话，rnn要抛出异常
            raise ValueError(f"Unknown nonlinearity: {self.nonlinearity}")

        return new_hidden

    # def forward(self, input, hidden):
    #     """前向传播，单向单层
    #
    #     Args:
    #         input: 输入张量 (batch_size, input_size)
    #         hidden: 隐藏状态 (batch_size, hidden_size)
    #
    #     Returns:
    #         new_hidden: 新的隐藏状态 (batch_size, hidden_size)
    #     """
    #     # 计算输入到隐藏状态的变换
    #     gi = input @ self.weight_ih.T
    #     if self.bias_ih is not None:
    #         gi = gi + self.bias_ih
    #
    #     # 计算隐藏状态到隐藏状态的变换
    #     gh = hidden @ self.weight_hh.T
    #     if self.bias_hh is not None:
    #         gh = gh + self.bias_hh
    #
    #     # 合并并应用激活函数
    #     i_h = gi + gh
    #
    #     if self.nonlinearity == 'tanh':
    #         new_hidden = i_h.tanh()
    #     elif self.nonlinearity == 'relu':
    #         new_hidden = i_h.relu()
    #     else:
    #         # 如果没有设置线性关系的话，rnn要抛出异常
    #         raise ValueError(f"Unknown nonlinearity: {self.nonlinearity}")
    #
    #     return new_hidden


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


# LSTM和GRU层实现和完整示例
import numpy as np

from core.nn.tensor_nn import Module, MSELoss, Dropout, Linear, CrossEntropyLoss
from core.tensor import randn, zeros, Tensor, cat


class LSTMCell(Module):
    """LSTM基本单元

    LSTM通过门控机制解决梯度消失问题：
    - 遗忘门(forget gate): 决定从细胞状态中丢弃什么信息
    - 输入门(input gate): 决定什么新信息被存储在细胞状态中
    - 输出门(output gate): 决定输出什么值

    公式:
    f_t = σ(W_f @ [h_{t-1}, x_t] + b_f)  # 遗忘门
    i_t = σ(W_i @ [h_{t-1}, x_t] + b_i)  # 输入门
    C̃_t = tanh(W_C @ [h_{t-1}, x_t] + b_C)  # 候选值
    C_t = f_t * C_{t-1} + i_t * C̃_t  # 细胞状态
    o_t = σ(W_o @ [h_{t-1}, x_t] + b_o)  # 输出门
    h_t = o_t * tanh(C_t)  # 隐藏状态
    """

    def __init__(self, input_size, hidden_size, bias=True, device='cpu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        # 输入到隐藏状态的权重 (4个门: forget, input, candidate, output)
        self.weight_ih = randn(4 * hidden_size, input_size, requires_grad=True, device=device)
        # 隐藏状态到隐藏状态的权重
        self.weight_hh = randn(4 * hidden_size, hidden_size, requires_grad=True, device=device)

        if bias:
            self.bias_ih = zeros(4 * hidden_size, requires_grad=True, device=device)
            self.bias_hh = zeros(4 * hidden_size, requires_grad=True, device=device)
        else:
            self.bias_ih = None
            self.bias_hh = None

        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        std = 1.0 / (self.hidden_size) ** 0.5
        for weight in [self.weight_ih, self.weight_hh]:
            weight.data = (np.random.uniform(-std, std, weight.shape)).astype(np.float32)

    def forward(self, input, hidden):
        """前向传播

        Args:
            input: 输入张量 (batch_size, input_size)
            hidden: 元组 (h_0, c_0)，其中：
                   h_0: 隐藏状态 (batch_size, hidden_size)
                   c_0: 细胞状态 (batch_size, hidden_size)

        Returns:
            (new_h, new_c): 新的隐藏状态和细胞状态
        """
        h_prev, c_prev = hidden

        # 计算所有门的线性变换
        gi = input @ self.weight_ih.T
        gh = h_prev @ self.weight_hh.T

        if self.bias_ih is not None:
            gi = gi + self.bias_ih
        if self.bias_hh is not None:
            gh = gh + self.bias_hh

        i_h = gi + gh
        """
        分割成4个门
        forget gate, input gate, candidate gate, output gate
        input (batch_size, input_size)
        hidden: 元组 (h_0, c_0)，其中：
                   h_0: 隐藏状态 (batch_size, hidden_size)
                   c_0: 细胞状态 (batch_size, hidden_size)
        """
        forget_gate = i_h[:, :self.hidden_size].sigmoid()
        input_gate = i_h[:, self.hidden_size:2 * self.hidden_size].sigmoid()
        """
        sigmoid：输入门决定"要不要"添加新信息（0-1的权重），
        tanh：候选信息 决定"添加什么"新信息（-1到1的具体内容）
        Examples : 当前输入：单词"坐"，历史状态：包含"那只猫"的语义信息，候选信息：结合"坐"和"那只猫"，生成"猫坐着"这个动作概念
              >>> 输入门：决定这个动作信息有多重要（比如0.8）
              >>>  最终更新：将80%的"猫坐着"信息添加到细胞状态中
        """
        candidate_gate = i_h[:, 2 * self.hidden_size:3 * self.hidden_size].tanh()
        """
        tanh：细胞状态在长时间累积后可能变得很大或很小，需要tanh让数值标准化
        sigmoid：作为"开关"控制细胞状态的哪些部分要输出
            0：完全不输出这部分信息
            1：完全输出这部分信息
            0.5：输出一半强度
        """
        output_gate = i_h[:, 3 * self.hidden_size:].sigmoid()

        # 更新细胞状态
        new_c = forget_gate * c_prev + input_gate * candidate_gate

        # 更新隐藏状态
        new_h = output_gate * new_c.tanh()

        return new_h, new_c


class LSTM(Module):
    """多层LSTM

    支持多层、双向、dropout等特性
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0., bidirectional=False, device='cpu'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.device = device

        # 创建LSTM单元
        self.lstm_cells = []

        # 第一层的输入大小
        layer_input_size = input_size

        for layer in range(num_layers):
            # 前向LSTM单元
            cell_fw = LSTMCell(layer_input_size, hidden_size, bias, device)
            self.lstm_cells.append(cell_fw)

            # 双向LSTM的后向单元
            if bidirectional:
                cell_bw = LSTMCell(layer_input_size, hidden_size, bias, device)
                self.lstm_cells.append(cell_bw)

            # 下一层的输入大小
            layer_input_size = hidden_size * (2 if bidirectional else 1)

        # Dropout层
        if dropout > 0:
            self.dropout_layer = Dropout(dropout)
        else:
            self.dropout_layer = None

    def forward(self, input, hidden=None):
        """前向传播

        Args:
            input: 输入序列
                - 如果 batch_first=True: (batch_size, seq_len, input_size)
                - 如果 batch_first=False: (seq_len, batch_size, input_size)
            hidden: 初始隐藏状态元组 (h_0, c_0)，其中：
                   h_0: (num_layers * num_directions, batch_size, hidden_size)
                   c_0: (num_layers * num_directions, batch_size, hidden_size)

        Returns:
            output: 输出序列
            (h_n, c_n): 最终隐藏状态和细胞状态
        """
        # 调整输入格式为 (seq_len, batch_size, input_size)
        if self.batch_first:
            input = input.transpose(1, 0)

        seq_len, batch_size, _ = input.shape

        # 初始化隐藏状态
        num_directions = 2 if self.bidirectional else 1
        if hidden is None:
            h_0 = zeros(self.num_layers * num_directions, batch_size, self.hidden_size, device=self.device)
            c_0 = zeros(self.num_layers * num_directions, batch_size, self.hidden_size, device=self.device)
        else:
            h_0, c_0 = hidden

        # 将隐藏状态按层和方向分组
        h_0_layers = []
        c_0_layers = []
        for layer in range(self.num_layers):
            if self.bidirectional:
                h_fw = h_0[layer * 2]
                h_bw = h_0[layer * 2 + 1]
                c_fw = c_0[layer * 2]
                c_bw = c_0[layer * 2 + 1]
                h_0_layers.append((h_fw, h_bw))
                c_0_layers.append((c_fw, c_bw))
            else:
                h_0_layers.append(h_0[layer])
                c_0_layers.append(c_0[layer])

        # 逐层处理
        layer_input = input
        final_h_states = []
        final_c_states = []

        for layer in range(self.num_layers):
            if self.bidirectional:
                # 双向处理
                cell_fw = self.lstm_cells[layer * 2]
                cell_bw = self.lstm_cells[layer * 2 + 1]
                h_fw, h_bw = h_0_layers[layer]
                c_fw, c_bw = c_0_layers[layer]

                # 前向传播
                fw_outputs = []
                hidden_fw, cell_fw_state = h_fw, c_fw
                for t in range(seq_len):
                    hidden_fw, cell_fw_state = cell_fw.forward(layer_input[t], (hidden_fw, cell_fw_state))
                    fw_outputs.append(hidden_fw)

                # 后向传播
                bw_outputs = []
                hidden_bw, cell_bw_state = h_bw, c_bw
                for t in range(seq_len - 1, -1, -1):
                    hidden_bw, cell_bw_state = cell_bw.forward(layer_input[t], (hidden_bw, cell_bw_state))
                    bw_outputs.insert(0, hidden_bw)

                # 合并前向和后向输出
                layer_outputs = []
                for t in range(seq_len):
                    combined = cat([fw_outputs[t], bw_outputs[t]], dim=1)
                    layer_outputs.append(combined)

                final_h_states.extend([hidden_fw, hidden_bw])
                final_c_states.extend([cell_fw_state, cell_bw_state])

            else:
                # 单向处理
                cell = self.lstm_cells[layer]
                hidden_state = h_0_layers[layer]
                cell_state = c_0_layers[layer]

                layer_outputs = []
                for t in range(seq_len):
                    hidden_state, cell_state = cell.forward(layer_input[t], (hidden_state, cell_state))
                    layer_outputs.append(hidden_state)

                final_h_states.append(hidden_state)
                final_c_states.append(cell_state)

            # 构建下一层的输入
            layer_input = Tensor.stack(layer_outputs, dim=0)

            # 应用dropout（除了最后一层）
            if self.dropout_layer is not None and layer < self.num_layers - 1:
                layer_input = self.dropout_layer(layer_input)

        # 最终输出
        output = layer_input
        h_n = Tensor.stack(final_h_states, dim=0)
        c_n = Tensor.stack(final_c_states, dim=0)

        # 调整输出格式
        if self.batch_first:
            output = output.transpose(1, 0)

        return output, (h_n, c_n)


class GRUCell(Module):
    """GRU基本单元

    GRU是LSTM的简化版本，将遗忘门和输入门合并为更新门：
    - 重置门(reset gate): 决定如何将新输入与前一记忆相结合
    - 更新门(update gate): 定义前一记忆保存到当前时间步的量

    公式:
    r_t = σ(W_r @ [h_{t-1}, x_t] + b_r)  # 重置门
    z_t = σ(W_z @ [h_{t-1}, x_t] + b_z)  # 更新门
    h̃_t = tanh(W_h @ [r_t * h_{t-1}, x_t] + b_h)  # 候选隐藏状态
    h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t  # 新隐藏状态

    ============================== 从LSTM推导 ==============================

    f_t = σ(W_f @ [h_{t-1}, x_t] + b_f)      # 遗忘门
    i_t = σ(W_i @ [h_{t-1}, x_t] + b_i)      # 输入门
    C̃_t = tanh(W_C @ [h_{t-1}, x_t] + b_C)   # 候选细胞状态
    C_t = f_t * C_{t-1} + i_t * C̃_t          # 细胞状态更新
    o_t = σ(W_o @ [h_{t-1}, x_t] + b_o)      # 输出门
    h_t = o_t * tanh(C_t)                    # 隐藏状态

    观察: LSTM中细胞状态C_t和隐藏状态h_t高度相关，我们可以尝试只保留一个状态。
    简化思路: 让 h_t ≈ C_t，即去掉输出门的非线性变换：
    h_t ≈ C_t = f_t * C_{t-1} + i_t * C̃_t
    由于现在 h_t ≈ C_t，我们可以写成：
    h_t = f_t * h_{t-1} + i_t * h̃_t
    其中 h̃_t 对应原来的候选细胞状态 C̃_t。
    问题: 现在我们有两个门 f_t 和 i_t，它们的作用是：
     f_t: 控制保留多少旧的隐藏状态
     i_t: 控制接受多少新的候选状态
    反向理解: 在很多情况下，当我们想要记住更多新信息时，就应该忘记更多旧信息，反之亦然。
    数学表达: 假设 i_t ≈ 1 - f_t，即：
    h_t = f_t * h_{t-1} + (1 - f_t) * h̃_t
    重新定义：令 z_t = 1 - f_t（称为更新门），则：
    h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
    GRU的核心更新公式！

    ====== 引入重置门======
    问题: 在计算候选隐藏状态 h̃_t 时，我们直接使用了 h_{t-1}：
    h̃_t = tanh(W_h @ [h_{t-1}, x_t] + b_h)
    改进思路: 有时我们希望在计算新的候选状态时，能够选择性地"重置"或"忽略"之前的隐藏状态的某些部分。
    引入重置门: 添加重置门 r_t 来控制前一时刻隐藏状态的影响：
    r_t = σ(W_r @ [h_{t-1}, x_t] + b_r)      # 重置门
    h̃_t = tanh(W_h @ [r_t * h_{t-1}, x_t] + b_h)  # 重置后的候选状态

    通过以上三个简化步骤，GRU的完整公式：
    r_t = σ(W_r @ [h_{t-1}, x_t] + b_r)      # 重置门
    z_t = σ(W_z @ [h_{t-1}, x_t] + b_z)      # 更新门
    h̃_t = tanh(W_h @ [r_t * h_{t-1}, x_t] + b_h)  # 候选隐藏状态
    h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t    # 最终隐藏状态

    遗忘门 f_t  =  更新门 z_t	（ 1 - f_t）（反向对应）

    ====== 参数对比======
    LSTM参数: 4个权重矩阵 + 4个偏置向量 （W_f，W_i，W_C，W_o，并与之对应的bias）
    GRU参数: 3个权重矩阵 + 3个偏置向量（W_r，W_z，W_C，W_h，并与之对应的bias）
    结论：减少比例: GRU比LSTM减少了约25%的参数量，训练更快，不容易过拟合。

    25%参数削减的代价在哪里？

    # LSTM: 独立控制
    C_t = f_t * C_{t-1} + i_t * C̃_t    # 写入记忆（独立的f_t和i_t）
    h_t = o_t * tanh(C_t)               # 读取记忆（独立的o_t）

    # GRU: 耦合控制
    h_t = (1-z_t) * h_{t-1} + z_t * h̃_t  # 读写耦合（z_t同时控制读写）
    代价: 无法独立控制"记住什么"和"忘记什么"，z_t = 1-f_t 强制了反比关系
    举例子：
    长期依赖场景
    # 问题场景：需要长期记住某些信息，但短期内频繁更新其他信息
    # LSTM的优势：
    f_t = [0.9, 0.1, 0.9]  # 长期记忆维度设高保留率
    i_t = [0.1, 0.8, 0.1]  # 短期更新维度设高输入率
    # 可以实现：维度0,2长期保持，维度1频繁更新

    # GRU的局限：
    z_t = [0.1, 0.8, 0.1]  # 更新门
    # 1-z_t = [0.9, 0.2, 0.9]  # 保持率被强制绑定
    # 无法独立控制某个维度的记忆保持和新信息接入
    GRU用25%的参数削减换取了训练效率，但牺牲了在复杂序列建模任务中的精细控制能力。这是一个典型的效率与表达能力的权衡。

    """

    def __init__(self, input_size, hidden_size, bias=True, device='cpu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        # 输入到隐藏状态的权重 (3个门: reset, update, new)
        self.weight_ih = randn(3 * hidden_size, input_size, requires_grad=True, device=device)
        # 隐藏状态到隐藏状态的权重
        self.weight_hh = randn(3 * hidden_size, hidden_size, requires_grad=True, device=device)

        if bias:
            self.bias_ih = zeros(3 * hidden_size, requires_grad=True, device=device)
            self.bias_hh = zeros(3 * hidden_size, requires_grad=True, device=device)
        else:
            self.bias_ih = None
            self.bias_hh = None

        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        std = 1.0 / (self.hidden_size) ** 0.5
        for weight in [self.weight_ih, self.weight_hh]:
            weight.data = (np.random.uniform(-std, std, weight.shape)).astype(np.float32)

    def forward(self, input, hidden):
        """前向传播

        Args:
            input: 输入张量 (batch_size, input_size)
            hidden: 隐藏状态 (batch_size, hidden_size)

        Returns:
            new_hidden: 新的隐藏状态 (batch_size, hidden_size)
        """
        # 计算输入到隐藏状态的变换
        gi = input @ self.weight_ih.T
        gh = hidden @ self.weight_hh.T

        if self.bias_ih is not None:
            gi = gi + self.bias_ih
        if self.bias_hh is not None:
            gh = gh + self.bias_hh

        # 分割重置门和更新门的部分
        i_reset, i_update, i_new = gi[:, :self.hidden_size], gi[:, self.hidden_size:2 * self.hidden_size], gi[:,
                                                                                                           2 * self.hidden_size:]
        h_reset, h_update, h_new = gh[:, :self.hidden_size], gh[:, self.hidden_size:2 * self.hidden_size], gh[:,
                                                                                                           2 * self.hidden_size:]

        # 计算重置门和更新门
        reset_gate = (i_reset + h_reset).sigmoid()
        update_gate = (i_update + h_update).sigmoid()

        # 计算候选隐藏状态
        new_gate = (i_new + reset_gate * h_new).tanh()

        # 计算新的隐藏状态
        new_hidden = (1 - update_gate) * hidden + update_gate * new_gate

        return new_hidden


class GRU(Module):
    """多层GRU

    支持多层、双向、dropout等特性
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0., bidirectional=False, device='cpu'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.device = device

        # 创建GRU单元
        self.gru_cells = []

        # 第一层的输入大小
        layer_input_size = input_size

        for layer in range(num_layers):
            # 前向GRU单元
            cell_fw = GRUCell(layer_input_size, hidden_size, bias, device)
            self.gru_cells.append(cell_fw)

            # 双向GRU的后向单元
            if bidirectional:
                cell_bw = GRUCell(layer_input_size, hidden_size, bias, device)
                self.gru_cells.append(cell_bw)

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
            input = input.transpose(1, 0)

        seq_len, batch_size, _ = input.shape

        # 初始化隐藏状态
        num_directions = 2 if self.bidirectional else 1
        if h_0 is None:
            h_0 = zeros(self.num_layers * num_directions, batch_size, self.hidden_size, device=self.device)

        # 将隐藏状态按层和方向分组
        h_0_layers = []
        for layer in range(self.num_layers):
            if self.bidirectional:
                h_fw = h_0[layer * 2]
                h_bw = h_0[layer * 2 + 1]
                h_0_layers.append((h_fw, h_bw))
            else:
                h_0_layers.append(h_0[layer])

        # 逐层处理
        layer_input = input
        final_hiddens = []

        for layer in range(self.num_layers):
            if self.bidirectional:
                # 双向处理
                cell_fw = self.gru_cells[layer * 2]
                cell_bw = self.gru_cells[layer * 2 + 1]
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
                    bw_outputs.insert(0, hidden_bw)

                # 合并前向和后向输出
                layer_outputs = []
                for t in range(seq_len):
                    combined = cat([fw_outputs[t], bw_outputs[t]], dim=1)
                    layer_outputs.append(combined)

                final_hiddens.extend([hidden_fw, hidden_bw])

            else:
                # 单向处理
                cell = self.gru_cells[layer]
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
            output = output.transpose(1, 0)

        return output, h_n


# ==================== LSTM和GRU使用示例 ====================

def lstm_example_1_basic():
    """示例1: 基础LSTM使用"""
    print("=== 示例1: 基础LSTM使用 ===")

    batch_size = 2
    seq_len = 5
    input_size = 10
    hidden_size = 20

    # 创建LSTM
    lstm = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

    # 创建输入数据
    x = randn(batch_size, seq_len, input_size, requires_grad=True)

    print(f"输入形状: {x.shape}，分别对应：[batch_size, seq_len, input_size]")

    # 前向传播
    output, (h_n, c_n) = lstm.forward(x)

    print(f"输出形状: {output.shape}，分别对应：[batch_size, seq_len, num_directions × hidden_size]")
    print(f"最终隐藏状态形状: {h_n.shape}，分别对应：[num_layers × num_directions, batch_size, hidden_size]")
    print(f"最终细胞状态形状: {c_n.shape}，分别对应：[num_layers × num_directions, batch_size, hidden_size]")

    # 计算损失并反向传播
    target = randn(batch_size, seq_len, hidden_size)
    loss = MSELoss()
    l = loss(output, target)

    print(f"损失值: {l.data}")

    # 反向传播
    l.backward()
    print("LSTM反向传播完成")
    print()


def gru_example_1_basic():
    """示例1: 基础GRU使用"""
    print("=== 示例1: 基础GRU使用 ===")

    batch_size = 2
    seq_len = 5
    input_size = 10
    hidden_size = 20

    # 创建GRU
    gru = GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

    # 创建输入数据
    x = randn(batch_size, seq_len, input_size, requires_grad=True)

    print(f"输入形状: {x.shape}，分别对应：[batch_size, seq_len, input_size]")

    # 前向传播
    output, h_n = gru.forward(x)

    print(f"输出形状: {output.shape}，分别对应：[batch_size, seq_len, num_directions × hidden_size]")
    print(f"最终隐藏状态形状: {h_n.shape}，分别对应：[num_layers × num_directions, batch_size, hidden_size]")

    # 计算损失并反向传播
    target = randn(batch_size, seq_len, hidden_size)
    loss = MSELoss()
    l = loss(output, target)

    print(f"损失值: {l.data}")

    # 反向传播
    l.backward()
    print("GRU反向传播完成")
    print()


def lstm_gru_comparison():
    """示例2: LSTM与GRU性能对比"""
    print("=== 示例2: LSTM与GRU性能对比 ===")

    batch_size = 3
    seq_len = 8
    input_size = 15
    hidden_size = 25
    num_layers = 2

    # 创建相同配置的LSTM和GRU
    lstm = LSTM(input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, dropout=0.1, batch_first=True)

    gru = GRU(input_size=input_size, hidden_size=hidden_size,
              num_layers=num_layers, dropout=0.1, batch_first=True)

    # 相同的输入数据
    x = randn(batch_size, seq_len, input_size, requires_grad=True)

    print(f"输入形状: {x.shape}")

    # LSTM前向传播
    lstm_output, (lstm_h, lstm_c) = lstm.forward(x)
    print(f"LSTM输出形状: {lstm_output.shape}")
    print(f"LSTM参数数量: {len(lstm.parameters())}")

    # GRU前向传播
    gru_output, gru_h = gru.forward(x)
    print(f"GRU输出形状: {gru_output.shape}")
    print(f"GRU参数数量: {len(gru.parameters())}")

    print("注意：GRU参数更少，计算更快，但LSTM在某些任务上可能表现更好")
    print()


def bidirectional_lstm_example():
    """示例3: 双向LSTM"""
    print("=== 示例3: 双向LSTM ===")

    batch_size = 2
    seq_len = 6
    input_size = 8
    hidden_size = 16

    # 创建双向LSTM
    bi_lstm = LSTM(input_size=input_size, hidden_size=hidden_size,
                   num_layers=1, bidirectional=True, batch_first=True)

    x = randn(batch_size, seq_len, input_size, requires_grad=True)

    print(f"输入形状: {x.shape}")

    output, (h_n, c_n) = bi_lstm.forward(x)

    print(f"双向LSTM输出形状: {output.shape}")  # 输出大小是 hidden_size * 2
    print(f"最终隐藏状态形状: {h_n.shape}")  # 有2个方向
    print(f"最终细胞状态形状: {c_n.shape}")
    print()


def lstm_text_classification():
    """示例4: 基于LSTM的文本分类"""
    print("=== 示例4: 基于LSTM的文本分类 ===")

    vocab_size = 1000
    embedding_dim = 50
    hidden_size = 64
    num_classes = 3
    seq_len = 20
    batch_size = 4

    class LSTMTextClassifier(Module):
        def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
            super().__init__()
            self.embedding_dim = embedding_dim
            self.lstm = LSTM(embedding_dim, hidden_size, batch_first=True)
            self.classifier = Linear(hidden_size, num_classes)

        def forward(self, x):
            # 简化的embedding（实际应该是embedding层）
            embedded = randn(x.shape[0], x.shape[1], self.embedding_dim)

            # LSTM处理序列
            lstm_output, (hidden, cell) = self.lstm.forward(embedded)

            # 使用最后一个时间步的输出进行分类
            last_output = lstm_output[:, -1, :]

            # 分类
            logits = self.classifier.forward(last_output)
            return logits

    # 创建模型
    model = LSTMTextClassifier(vocab_size, embedding_dim, hidden_size, num_classes)

    # 模拟输入数据
    input_ids = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))

    # 前向传播
    logits = model.forward(input_ids)

    print(f"输入形状: {input_ids.shape}")
    print(f"LSTM分类输出形状: {logits.shape}")

    # 模拟标签和损失计算
    labels = Tensor(np.random.randint(0, num_classes, (batch_size,)))
    loss_fn = CrossEntropyLoss()
    loss = loss_fn.forward(logits, labels)

    print(f"分类损失: {loss.data}")
    print()


def gru_sequence_generation():
    """示例5: 基于GRU的序列生成"""
    print("=== 示例5: 基于GRU的序列生成 ===")

    vocab_size = 100
    embedding_dim = 32
    hidden_size = 48
    seq_len = 10
    batch_size = 2

    class GRUSequenceGenerator(Module):
        def __init__(self, vocab_size, embedding_dim, hidden_size):
            super().__init__()
            self.embedding_dim = embedding_dim
            self.gru = GRU(embedding_dim, hidden_size, batch_first=True)
            self.output_projection = Linear(hidden_size, vocab_size)

        def forward(self, x):
            # 简化的embedding
            embedded = randn(x.shape[0], x.shape[1], self.embedding_dim)

            # GRU处理
            gru_output, hidden = self.gru.forward(embedded)

            # 投影到词汇表
            batch_size, seq_len, hidden_size = gru_output.shape
            gru_output_reshaped = gru_output.reshape(batch_size * seq_len, hidden_size)

            output_logits = self.output_projection.forward(gru_output_reshaped)
            output_logits = output_logits.reshape(batch_size, seq_len, vocab_size)

            return output_logits

    # 创建模型
    model = GRUSequenceGenerator(vocab_size, embedding_dim, hidden_size)

    # 创建输入数据
    input_ids = Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)))

    # 前向传播
    output_logits = model.forward(input_ids)

    print(f"输入形状: {input_ids.shape}")
    print(f"GRU生成输出形状: {output_logits.shape}")
    print(f"每个时间步输出词汇表大小: {vocab_size}")
    print()


def lstm_gru_architecture_explanation():
    """LSTM和GRU架构详细说明"""
    print("=== LSTM和GRU架构详细说明 ===")
    print("""
    RNN变体对比：

    1. 传统RNN的问题：
       - 梯度消失：长序列训练时梯度会指数级衰减
       - 梯度爆炸：梯度可能变得非常大
       - 长期依赖：难以学习长距离的依赖关系

    2. LSTM (Long Short-Term Memory)：
       核心思想：通过门控机制控制信息流

       组件：
       - 细胞状态(Cell State): C_t，长期记忆
       - 隐藏状态(Hidden State): h_t，短期记忆
       - 遗忘门(Forget Gate): 决定丢弃哪些信息
       - 输入门(Input Gate): 决定存储哪些新信息
       - 输出门(Output Gate): 决定输出哪些信息

       优点：
       - 有效解决梯度消失问题
       - 能够学习长期依赖
       - 在许多序列任务上表现优秀

       缺点：
       - 参数多，计算复杂
       - 训练时间较长

    3. GRU (Gated Recurrent Unit)：
       核心思想：简化LSTM，合并细胞状态和隐藏状态

       组件：
       - 重置门(Reset Gate): 控制前一时刻状态的影响
       - 更新门(Update Gate): 控制当前候选状态的影响
       - 候选状态: 当前时刻的候选隐藏状态

       优点：
       - 参数比LSTM少约25%
       - 训练速度更快
       - 在很多任务上性能接近LSTM

       缺点：
       - 在某些复杂任务上可能不如LSTM

    4. 选择建议：
       - 数据量小或计算资源有限：选择GRU
       - 复杂任务或有充足资源：选择LSTM
       - 需要最佳性能：两者都试试，选择更好的
       - 实时应用：GRU通常更快

    5. 参数量对比（以hidden_size=H, input_size=I为例）：
       - RNN: 参数量 ≈ H×(I+H+1)
       - LSTM: 参数量 ≈ 4×H×(I+H+1)  
       - GRU: 参数量 ≈ 3×H×(I+H+1)
    """)


# ==================== RNN使用示例 ====================

def rnn_example_1_basic():
    """示例1: 基础RNN使用"""
    print("=== 示例1: 基础RNN使用 ===")

    # 参数设置
    batch_size = 2
    seq_len = 5
    input_size = 10
    hidden_size = 20
    print(
        f"原始参数分别：batch_size: {batch_size}, seq_len: {seq_len}, input_size: {input_size}, hidden_size: {hidden_size}， num_layers ：1，")
    print('')
    # 创建RNN
    rnn = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

    # 创建输入数据 (batch_size, seq_len, input_size)
    x = randn(batch_size, seq_len, input_size, requires_grad=True)

    print(f"输入形状: {x.shape}，分别对应：[batch_size , seq_len , input_size]")

    # 前向传播
    output, h_n = rnn.forward(x)

    print(
        f"输出形状: {output.shape}，分别对应：[batch_size , seq_len , num_directions × hidden_size]")  # (batch_size, seq_len, hidden_size)
    print(
        f"最终隐藏状态形状: {h_n.shape}，分别对应：[num_layers × num_directions , batch_size ,hidden_size]")  # (num_layers, batch_size, hidden_size)

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
    print(
        f"原始参数分别：batch_size: {batch_size}, seq_len: {seq_len}, input_size: {input_size}, hidden_size: {hidden_size}， num_layers ：{num_layers}，")
    print('')
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
    print(
        f"原始参数分别：batch_size: {batch_size}, seq_len: {seq_len}, input_size: {input_size}, hidden_size: {hidden_size}， num_layers: 1, num_directions= 2")
    print('')
    # 创建双向RNN
    rnn = RNN(input_size=input_size,
              hidden_size=hidden_size,
              num_layers=1,
              bidirectional=True,
              batch_first=True)

    x = randn(batch_size, seq_len, input_size, requires_grad=True)

    print(f"输入形状: {x.shape}，分别对应：[batch_size , seq_len , input_size]")

    output, h_n = rnn.forward(x)

    print(
        f"输出形状: {output.shape}，分别对应：[batch_size , seq_len , num_directions × hidden_size]")  # (batch_size, seq_len, hidden_size * 2)
    print(
        f"最终隐藏状态形状: {h_n.shape}，分别对应：[num_layers × num_directions , batch_size ,hidden_size]")  # (num_layers * 2, batch_size, hidden_size)
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


# 添加详细的性能对比函数
def detailed_performance_comparison():
    """详细的性能对比实验"""
    print("=== 详细的RNN/LSTM/GRU性能对比实验 ===")

    batch_size = 4
    seq_len = 20
    input_size = 50
    hidden_size = 100

    # 创建三种模型
    rnn_model = RNN(input_size, hidden_size, batch_first=True)
    lstm_model = LSTM(input_size, hidden_size, batch_first=True)
    gru_model = GRU(input_size, hidden_size, batch_first=True)

    print("=== 详细参数统计 ===")

    print("\n1. RNN模型参数:")
    rnn_params = rnn_model.count_parameters()

    print("\n2. LSTM模型参数:")
    lstm_params = lstm_model.count_parameters()

    print("\n3. GRU模型参数:")
    gru_params = gru_model.count_parameters()

    print("\n=== 参数数量对比 ===")
    print(f"RNN总参数:   {rnn_params:,}")
    print(f"LSTM总参数:  {lstm_params:,}")
    print(f"GRU总参数:   {gru_params:,}")

    print(f"\nLSTM相对RNN参数比例: {lstm_params / rnn_params:.2f}x")
    print(f"GRU相对RNN参数比例:  {gru_params / rnn_params:.2f}x")
    print(f"LSTM相对GRU参数比例: {lstm_params / gru_params:.2f}x")

    # 理论参数计算
    print("\n=== 理论参数计算验证 ===")
    # RNN: W_ih(H×I) + W_hh(H×H) + b_ih(H) + b_hh(H) = H×(I+H+2)
    rnn_theory = hidden_size * (input_size + hidden_size + 2)

    # LSTM: 4个门，每个门有相同的参数结构
    lstm_theory = 4 * hidden_size * (input_size + hidden_size + 2)

    # GRU: 3个门
    gru_theory = 3 * hidden_size * (input_size + hidden_size + 2)

    print(f"RNN理论参数:  {rnn_theory:,}")
    print(f"LSTM理论参数: {lstm_theory:,}")
    print(f"GRU理论参数:  {gru_theory:,}")

    print(f"\n实际vs理论对比:")
    print(f"RNN:  实际 {rnn_params:,} vs 理论 {rnn_theory:,}")
    print(f"LSTM: 实际 {lstm_params:,} vs 理论 {lstm_theory:,}")
    print(f"GRU:  实际 {gru_params:,} vs 理论 {gru_theory:,}")


def parameter_breakdown_example():
    """参数分解示例"""
    print("\n=== 参数分解详细分析 ===")

    input_size = 10
    hidden_size = 20

    print(f"配置: input_size={input_size}, hidden_size={hidden_size}")

    # 创建单层模型进行详细分析
    print("\n1. RNN参数分解:")
    rnn = RNN(input_size, hidden_size, num_layers=1, batch_first=True)
    rnn.parameter_summary()

    print("\n2. LSTM参数分解:")
    lstm = LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
    lstm.parameter_summary()

    print("\n3. GRU参数分解:")
    gru = GRU(input_size, hidden_size, num_layers=1, batch_first=True)
    gru.parameter_summary()

def computational_complexity_analysis():
    """计算复杂度分析"""
    print("\n=== 计算复杂度分析 ===")
    print("""
    时间复杂度分析（每个时间步）:

    设 I = input_size, H = hidden_size, B = batch_size

    1. RNN:
       - 矩阵乘法: O(B × I × H) + O(B × H × H)
       - 总复杂度: O(B × H × (I + H))

    2. LSTM:
       - 4个门的计算: 4 × O(B × H × (I + H))
       - 总复杂度: O(4 × B × H × (I + H))

    3. GRU:
       - 3个门的计算: 3 × O(B × H × (I + H))
       - 总复杂度: O(3 × B × H × (I + H))

    相对复杂度:
    - LSTM ≈ 4x RNN
    - GRU ≈ 3x RNN
    - LSTM ≈ 1.33x GRU

    空间复杂度:
    - RNN: 存储 h_t
    - LSTM: 存储 h_t + c_t (双倍隐藏状态)
    - GRU: 存储 h_t (与RNN相同)
    """)


# 运行所有示例
if __name__ == "__main__":
    rnn_data_flow_explanation()
    rnn_example_1_basic()
    rnn_example_2_multilayer()
    rnn_example_3_bidirectional()
    rnn_example_4_sequence_classification()
    rnn_example_5_sequence_to_sequence()

    lstm_gru_architecture_explanation()
    lstm_example_1_basic()
    gru_example_1_basic()
    lstm_gru_comparison()
    bidirectional_lstm_example()
    lstm_text_classification()
    gru_sequence_generation()

    detailed_performance_comparison()
    parameter_breakdown_example()
    computational_complexity_analysis()
