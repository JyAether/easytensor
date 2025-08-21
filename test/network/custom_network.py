from core.v1.nn import Neuron, Module


class CustomNetwork(Module):
    def __init__(self, hidden_activation='sigmoid', output_activation='sigmoid'):
        """构建2-2-2网络"""
        # 隐藏层 - 2个神经元
        # h1: 权重[w1=0.15, w2=0.20], 偏置b1=0.35
        # h2: 权重[w3=0.25, w4=0.30], 偏置b2=0.60
        self.hidden_layer = [
            Neuron(2, activation=hidden_activation, weights=[0.15, 0.20], bias=0.35),  # h1
            Neuron(2, activation=hidden_activation, weights=[0.25, 0.30], bias=0.60)  # h2
        ]

        # 输出层 - 2个神经元
        # o1: 权重[w5=0.40, w6=0.45], 无偏置
        # o2: 权重[w7=0.50, w8=0.55], 无偏置
        self.output_layer = [
            Neuron(2, activation=output_activation, weights=[0.40, 0.45], bias=0.0),  # o1
            Neuron(2, activation=output_activation, weights=[0.50, 0.55], bias=0.0)  # o2
        ]

    def __call__(self, x):
        # 隐藏层前向传播
        hidden_out = [neuron(x) for neuron in self.hidden_layer]
        # 输出层前向传播
        output = [neuron(hidden_out) for neuron in self.output_layer]
        return output

    def parameters(self):
        params = []
        for neuron in self.hidden_layer:
            params.extend(neuron.parameters())
        for neuron in self.output_layer:
            params.extend(neuron.parameters())
        return params

    def print_weights(self):
        print("=== 网络权重 ===")
        print("隐藏层:")
        print( f"  h1 ({self.hidden_layer[0].activation}): w1={self.hidden_layer[0].w[0].data:.4f}, w2={self.hidden_layer[0].w[1].data:.4f}, b1={self.hidden_layer[0].b.data:.4f}")
        print(f"  h2 ({self.hidden_layer[1].activation}): w3={self.hidden_layer[1].w[0].data:.4f}, w4={self.hidden_layer[1].w[1].data:.4f}, b2={self.hidden_layer[1].b.data:.4f}")
        print("输出层:")
        print(f"  o1 ({self.output_layer[0].activation}): w5={self.output_layer[0].w[0].data:.4f}, w6={self.output_layer[0].w[1].data:.4f}")
        print(f"  o2 ({self.output_layer[1].activation}): w7={self.output_layer[1].w[0].data:.4f}, w8={self.output_layer[1].w[1].data:.4f}")
