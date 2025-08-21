import random
from core.v1.engine import Node
class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, activation='relu',weights=None, bias=None):
        """
        nin: 输入维度
        activation: 激活函数类型 - 'sigmoid', 'relu', 'tanh', 'linear'
        weights: 指定的权重列表，如果为None则随机初始化
        bias: 指定的偏置值，如果为None则初始化为0
        """
        if weights is not None:
            assert len(weights) == nin, f"权重数量 {len(weights)} 不匹配输入维度 {nin}"
            self.w = [Node(w) for w in weights]
        else:
            self.w = [Node(random.uniform(-1, 1)) for _ in range(nin)]

        if bias is not None:
            self.b = Node(bias)
        else:
            self.b = Node(0)

        # 支持的激活函数
        self.activation = activation
        assert activation in ['sigmoid', 'relu', 'tanh', 'linear'], \
            f"不支持的激活函数: {activation}. 支持的函数: sigmoid, relu, tanh, linear"


    def __call__(self, x):
            act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)

            if self.activation == 'sigmoid':
                return act.sigmoid()
            elif self.activation == 'relu':
                return act.relu()
            elif self.activation == 'tanh':
                return act.tanh()
            elif self.activation == 'linear':
                return act
            else:
                return act

    def parameters(self):
        """"
        返回神经元的所有参数

        返回: 权重列表 + 偏置 = [w1, w2, ..., wn, b]
        比如： [w1 :Node(data=0.8037110741937401, grad=0), w2: Node(data=0.40008580046912257, grad=0), b :Node(data=0, grad=0)]
        """
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, activation='relu', **kwargs):
        self.neurons = [Neuron(nin, activation=activation, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts, hidden_activation='relu', output_activation='linear'):
        sz = [nin] + nouts
        self.layers = []

        for i in range(len(nouts)):
            if i == len(nouts) - 1:  # 输出层
                activation = output_activation
            else:  # 隐藏层
                activation = hidden_activation

            self.layers.append(Layer(sz[i], sz[i + 1], activation=activation))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
