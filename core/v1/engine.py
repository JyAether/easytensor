import numpy as np


class Node:

    def __init__(self, data, _children=(), _op=''):
        # 存储数值
        self.data = data
        # 存储梯度
        self.grad = 0
        # 反向传播函数
        self._backward = lambda: None
        # 父节点集合，用于构建计算图
        self._prev = set(_children)
        # 操作类型，用于调试和可视化
        self._op = _op

    def __add__(self, other):
        # 确保other也是Node对象
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Node(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        out = Node(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        # 防止数值溢出
        # x = max(-500, min(500, x))
        t = (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
        out = Node(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward

        return out

    def sigmoid(self):
        x = self.data
        # 防止数值溢出
        # x = max(-500, min(500, x))
        s = 1 / (1 + np.exp(-x))
        out = Node(s, (self,), 'sigmoid')

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward

        return out

    def backward(self):

        # 第一步：拓扑图排序，确保按正确顺序处理节点
        topo = []
        visited = set()

        def build_topological_order(v):
            if v not in visited:
                visited.add(v)
                # 先处理所有子节点
                for child in v._prev:
                    build_topological_order(child)
                # 再将当前节点加入拓扑序列
                topo.append(v)

        build_topological_order(self)

        # 添加调试信息：打印topo的长度
        # print(f"拓扑排序节点数: {len(topo)}")

        # 第二步：初始化输出节点的梯度为1
        self.grad = 1
        # 第三步：按拓扑图逆序进行反向传播
        for v in reversed(topo):
            v._backward()

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other ** -1

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1

    def __repr__(self):
        return f"Node(data={self.data}, grad={self.grad})"
    
    def exp(self):
        out = Node(np.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += np.exp(self.data) * out.grad

        out._backward = _backward
        return out

    def log(self):
        assert self.data > 0, f"log函数要求输入大于0，但得到 {self.data}"
        out = Node(np.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def sqrt(self):
        assert self.data >= 0, f"sqrt函数要求输入大于等于0，但得到 {self.data}"
        out = Node(np.sqrt(self.data), (self,), 'sqrt')

        def _backward():
            if self.data == 0:
                self.grad += 0
            else:
                self.grad += (0.5 / np.sqrt(self.data)) * out.grad

        out._backward = _backward
        return out

    def abs(self):
        out = Node(abs(self.data), (self,), 'abs')

        def _backward():
            if self.data > 0:
                self.grad += out.grad
            elif self.data < 0:
                self.grad += -out.grad

        out._backward = _backward
        return out

    def max_with(self, other):
        other = other if isinstance(other, Node) else Node(other)
        if self.data >= other.data:
            out = Node(self.data, (self, other), 'max')

            def _backward():
                self.grad += out.grad
        else:
            out = Node(other.data, (self, other), 'max')

            def _backward():
                other.grad += out.grad
        out._backward = _backward
        return out

    def clamp(self, min_val=None, max_val=None):
        clamped_data = self.data
        if min_val is not None:
            clamped_data = max(clamped_data, min_val)
        if max_val is not None:
            clamped_data = min(clamped_data, max_val)

        out = Node(clamped_data, (self,), 'clamp')

        def _backward():
            grad_passes = True
            if min_val is not None and self.data < min_val:
                grad_passes = False
            if max_val is not None and self.data > max_val:
                grad_passes = False
            if grad_passes:
                self.grad += out.grad

        out._backward = _backward
        return out
