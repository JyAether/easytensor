# brew install graphviz
# pip install graphviz
from graphviz import Digraph
import uuid

"""
假设我们有一个简单的计算图节点类：

class Node:
    def __init__(self, data, grad=0.0, _prev=(), _op=''):
        self.data = data      # 节点的数值
        self.grad = grad      # 梯度值
        self._prev = set(_prev)  # 输入节点集合
        self._op = _op        # 操作类型

# 使用示例：
a = Node(2.0)
b = Node(3.0)
c = Node(a.data + b.data, _prev=(a, b), _op='+')
d = Node(c.data * 2, _prev=(c,), _op='*')

# 可视化计算图
dot = draw_dot(d)
dot.view()  # 将显示: a,b -> + -> c -> * -> d
"""


def trace(root):
    """
    遍历计算图，收集所有节点和边

    参数:
        root: 计算图的根节点（通常是最终的输出节点）

    返回:
        nodes: set，包含计算图中所有节点的集合
        edges: set，包含所有边的集合，每条边是一个元组(parent, child)

    用法示例:
        nodes, edges = trace(loss_node)  # loss_node是损失函数的输出
    """
    nodes, edges = set(), set()

    def build(v):
        """
        递归构建函数，深度优先遍历计算图

        参数:
            v: 当前访问的节点

        功能:
            - 将节点添加到nodes集合中（避免重复访问）
            - 为每个父节点创建指向当前节点的边
            - 递归访问所有父节点
        """
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root, format='svg', rankdir='LR'):
    """
        使用Graphviz创建计算图的可视化

        参数:
            root: 计算图的根节点
            format: 输出格式，默认'svg'
                    - 'svg': 矢量图格式，适合网页显示
                    - 'png': 位图格式
                    - 'pdf': PDF格式
            rankdir: 图的布局方向，默认'LR'
                    - 'LR': Left to Right，从左到右布局
                    - 'TB': Top to Bottom，从上到下布局

        返回:
            dot: Graphviz的Digraph对象，可以调用.render()或.view()方法

        用法示例:
            # 创建SVG格式的横向布局图
            dot = draw_dot(output_node)
            dot.view()  # 显示图形

            # 创建PNG格式的纵向布局图
            dot = draw_dot(output_node, format='png', rankdir='TB')
            dot.render('my_graph')  # 保存为文件
        """

    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    unique_id = str(uuid.uuid4())
    # 创建有向图对象
    # graph_attr={'rankdir': rankdir} 设置图的全局属性
    dot = Digraph(name=unique_id, format=format, graph_attr={'rankdir': rankdir})

    for n in nodes:
        # 节点显示格式：{ data 数值 | grad 梯度值 }
        # shape='record' 创建带分割线的矩形节点
        dot.node(name=str(id(n)), label="{ data %.4f | grad %.4f }" % (n.data, n.grad), shape='record')
        # 如果节点有操作符，创建操作符节点
        if n._op:  # n._op 假设存储操作类型如 '+', '*', 'relu' 等
            dot.node(name=str(id(n)) + n._op, label=n._op)
            # 从操作符节点指向数据节点
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
