"""
可视化拓扑图结构的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.v1.nn import MLP
from core.v1.engine import Node
from my_custom_dataset import X_train, y_train


def visualize_topo_structure():
    """可视化拓扑图结构"""
    print("=== 拓扑图结构可视化 ===")
    
    # 创建简单的网络
    net = MLP(2, [2, 1])  # 简化网络：2输入 -> 2隐藏 -> 1输出
    
    print(f"网络结构: {net}")
    print(f"参数总数: {len(net.parameters())}")
    
    # 创建单个样本的计算图
    x = [Node(X_train[0, 0]), Node(X_train[0, 1])]
    y_true = Node(y_train[0])
    
    print(f"\n输入节点:")
    for i, xi in enumerate(x):
        print(f"  x[{i}] = Node({xi.data:.4f})")
    
    print(f"\n真实值节点:")
    print(f"  y_true = Node({y_true.data:.4f})")
    
    # 前向传播
    y_pred = net(x)
    
    print(f"\n预测值节点:")
    print(f"  y_pred = Node({y_pred.data:.4f})")
    
    # 计算损失
    diff = y_pred - y_true
    loss = diff * diff
    
    print(f"\n损失计算节点:")
    print(f"  diff = y_pred - y_true = Node({diff.data:.4f})")
    print(f"  loss = diff * diff = Node({loss.data:.4f})")
    
    # 分析计算图的依赖关系
    print(f"\n=== 计算图依赖关系分析 ===")
    
    def analyze_dependencies(node, depth=0):
        """递归分析节点的依赖关系"""
        indent = "  " * depth
        print(f"{indent}节点: Node({node.data:.4f}) [操作: {node._op}]")
        
        if hasattr(node, '_prev') and node._prev:
            print(f"{indent}依赖的父节点:")
            for parent in node._prev:
                analyze_dependencies(parent, depth + 1)
        else:
            print(f"{indent}无依赖（叶子节点）")
    
    print("从损失节点开始的依赖链:")
    analyze_dependencies(loss)
    
    # 分析网络参数
    print(f"\n=== 网络参数分析 ===")
    params = net.parameters()
    for i, param in enumerate(params):
        print(f"参数[{i}]: Node({param.data:.4f}) [操作: {param._op}]")
    
    return loss


def analyze_topo_order():
    """分析拓扑排序的顺序"""
    print(f"\n=== 拓扑排序顺序分析 ===")
    
    # 创建损失计算
    loss = visualize_topo_structure()
    
    print(f"\n运行反向传播，观察拓扑排序...")
    
    # 反向传播 - 这会触发拓扑排序
    loss.backward()
    
    print("反向传播完成！")


if __name__ == "__main__":
    analyze_topo_order()

