#!/usr/bin/env python3
"""
分析topo节点数量的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.v1.nn import MLP
from core.v1.engine import Node
from my_custom_dataset import X_train, y_train


def analyze_topo_nodes():
    """分析topo节点的数量和构成"""
    print("=== Topo节点分析 ===")
    
    # 创建网络：2输入 -> 8隐藏 -> 8隐藏 -> 1输出
    net = MLP(2, [8, 8, 1])
    
    # 分析网络结构
    print(f"网络结构: {net}")
    print(f"参数总数: {len(net.parameters())}")
    
    # 计算单个样本的计算图节点数
    print("\n=== 单个样本的计算图分析 ===")
    
    # 选择第一个样本
    x = [Node(X_train[0, 0]), Node(X_train[0, 1])]
    y_true = Node(y_train[0])
    
    # 前向传播
    y_pred = net(x)
    
    # 计算损失
    diff = y_pred - y_true
    loss = diff * diff
    
    print("输入节点数: 2")
    print(f"参数节点数: {len(net.parameters())}")
    
    # 分析计算节点
    # 第一层: 8个神经元，每个神经元: 2个乘法 + 1个加法 + 1个ReLU = 4个节点
    layer1_nodes = 8 * 4
    print(f"第一隐藏层计算节点: {layer1_nodes}")
    
    # 第二层: 8个神经元，每个神经元: 8个乘法 + 1个加法 + 1个ReLU = 10个节点  
    layer2_nodes = 8 * 10
    print(f"第二隐藏层计算节点: {layer2_nodes}")
    
    # 输出层: 1个神经元，每个神经元: 8个乘法 + 1个加法 = 9个节点
    output_nodes = 1 * 9
    print(f"输出层计算节点: {output_nodes}")
    
    # 损失计算节点: 1个减法 + 1个乘法 = 2个节点
    loss_nodes = 2
    print(f"损失计算节点: {loss_nodes}")
    
    # 单个样本总节点数
    single_sample_nodes = (2 + len(net.parameters()) + layer1_nodes + 
                          layer2_nodes + output_nodes + loss_nodes)
    print(f"单个样本总节点数: {single_sample_nodes}")
    
    # 分析完整训练的计算图
    print("\n=== 完整训练的计算图分析 ===")
    
    # 创建完整的损失计算
    total_loss = Node(0.0)
    
    # 对每个训练样本计算损失
    for i in range(len(X_train)):
        x = [Node(X_train[i, 0]), Node(X_train[i, 1])]
        y_true = Node(y_train[i])
        
        y_pred = net(x)
        diff = y_pred - y_true
        loss = diff * diff
        total_loss = total_loss + loss
    
    # 平均损失
    avg_loss = total_loss * Node(1.0 / len(X_train))
    
    print(f"训练样本数: {len(X_train)}")
    print("每个样本的损失计算节点: 2 (减法+乘法)")
    print(f"损失累加节点数: {len(X_train)} (每个样本一个加法)")
    print("平均损失计算节点: 1 (除法)")
    
    # 计算完整计算图的节点数
    # 注意：参数节点在多个样本间共享，所以只计算一次
    shared_nodes = (len(net.parameters()) + layer1_nodes + 
                   layer2_nodes + output_nodes)
    per_sample_nodes = 2 + 2 + 1  # 输入 + 损失计算 + 累加
    total_nodes = shared_nodes + len(X_train) * per_sample_nodes + 1
    
    print(f"\n共享节点数: {shared_nodes}")
    print(f"每样本新增节点数: {per_sample_nodes}")
    print(f"样本相关总节点数: {len(X_train) * per_sample_nodes}")
    print("最终计算节点数: 1")
    print(f"理论总节点数: {total_nodes}")
    
    # 实际运行backward()来获取真实的topo长度
    print("\n=== 实际运行验证 ===")
    
    # 重置梯度
    net.zero_grad()
    
    # 运行反向传播
    avg_loss.backward()
    
    print("\n注意：要获取准确的topo长度，需要在core/engine.py的backward()方法中添加:")
    print("print(f'拓扑排序节点数: {len(topo)}')")
    
    return total_nodes


if __name__ == "__main__":
    total_nodes = analyze_topo_nodes()
    print(f"\n估算的总节点数: {total_nodes}")
