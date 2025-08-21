#!/usr/bin/env python3
"""
测试完整训练过程的topo长度
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.v1.nn import MLP
from core.v1.engine import Node
from my_custom_dataset import X_train, y_train


def test_full_training_topo():
    """测试完整训练过程的topo长度"""
    print("=== 测试完整训练过程的Topo长度 ===")
    
    # 创建网络
    net = MLP(2, [8, 8, 1])
    
    print(f"网络参数总数: {len(net.parameters())}")
    print(f"训练样本数: {len(X_train)}")
    
    # 创建完整的损失计算
    total_loss = Node(0.0)
    
    print("\n构建完整计算图...")
    
    # 对每个训练样本计算损失
    for i in range(len(X_train)):
        x = [Node(X_train[i, 0]), Node(X_train[i, 1])]
        y_true = Node(y_train[i])
        
        y_pred = net(x)
        diff = y_pred - y_true
        loss = diff * diff
        total_loss = total_loss + loss
        
        if i % 20 == 0:
            print(f"处理样本 {i+1}/{len(X_train)}")
    
    # 平均损失
    avg_loss = total_loss * Node(1.0 / len(X_train))
    
    print("\n计算图构建完成！")
    print("运行反向传播，观察完整topo长度...")
    
    # 反向传播 - 这会触发topo长度的打印
    avg_loss.backward()
    
    print("反向传播完成！")


if __name__ == "__main__":
    test_full_training_topo()

