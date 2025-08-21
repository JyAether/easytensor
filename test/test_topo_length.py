#!/usr/bin/env python3
"""
测试topo长度的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.v1.nn import MLP
from core.v1.engine import Node
from my_custom_dataset import X_train, y_train


def test_topo_length():
    """测试topo的长度"""
    print("=== 测试Topo长度 ===")
    
    # 创建网络
    net = MLP(2, [8, 8, 1])
    
    # 创建单个样本的计算图
    x = [Node(X_train[0, 0]), Node(X_train[0, 1])]
    y_true = Node(y_train[0])
    
    # 前向传播
    y_pred = net(x)
    
    # 计算损失
    diff = y_pred - y_true
    loss = diff * diff
    
    print("运行反向传播，观察topo长度...")
    
    # 反向传播 - 这会触发topo长度的打印
    loss.backward()
    
    print("反向传播完成！")


if __name__ == "__main__":
    test_topo_length()

