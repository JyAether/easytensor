from core.v1.nn import Module
from core.v1.engine import Node
import numpy.random as random


class Dropout(Module):
    """ Dropout正则化层 """

    def __init__(self, p):
        """
        p: dropout概率，训练时随机将p比例的神经元输出置为0
        """
        self.p = p
        self.training = True

    def __call__(self, x):
        """
        x: 输入数据，可以是单个样本或批次数据
        """
        if not self.training:
            # 评估模式：不进行dropout，直接返回输入
            # 注意：这里采用的是"inverted dropout"，训练时已经缩放过了
            return x

        # 处理单个样本的情况
        is_single_sample = isinstance(x[0], Node)
        if is_single_sample:
            x = [x]

        # 应用dropout
        dropped = []
        for sample in x:
            dropped_sample = []
            # 遍历样本里的每个特征，通过随机数进行判断
            for feature in sample:
                # 生成随机变量：以概率(1-p)为1，概率p为0
                keep_prob = 1 - self.p
                if random.random() < keep_prob:
                    # 保留该神经元，并进行inverted scaling
                    # 这样在推理时就不需要缩放了
                    scale_factor = 1.0 / keep_prob
                    mask = Node(scale_factor)
                else:
                    # 丢弃该神经元
                    mask = Node(0.0)

                # 应用mask，这个运算会正确地参与backward
                # 当mask=0时，梯度为0；当mask=scale_factor时，梯度会被相应缩放
                dropped_feature = feature * mask
                dropped_sample.append(dropped_feature)

            dropped.append(dropped_sample)

        # 如果输入是单个样本，返回单个样本
        return dropped[0] if is_single_sample else dropped

    def train(self):
        """设置为训练模式"""
        self.training = True

    def eval(self):
        """设置为评估模式"""
        self.training = False

    def parameters(self):
        """Dropout层没有可学习参数"""
        return []
