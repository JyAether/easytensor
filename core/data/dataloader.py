import numpy as np

from core.tensor import tensor, Tensor
from abc import ABC, abstractmethod

class Dataset(ABC):
    """数据集抽象基类"""

    def __init__(self):
        pass

    @abstractmethod
    def __len__(self):
        """返回数据集大小"""
        pass

    @abstractmethod
    def __getitem__(self, index):
        """根据索引获取数据项"""
        pass


class TensorDataset(Dataset):
    """张量数据集 - 将多个张量包装成数据集"""

    def __init__(self, *tensors, device='cpu'):
        """
        Args:
            *tensors: 输入张量，第一维必须相同（样本数量）
            device: 设备类型
        """
        assert len(tensors) > 0, "至少需要一个张量"
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors), \
            "所有张量的第一维大小必须相同"

        self.tensors = []
        for tensor in tensors:
            if isinstance(tensor, Tensor):
                self.tensors.append(tensor.to(device))
            elif isinstance(tensor, np.ndarray):
                self.tensors.append(Tensor(tensor, device=device))
            else:
                self.tensors.append(Tensor(np.array(tensor), device=device))

        self.device = device
        self.length = self.tensors[0].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if len(self.tensors) == 1:
            return self.tensors[0][index]
        return tuple(tensor[index] for tensor in self.tensors)

    def to(self, device):
        """移动数据集到指定设备"""
        self.device = device
        self.tensors = [tensor.to(device) for tensor in self.tensors]
        return self


class DataLoader:
    """数据加载器 - 提供批量数据迭代功能"""

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        Args:
            dataset: 数据集对象
            batch_size: 批次大小
            shuffle: 是否打乱数据
            drop_last: 是否丢弃最后不完整的批次
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.length = len(dataset)

        # 计算批次数量
        if drop_last:
            self.num_batches = self.length // batch_size
        else:
            self.num_batches = (self.length + batch_size - 1) // batch_size

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # 生成索引
        indices = list(range(self.length))
        if self.shuffle:
            np.random.shuffle(indices)

        # 按批次返回数据
        for i in range(0, self.length, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]

            # 如果是最后一个批次且不足batch_size
            if len(batch_indices) < self.batch_size and self.drop_last:
                break

            batch_data = []
            for idx in batch_indices:
                item = self.dataset[idx]
                if isinstance(item, tuple):
                    batch_data.append(item)
                else:
                    batch_data.append((item,))

            # 组织批次数据
            if len(batch_data[0]) == 1:
                # 单个张量的情况
                batch_tensor_data = [item[0].data for item in batch_data]
                batch_tensor = Tensor(np.stack(batch_tensor_data), device=batch_data[0][0].device)
                yield batch_tensor
            else:
                # 多个张量的情况
                num_tensors = len(batch_data[0])
                batched_tensors = []
                for tensor_idx in range(num_tensors):
                    tensor_data = [item[tensor_idx].data for item in batch_data]
                    device = batch_data[0][tensor_idx].device
                    batched_tensor = Tensor(np.stack(tensor_data), device=device)
                    batched_tensors.append(batched_tensor)
                yield tuple(batched_tensors)