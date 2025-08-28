import numpy as np

from core.tensor import tensor


class DataLoader:
    """简单的数据加载器"""

    def __init__(self, dataset, batch_size=8, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(dataset[0])
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size

    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_x = self.dataset[0][batch_indices]
            batch_y = self.dataset[1][batch_indices]
            yield tensor(batch_x, requires_grad=False), tensor(batch_y, requires_grad=False)

    def __len__(self):
        return self.num_batches