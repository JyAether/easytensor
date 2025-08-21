import numpy as np

# 生成训练数据
np.random.seed(42)
n_samples = 100
X_train = np.random.uniform(-2, 2, (n_samples, 2))

def target_function(x1, x2):
    return 0.5*np.sin(x1) + 0.3*x2**2 + 0.1*x1*x2 + 0.2

y_train = target_function(X_train[:, 0], X_train[:, 1]) + 0.05 * np.random.randn(n_samples)

print(f"训练数据: {n_samples} 个样本")
print(f"输入范围: x1∈[{X_train[:, 0].min():.2f}, {X_train[:, 0].max():.2f}], x2∈[{X_train[:, 1].min():.2f}, {X_train[:, 1].max():.2f}]")
print(f"输出范围: y∈[{y_train.min():.2f}, {y_train.max():.2f}]")