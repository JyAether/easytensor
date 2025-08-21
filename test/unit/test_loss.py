from core.v1.engine import Node
from core.v1.nn2 import MSELoss, BCELoss, CrossEntropyLoss

print("\n" + "=" * 60)
print("=== 损失函数使用示例 ===")

# 创建不同的损失函数
mse_loss = MSELoss(reduction='mean')
cross_entropy_loss = CrossEntropyLoss(reduction='mean')
bce_loss = BCELoss(reduction='mean')

print("1. 测试MSE损失函数:")

# 测试数据
pred_single = Node(0.8)
target_single = Node(1.0)

pred_multi = [Node(0.8), Node(0.3)]
target_multi = [Node(1.0), Node(0.0)]

# 单个输出的损失计算
loss_single = mse_loss(pred_single, target_single)
print(f"  单个输出 MSE损失: {loss_single.data:.6f}")

# 多个输出的损失计算
loss_multi = mse_loss(pred_multi, target_multi)
print(f"  多个输出 MSE损失: {loss_multi.data:.6f}")

print("\n2. 测试多分类交叉熵损失函数:")

# 3分类问题示例
logits_3class = [Node(2.0), Node(1.0), Node(0.1)]  # 预测logits
target_class = 0  # 真实类别为第0类

ce_loss_3class = cross_entropy_loss(logits_3class, target_class)
print(f"  3分类CE损失: {ce_loss_3class.data:.6f}")
print(f"  预测logits: [{logits_3class[0].data:.1f}, {logits_3class[1].data:.1f}, {logits_3class[2].data:.1f}]")
print(f"  真实类别: {target_class}")

# 手动验证softmax概率
softmax_probs = cross_entropy_loss._log_softmax(logits_3class)
print(f"  Softmax概率: [{softmax_probs[0].data:.4f}, {softmax_probs[1].data:.4f}, {softmax_probs[2].data:.4f}]")
print(f"  概率和: {sum(p.data for p in softmax_probs):.6f}")

# 批量样本示例
batch_logits = [
    [Node(2.0), Node(1.0), Node(0.1)],  # 样本1
    [Node(0.5), Node(2.0), Node(1.0)],  # 样本2
]
batch_targets = [0, 1]  # 对应的真实类别

batch_ce_loss = cross_entropy_loss(batch_logits, batch_targets)
print(f"  批量CE损失: {batch_ce_loss.data:.6f}")

print("\n3. 测试二分类交叉熵损失函数:")

# 二分类问题（已经过sigmoid）
pred_binary = [Node(0.8), Node(0.3)]  # sigmoid输出概率
target_binary = [Node(1.0), Node(0.0)]  # 真实标签

bce_loss_result = bce_loss(pred_binary, target_binary)
print(f"  二分类BCE损失: {bce_loss_result.data:.6f}")
print(f"  预测概率: [{pred_binary[0].data:.3f}, {pred_binary[1].data:.3f}]")
print(f"  真实标签: [{target_binary[0].data:.3f}, {target_binary[1].data:.3f}]")

print("\n4. 对比不同损失函数:")
print("  MSE适用于: 回归问题")
print("  CrossEntropyLoss适用于: 多分类问题（自动应用softmax）")
print("  BCELoss适用于: 二分类问题（需预先应用sigmoid）")
