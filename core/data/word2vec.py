from core.tensor import Tensor, zeros, randn
import numpy as np
import pickle
from collections import defaultdict, Counter
import re
from typing import List, Tuple, Dict, Optional


class Word2VecBase:
    """Word2Vec基类，包含共同的功能"""

    def __init__(self, vocab_size: int, embed_size: int, device='cpu'):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.device = device

        # 词汇表相关
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_freq = {}

    def build_vocab(self, sentences: List[List[str]], min_count: int = 1):
        """构建词汇表"""
        word_count = Counter()

        # 统计词频
        for sentence in sentences:
            for word in sentence:
                word_count[word] += 1

        # 过滤低频词，添加UNK标记
        vocab = ['<UNK>']
        for word, count in word_count.items():
            if count >= min_count:
                vocab.append(word)

        # 建立映射
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(vocab)
        self.word_freq = {word: word_count.get(word, 0) for word in vocab}

        print(f"词汇表构建完成，词汇量: {self.vocab_size}")
        return vocab

    def get_word_idx(self, word: str) -> int:
        """获取词的索引"""
        return self.word_to_idx.get(word, 0)  # 0是UNK的索引

    def generate_training_data(self, sentences: List[List[str]],
                               window_size: int = 2) -> List[Tuple]:
        """生成训练数据 - 子类需要重写"""
        raise NotImplementedError


class CBOW(Word2VecBase):
    """CBOW (Continuous Bag of Words) 模型

    给定上下文词，预测中心词
    """

    def __init__(self, vocab_size: int, embed_size: int, device='cpu'):
        super().__init__(vocab_size, embed_size, device)

        # 输入词嵌入矩阵 (vocab_size, embed_size)
        self.W_in = randn(vocab_size, embed_size, requires_grad=True, device=device) * 0.1

        # 输出词嵌入矩阵 (vocab_size, embed_size)
        self.W_out = randn(vocab_size, embed_size, requires_grad=True, device=device) * 0.1

    def generate_training_data(self, sentences: List[List[str]],
                               window_size: int = 2) -> List[Tuple]:
        """生成CBOW训练数据

        Returns:
            List[Tuple]: [(context_words_indices, center_word_index), ...]
        """
        training_data = []

        for sentence in sentences:
            # 转换为索引
            word_indices = [self.get_word_idx(word) for word in sentence]

            # 滑动窗口
            for center_idx in range(window_size, len(word_indices) - window_size):
                center_word = word_indices[center_idx]
                context_words = []

                # 收集上下文词（左边window_size个 + 右边window_size个）
                for i in range(center_idx - window_size, center_idx + window_size + 1):
                    if i != center_idx:  # 跳过中心词
                        context_words.append(word_indices[i])

                training_data.append((context_words, center_word))

        return training_data

    def forward(self, context_indices: List[int]) -> Tensor:
        """前向传播

        Args:
            context_indices: 上下文词的索引列表

        Returns:
            output_scores: 词汇表上的得分分布 (vocab_size,)
        """
        # 1. 获取上下文词的嵌入向量
        context_embeddings = []
        for idx in context_indices:
            embedding = self.W_in[idx]  # (embed_size,)
            context_embeddings.append(embedding)

        # 2. 计算上下文词嵌入的平均值
        if len(context_embeddings) == 1:
            context_avg = context_embeddings[0]
        else:
            # 手动计算平均值
            context_sum = context_embeddings[0]
            for i in range(1, len(context_embeddings)):
                context_sum = context_sum + context_embeddings[i]
            context_avg = context_sum / len(context_embeddings)

        # 3. 计算输出得分 (context_avg · W_out^T)
        output_scores = context_avg @ self.W_out.T  # (vocab_size,)

        return output_scores

    def compute_loss(self, output_scores: Tensor, target_idx: int) -> Tensor:
        """计算交叉熵损失（使用softmax）"""
        # 数值稳定的softmax
        max_score = output_scores.max()
        shifted_scores = output_scores - max_score
        exp_scores = shifted_scores.exp()
        sum_exp = exp_scores.sum()

        # 计算目标词的负对数似然
        target_score = shifted_scores[target_idx]
        log_prob = target_score - sum_exp.log()
        loss = -log_prob

        return loss

    def get_word_embeddings(self) -> np.ndarray:
        """获取训练好的词嵌入"""
        return self.W_in.data

    def parameters(self):
        """返回模型参数"""
        return [self.W_in, self.W_out]


class SkipGram(Word2VecBase):
    """Skip-gram 模型

    给定中心词，预测上下文词
    """

    def __init__(self, vocab_size: int, embed_size: int, device='cpu'):
        super().__init__(vocab_size, embed_size, device)

        # 输入词嵌入矩阵 (vocab_size, embed_size)
        self.W_in = randn(vocab_size, embed_size, requires_grad=True, device=device) * 0.1

        # 输出词嵌入矩阵 (vocab_size, embed_size)
        self.W_out = randn(vocab_size, embed_size, requires_grad=True, device=device) * 0.1

    def generate_training_data(self, sentences: List[List[str]],
                               window_size: int = 2) -> List[Tuple]:
        """生成Skip-gram训练数据

        Returns:
            List[Tuple]: [(center_word_index, context_word_index), ...]
        """
        training_data = []

        for sentence in sentences:
            # 转换为索引
            word_indices = [self.get_word_idx(word) for word in sentence]

            # 滑动窗口
            for center_idx in range(window_size, len(word_indices) - window_size):
                center_word = word_indices[center_idx]

                # 对每个上下文词位置生成一个训练样本
                for i in range(center_idx - window_size, center_idx + window_size + 1):
                    if i != center_idx:  # 跳过中心词
                        context_word = word_indices[i]
                        training_data.append((center_word, context_word))

        return training_data

    def forward(self, center_idx: int) -> Tensor:
        """前向传播

        Args:
            center_idx: 中心词的索引

        Returns:
            output_scores: 词汇表上的得分分布 (vocab_size,)
        """
        # 1. 获取中心词的嵌入向量
        center_embedding = self.W_in[center_idx]  # (embed_size,)

        # 2. 计算输出得分 (center_embedding · W_out^T)
        output_scores = center_embedding @ self.W_out.T  # (vocab_size,)

        return output_scores

    def compute_loss(self, output_scores: Tensor, target_idx: int) -> Tensor:
        """计算交叉熵损失（使用softmax）"""
        # 数值稳定的softmax
        max_score = output_scores.max()
        shifted_scores = output_scores - max_score
        exp_scores = shifted_scores.exp()
        sum_exp = exp_scores.sum()

        # 计算目标词的负对数似然
        target_score = shifted_scores[target_idx]
        log_prob = target_score - sum_exp.log()
        loss = -log_prob

        return loss

    def get_word_embeddings(self) -> np.ndarray:
        """获取训练好的词嵌入"""
        return self.W_in.data

    def parameters(self):
        """返回模型参数"""
        return [self.W_in, self.W_out]


class Word2VecTrainer:
    """Word2Vec训练器"""

    def __init__(self, model, learning_rate: float = 0.01):
        self.model = model
        self.lr = learning_rate

    def train_cbow(self, training_data: List[Tuple], epochs: int = 1,
                   verbose: bool = True):
        """训练CBOW模型"""
        total_loss = 0.0
        update_count = 0

        for epoch in range(epochs):
            epoch_loss = 0.0

            for context_words, center_word in training_data:
                # 前向传播
                output_scores = self.model.forward(context_words)

                # 计算损失
                loss = self.model.compute_loss(output_scores, center_word)

                # 反向传播
                self.zero_grad()
                loss.backward()

                # 更新参数
                self.update_parameters()

                epoch_loss += loss.data
                update_count += 1  # CBOW每个窗口更新1次

            if verbose:
                avg_loss = epoch_loss / len(training_data)
                print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.6f}")

        print(f"CBOW训练完成! 总更新次数: {update_count}")

    def train_skipgram(self, training_data: List[Tuple], epochs: int = 1,
                       verbose: bool = True):
        """训练Skip-gram模型"""
        total_loss = 0.0
        update_count = 0

        for epoch in range(epochs):
            epoch_loss = 0.0

            for center_word, context_word in training_data:
                # 前向传播
                output_scores = self.model.forward(center_word)

                # 计算损失
                loss = self.model.compute_loss(output_scores, context_word)

                # 反向传播
                self.zero_grad()
                loss.backward()

                # 更新参数
                self.update_parameters()

                epoch_loss += loss.data
                update_count += 1  # Skip-gram每个上下文词更新1次

            if verbose:
                avg_loss = epoch_loss / len(training_data)
                print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.6f}")

        print(f"Skip-gram训练完成! 总更新次数: {update_count}")

    def zero_grad(self):
        """清零梯度"""
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.fill(0)

    def update_parameters(self):
        """更新参数"""
        for param in self.model.parameters():
            if param.grad is not None:
                param.data = param.data - self.lr * param.grad.data


def preprocess_text(text: str) -> List[List[str]]:
    """简单的文本预处理"""
    # 转小写，去除标点
    text = re.sub(r'[^\w\s]', '', text.lower())

    # 分句
    sentences = text.split('.')

    # 分词
    tokenized_sentences = []
    for sentence in sentences:
        words = sentence.strip().split()
        if len(words) > 0:
            tokenized_sentences.append(words)

    return tokenized_sentences


def find_similar_words(word: str, model, top_k: int = 5) -> List[Tuple[str, float]]:
    """找到最相似的词"""
    if word not in model.word_to_idx:
        print(f"词 '{word}' 不在词汇表中")
        return []

    word_idx = model.get_word_idx(word)
    word_embedding = model.get_word_embeddings()[word_idx]

    similarities = []
    embeddings = model.get_word_embeddings()

    for i, other_word in model.idx_to_word.items():
        if i != word_idx:
            other_embedding = embeddings[i]

            # 计算余弦相似度
            dot_product = np.dot(word_embedding, other_embedding)
            norm1 = np.linalg.norm(word_embedding)
            norm2 = np.linalg.norm(other_embedding)

            if norm1 > 0 and norm2 > 0:
                similarity = dot_product / (norm1 * norm2)
                similarities.append((other_word, similarity))

    # 排序并返回top_k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 示例文本
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    Dogs are loyal animals and they make great pets.
    Cats are independent creatures but can be very affectionate.
    Both cats and dogs require proper care and attention.
    Animals bring joy to our lives.
    """

    print("=== Word2Vec 训练示例 ===\n")

    # 预处理文本
    sentences = preprocess_text(sample_text)
    print("预处理后的句子:")
    for i, sentence in enumerate(sentences):
        print(f"{i + 1}: {sentence}")

    # ==================== CBOW 训练 ====================
    print(f"\n{'=' * 50}")
    print("训练 CBOW 模型")
    print(f"{'=' * 50}")

    # 创建CBOW模型
    cbow_model = CBOW(vocab_size=100, embed_size=50)  # 初始vocab_size会被更新
    cbow_model.build_vocab(sentences, min_count=1)

    # 重新初始化嵌入矩阵（因为词汇表大小已确定）
    cbow_model.W_in = randn(cbow_model.vocab_size, cbow_model.embed_size,
                            requires_grad=True) * 0.1
    cbow_model.W_out = randn(cbow_model.vocab_size, cbow_model.embed_size,
                             requires_grad=True) * 0.1

    # 生成训练数据
    window_size = 2
    cbow_training_data = cbow_model.generate_training_data(sentences, window_size)
    print(f"\nCBOW训练样本数: {len(cbow_training_data)}")

    # 显示一些训练样本
    print("\n前5个CBOW训练样本:")
    for i, (context, center) in enumerate(cbow_training_data[:5]):
        context_words = [cbow_model.idx_to_word[idx] for idx in context]
        center_word = cbow_model.idx_to_word[center]
        print(f"  上下文: {context_words} -> 中心词: {center_word}")

    # 训练CBOW
    cbow_trainer = Word2VecTrainer(cbow_model, learning_rate=0.1)
    cbow_trainer.train_cbow(cbow_training_data, epochs=3)

    # ==================== Skip-gram 训练 ====================
    print(f"\n{'=' * 50}")
    print("训练 Skip-gram 模型")
    print(f"{'=' * 50}")

    # 创建Skip-gram模型
    skipgram_model = SkipGram(vocab_size=100, embed_size=50)
    skipgram_model.build_vocab(sentences, min_count=1)

    # 重新初始化嵌入矩阵
    skipgram_model.W_in = randn(skipgram_model.vocab_size, skipgram_model.embed_size,
                                requires_grad=True) * 0.1
    skipgram_model.W_out = randn(skipgram_model.vocab_size, skipgram_model.embed_size,
                                 requires_grad=True) * 0.1

    # 生成训练数据
    skipgram_training_data = skipgram_model.generate_training_data(sentences, window_size)
    print(f"\nSkip-gram训练样本数: {len(skipgram_training_data)}")

    # 显示一些训练样本
    print("\n前5个Skip-gram训练样本:")
    for i, (center, context) in enumerate(skipgram_training_data[:5]):
        center_word = skipgram_model.idx_to_word[center]
        context_word = skipgram_model.idx_to_word[context]
        print(f"  中心词: {center_word} -> 上下文词: {context_word}")

    # 训练Skip-gram
    skipgram_trainer = Word2VecTrainer(skipgram_model, learning_rate=0.1)
    skipgram_trainer.train_skipgram(skipgram_training_data, epochs=3)

    # ==================== 更新次数分析 ====================
    print(f"\n{'=' * 50}")
    print("更新次数分析")
    print(f"{'=' * 50}")

    # 计算每个窗口的更新次数
    unique_windows = set()
    for sentence in sentences:
        for center_idx in range(window_size, len(sentence) - window_size):
            # 创建窗口的唯一标识
            window_id = (tuple(sentence), center_idx)
            unique_windows.add(window_id)

    num_windows = len(unique_windows)
    cbow_updates_per_window = 1  # CBOW每个窗口1次更新
    skipgram_updates_per_window = window_size * 2  # Skip-gram每个窗口2*window_size次更新

    print(f"总窗口数: {num_windows}")
    print(f"窗口大小: {window_size} (左右各{window_size}个词)")
    print(f"CBOW每个窗口更新次数: {cbow_updates_per_window}")
    print(f"Skip-gram每个窗口更新次数: {skipgram_updates_per_window}")
    print(f"CBOW总更新次数: {num_windows * cbow_updates_per_window}")
    print(f"Skip-gram总更新次数: {num_windows * skipgram_updates_per_window}")

    # ==================== 词相似度测试 ====================
    print(f"\n{'=' * 50}")
    print("词相似度测试")
    print(f"{'=' * 50}")

    test_words = ['dog', 'cat', 'animals']

    print("CBOW模型相似词:")
    for word in test_words:
        if word in cbow_model.word_to_idx:
            similar_words = find_similar_words(word, cbow_model, top_k=3)
            print(f"  {word}: {similar_words}")

    print("\nSkip-gram模型相似词:")
    for word in test_words:
        if word in skipgram_model.word_to_idx:
            similar_words = find_similar_words(word, skipgram_model, top_k=3)
            print(f"  {word}: {similar_words}")