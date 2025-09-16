import numpy as np
from typing import List, Tuple, Dict, Optional

from core.nn import Module, Linear, CrossEntropyLoss, Adam
from core.nn.bert_gpt import create_bert_base
from core.tensor import Tensor


class BERTForSentencePairClassification(Module):
    """
    (a) 句子对分类任务 - Sentence Pair Classification Tasks
    用于: MNLI, QQP, QNLI, STS-B, MRPC, RTE, SWAG
    输入格式: [CLS] Sentence1 [SEP] Sentence2 [SEP]
    输出: 使用[CLS]位置的输出进行分类
    """

    def __init__(self, bert_model, num_classes, dropout_rate=0.1, device='cpu'):
        super().__init__()
        self.bert = bert_model
        self.num_classes = num_classes
        self.device = device

        # 分类头
        self.dropout = Dropout(dropout_rate)
        self.classifier = Linear(bert_model.d_model, num_classes, device=device)

    def forward(self, input_ids, segment_ids, attention_mask=None):
        """
        input_ids: [batch_size, seq_len] - 包含[CLS] Sent1 [SEP] Sent2 [SEP]
        segment_ids: [batch_size, seq_len] - Sent1部分为0，Sent2部分为1
        """
        # 获取BERT输出
        hidden_states = self.bert(input_ids, segment_ids, attention_mask)

        # 使用[CLS]标记的输出（第一个位置）
        cls_output = Tensor(
            hidden_states.data[:, 0, :],
            requires_grad=hidden_states.requires_grad,
            device=hidden_states.device
        )

        # Dropout和分类
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        return logits

    def predict_similarity(self, sent1_tokens, sent2_tokens):
        """预测句子相似度（用于STS-B任务）"""
        input_ids, segment_ids = self.prepare_sentence_pair_input(sent1_tokens, sent2_tokens)
        logits = self.forward(input_ids, segment_ids)

        # 对于回归任务，输出单个值
        if self.num_classes == 1:
            return logits.data[0, 0]  # 返回相似度分数
        else:
            # 对于分类任务，返回概率分布
            softmax_scores = self.softmax(logits)
            return softmax_scores

    def prepare_sentence_pair_input(self, sent1_tokens, sent2_tokens, max_len=128):
        """准备句子对输入"""
        # 特殊标记
        CLS_ID, SEP_ID = 101, 102  # 假设的特殊标记ID

        # 构建输入序列: [CLS] sent1 [SEP] sent2 [SEP]
        input_ids = [CLS_ID] + sent1_tokens + [SEP_ID] + sent2_tokens + [SEP_ID]
        segment_ids = [0] * (len(sent1_tokens) + 2) + [1] * (len(sent2_tokens) + 1)

        # 截断或填充
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            segment_ids = segment_ids[:max_len]
        else:
            padding_len = max_len - len(input_ids)
            input_ids.extend([0] * padding_len)  # 0为PAD标记
            segment_ids.extend([0] * padding_len)

        return (
            Tensor(np.array([input_ids]), device=self.device),
            Tensor(np.array([segment_ids]), device=self.device)
        )


class BERTForSingleSentenceClassification(Module):
    """
    (b) 单句分类任务 - Single Sentence Classification Tasks
    用于: SST-2, CoLA
    输入格式: [CLS] Sentence [SEP]
    输出: 使用[CLS]位置的输出进行分类
    """

    def __init__(self, bert_model, num_classes, dropout_rate=0.1, device='cpu'):
        super().__init__()
        self.bert = bert_model
        self.num_classes = num_classes
        self.device = device

        # 分类头
        self.dropout = Dropout(dropout_rate)
        self.classifier = Linear(bert_model.d_model, num_classes, device=device)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: [batch_size, seq_len] - 包含[CLS] Sentence [SEP]
        """
        # segment_ids全为0（单句任务）
        segment_ids = Tensor(np.zeros_like(input_ids.data), device=self.device)

        # 获取BERT输出
        hidden_states = self.bert(input_ids, segment_ids, attention_mask)

        # 使用[CLS]标记的输出
        cls_output = Tensor(
            hidden_states.data[:, 0, :],
            requires_grad=hidden_states.requires_grad,
            device=hidden_states.device
        )

        # Dropout和分类
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        return logits

    def predict_sentiment(self, sentence_tokens):
        """情感分析预测（SST-2）"""
        input_ids = self.prepare_single_sentence_input(sentence_tokens)
        logits = self.forward(input_ids)

        # 返回预测类别 (0: 负面, 1: 正面)
        predictions = np.argmax(logits.data, axis=1)
        return predictions[0]

    def predict_acceptability(self, sentence_tokens):
        """语法可接受性判断（CoLA）"""
        input_ids = self.prepare_single_sentence_input(sentence_tokens)
        logits = self.forward(input_ids)

        # 返回可接受性分数
        softmax_scores = self.softmax(logits)
        return softmax_scores.data[0, 1]  # 可接受的概率

    def prepare_single_sentence_input(self, sentence_tokens, max_len=128):
        """准备单句输入"""
        CLS_ID, SEP_ID = 101, 102

        # 构建输入序列: [CLS] sentence [SEP]
        input_ids = [CLS_ID] + sentence_tokens + [SEP_ID]

        # 截断或填充
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
        else:
            padding_len = max_len - len(input_ids)
            input_ids.extend([0] * padding_len)

        return Tensor(np.array([input_ids]), device=self.device)


class BERTForQuestionAnswering(Module):
    """
    (c) 问答任务 - Question Answering Tasks
    用于: SQuAD v1.1
    输入格式: [CLS] Question [SEP] Paragraph [SEP]
    输出: 预测答案的起始和结束位置
    """

    def __init__(self, bert_model, device='cpu'):
        super().__init__()
        self.bert = bert_model
        self.device = device

        # QA输出层：预测start和end位置
        self.qa_outputs = Linear(bert_model.d_model, 2, device=device)

    def forward(self, input_ids, segment_ids, attention_mask=None):
        """
        input_ids: [batch_size, seq_len] - 包含[CLS] Question [SEP] Paragraph [SEP]
        segment_ids: [batch_size, seq_len] - Question部分为0，Paragraph部分为1
        """
        # 获取BERT输出
        hidden_states = self.bert(input_ids, segment_ids, attention_mask)

        # 通过QA输出层
        logits = self.qa_outputs(hidden_states)

        # 分离start和end logits
        start_logits = Tensor(
            logits.data[:, :, 0],
            requires_grad=logits.requires_grad,
            device=logits.device
        )
        end_logits = Tensor(
            logits.data[:, :, 1],
            requires_grad=logits.requires_grad,
            device=logits.device
        )

        return start_logits, end_logits

    def predict_answer(self, question_tokens, paragraph_tokens, paragraph_text):
        """预测答案"""
        input_ids, segment_ids, token_to_char_map = self.prepare_qa_input(
            question_tokens, paragraph_tokens, paragraph_text
        )

        start_logits, end_logits = self.forward(input_ids, segment_ids)

        # 找到最佳的start和end位置
        start_idx = np.argmax(start_logits.data[0])
        end_idx = np.argmax(end_logits.data[0])

        # 确保end >= start且都在paragraph范围内
        question_len = len(question_tokens) + 2  # [CLS] + question + [SEP]

        if start_idx < question_len:
            start_idx = question_len
        if end_idx < start_idx:
            end_idx = start_idx

        # 提取答案文本
        if start_idx in token_to_char_map and end_idx in token_to_char_map:
            start_char = token_to_char_map[start_idx][0]
            end_char = token_to_char_map[end_idx][1]
            answer = paragraph_text[start_char:end_char]
        else:
            answer = ""

        return {
            'answer': answer,
            'start_logit': float(start_logits.data[0, start_idx]),
            'end_logit': float(end_logits.data[0, end_idx]),
            'start_idx': start_idx,
            'end_idx': end_idx
        }

    def prepare_qa_input(self, question_tokens, paragraph_tokens, paragraph_text, max_len=384):
        """准备问答输入"""
        CLS_ID, SEP_ID = 101, 102

        # 构建输入序列: [CLS] question [SEP] paragraph [SEP]
        input_ids = [CLS_ID] + question_tokens + [SEP_ID] + paragraph_tokens + [SEP_ID]
        segment_ids = [0] * (len(question_tokens) + 2) + [1] * (len(paragraph_tokens) + 1)

        # 创建token到字符的映射（用于提取答案）
        token_to_char_map = {}
        char_idx = 0

        # 跳过question部分
        question_len = len(question_tokens) + 2
        for i in range(question_len, len(input_ids) - 1):  # 跳过最后的[SEP]
            token_start = char_idx
            # 假设每个token对应一个词，这里简化处理
            token_end = char_idx + 1  # 实际应该根据tokenizer计算
            token_to_char_map[i] = (token_start, token_end)
            char_idx = token_end

        # 截断处理
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            segment_ids = segment_ids[:max_len]

        return (
            Tensor(np.array([input_ids]), device=self.device),
            Tensor(np.array([segment_ids]), device=self.device),
            token_to_char_map
        )


class BERTForTokenClassification(Module):
    """
    (d) 序列标注任务 - Single Sentence Tagging Tasks
    用于: CoNLL-2003 NER (命名实体识别)
    输入格式: [CLS] Token1 Token2 ... TokenN [SEP]
    输出: 对每个token进行分类 (O, B-PER, I-PER, B-LOC, I-LOC, 等)
    """

    def __init__(self, bert_model, num_labels, dropout_rate=0.1, device='cpu'):
        super().__init__()
        self.bert = bert_model
        self.num_labels = num_labels
        self.device = device

        # 标注头
        self.dropout = Dropout(dropout_rate)
        self.classifier = Linear(bert_model.d_model, num_labels, device=device)

        # NER标签映射 (CoNLL-2003 NER)
        self.id2label = {
            0: 'O',  # Outside
            1: 'B-PER',  # Begin Person
            2: 'I-PER',  # Inside Person
            3: 'B-LOC',  # Begin Location
            4: 'I-LOC',  # Inside Location
            5: 'B-ORG',  # Begin Organization
            6: 'I-ORG',  # Inside Organization
            7: 'B-MISC',  # Begin Miscellaneous
            8: 'I-MISC'  # Inside Miscellaneous
        }

        self.label2id = {v: k for k, v in self.id2label.items()}

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: [batch_size, seq_len] - 包含[CLS] tokens [SEP]
        返回每个token的分类logits
        """
        # segment_ids全为0（单句任务）
        segment_ids = Tensor(np.zeros_like(input_ids.data), device=self.device)

        # 获取BERT输出
        hidden_states = self.bert(input_ids, segment_ids, attention_mask)

        # Dropout
        hidden_states = self.dropout(hidden_states)

        # 对每个token进行分类
        logits = self.classifier(hidden_states)

        return logits

    def predict_entities(self, sentence_tokens):
        """预测命名实体"""
        input_ids, token_mapping = self.prepare_token_classification_input(sentence_tokens)

        # 获取预测logits
        logits = self.forward(input_ids)

        # 获取预测标签
        predictions = np.argmax(logits.data[0], axis=1)

        # 提取实际token的标签（排除[CLS]和[SEP]）
        token_labels = []
        entities = []
        current_entity = None

        for i, (token, pred_id) in enumerate(zip(sentence_tokens, predictions[1:-1])):  # 跳过[CLS]和[SEP]
            label = self.id2label[pred_id]
            token_labels.append((token, label))

            # 提取实体
            if label.startswith('B-'):
                # 开始新实体
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'text': token,
                    'label': label[2:],  # 移除B-前缀
                    'start': i,
                    'end': i
                }
            elif label.startswith('I-') and current_entity:
                # 继续当前实体
                if label[2:] == current_entity['label']:
                    current_entity['text'] += ' ' + token
                    current_entity['end'] = i
                else:
                    # 标签不匹配，结束当前实体
                    entities.append(current_entity)
                    current_entity = None
            else:
                # O标签或不匹配的I-标签
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        # 添加最后一个实体
        if current_entity:
            entities.append(current_entity)

        return {
            'token_labels': token_labels,
            'entities': entities
        }

    def prepare_token_classification_input(self, sentence_tokens, max_len=128):
        """准备序列标注输入"""
        CLS_ID, SEP_ID = 101, 102

        # 构建输入序列: [CLS] tokens [SEP]
        input_ids = [CLS_ID] + sentence_tokens + [SEP_ID]

        # 创建token映射
        token_mapping = {}
        for i, token_id in enumerate(sentence_tokens):
            token_mapping[i + 1] = token_id  # +1 因为[CLS]占据位置0

        # 截断或填充
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
        else:
            padding_len = max_len - len(input_ids)
            input_ids.extend([0] * padding_len)

        return Tensor(np.array([input_ids]), device=self.device), token_mapping


class Dropout(Module):
    """Dropout层"""

    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x):
        if not self.training:
            return x

        # 简化的dropout实现
        xp = x._get_array_module() if hasattr(x, '_get_array_module') else np
        mask = xp.random.rand(*x.shape) > self.dropout_rate
        output_data = x.data * mask / (1.0 - self.dropout_rate)

        return Tensor(output_data, requires_grad=x.requires_grad, device=x.device)

    def softmax(self, x):
        """Softmax函数"""
        xp = x._get_array_module() if hasattr(x, '_get_array_module') else np
        exp_x = xp.exp(x.data - xp.max(x.data, axis=-1, keepdims=True))
        softmax_x = exp_x / xp.sum(exp_x, axis=-1, keepdims=True)
        return Tensor(softmax_x, requires_grad=x.requires_grad, device=x.device)


# 任务特定的损失函数
class QALoss(Module):
    """问答任务损失函数"""

    def __init__(self):
        super().__init__()
        self.cross_entropy = CrossEntropyLoss()

    def forward(self, start_logits, end_logits, start_positions, end_positions):
        """
        start_logits, end_logits: [batch_size, seq_len]
        start_positions, end_positions: [batch_size]
        """
        start_loss = self.cross_entropy(start_logits, start_positions)
        end_loss = self.cross_entropy(end_logits, end_positions)

        total_loss = Tensor(
            (start_loss.data + end_loss.data) / 2,
            requires_grad=True,
            device=start_logits.device
        )

        return total_loss


# 使用示例和训练函数
def train_sentence_pair_classification(model, train_data, epochs=3, lr=2e-5):
    """训练句子对分类模型"""
    optimizer = Adam(model.parameters(), lr=lr)  # 需要实现Adam优化器
    loss_fn = CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_data:
            input_ids, segment_ids, labels = batch

            # 前向传播
            logits = model(input_ids, segment_ids)
            loss = loss_fn(logits, labels)

            # 反向传播
            model.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.data

        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(train_data)}")


def train_question_answering(model, train_data, epochs=3, lr=3e-5):
    """训练问答模型"""
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = QALoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_data:
            input_ids, segment_ids, start_positions, end_positions = batch

            # 前向传播
            start_logits, end_logits = model(input_ids, segment_ids)
            loss = loss_fn(start_logits, end_logits, start_positions, end_positions)

            # 反向传播
            model.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.data

        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(train_data)}")


def train_token_classification(model, train_data, epochs=3, lr=5e-5):
    """训练序列标注模型"""
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_data:
            input_ids, labels = batch

            # 前向传播
            logits = model(input_ids)

            # 重塑logits和labels以匹配损失函数期望的形状
            batch_size, seq_len, num_labels = logits.shape
            logits_flat = Tensor(
                logits.data.reshape(-1, num_labels),
                requires_grad=logits.requires_grad,
                device=logits.device
            )
            labels_flat = Tensor(
                labels.data.reshape(-1),
                requires_grad=False,
                device=labels.device
            )

            loss = loss_fn(logits_flat, labels_flat)

            # 反向传播
            model.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.data

        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(train_data)}")


if __name__ == "__main__":
    print("BERT四种微调任务测试！")


    bert_base = create_bert_base()
    device = 'cpu'

    # 1. 句子对分类任务（如MNLI, QQP）
    pair_classifier = BERTForSentencePairClassification(bert_base, num_classes=3, device=device)
    print("句子对分类模型创建完成")

    # 2. 单句分类任务（如SST-2, CoLA）
    single_classifier = BERTForSingleSentenceClassification(bert_base, num_classes=2, device=device)
    print("单句分类模型创建完成")

    # 3. 问答任务（如SQuAD）
    qa_model = BERTForQuestionAnswering(bert_base, device=device)
    print("问答模型创建完成")

    # 4. 序列标注任务（如NER）
    ner_model = BERTForTokenClassification(bert_base, num_labels=9, device=device)  # CoNLL-2003 NER
    print("序列标注模型创建完成")

    print("\n所有四种BERT微调任务模型都已准备就绪！")

    # 示例使用
    print("\n示例使用：")

    # 句子对分类示例
    sent1 = [1, 2, 3, 4]  # 假设的token IDs
    sent2 = [5, 6, 7, 8]
    # similarity_score = pair_classifier.predict_similarity(sent1, sent2)

    # 单句分类示例
    sentence = [1, 2, 3, 4, 5]
    sentiment = single_classifier.predict_sentiment(sentence)

    # 问答示例
    question = [1, 2, 3]
    paragraph = [4, 5, 6, 7, 8]
    paragraph_text = "This is a sample paragraph."
    # answer = qa_model.predict_answer(question, paragraph, paragraph_text)

    # 序列标注示例
    tokens = [1, 2, 3, 4, 5]
    # entities = ner_model.predict_entities(tokens)

    print("所有示例接口都已准备完毕！")