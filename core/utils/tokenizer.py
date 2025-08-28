from pathlib import Path

import jieba
from tqdm import tqdm


class Tokenizer:
    """分词器和词汇表管理类"""

    unk_token = '<unk>'
    pad_token = '<pad>'

    def __init__(self, vocab_list, file_path):
        self.file_path = file_path
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)

        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        self.index2word = {index: word for index, word in enumerate(vocab_list)}

        self.unk_token_id = self.word2index[self.unk_token]
        self.pad_token_id = self.word2index[self.pad_token]

    @staticmethod
    def tokenize(text):
        return jieba.lcut(text)

    def encode(self, text, seq_len):
        """
        文本生成id列表，如果超过阀值截断，如果不足阀值补充pad
        :param text:
        :return:
        """
        word_list = self.tokenize(text)
        if len(word_list) > seq_len:
            word_list = word_list[0:seq_len]
        elif len(word_list) < seq_len:
            word_list += [self.pad_token] * (seq_len - len(word_list))

        # print(f'填补后的词长度:{len(word_list)}')
        word_index_list = [self.word2index.get(word, self.unk_token_id) for word in word_list]
        return word_index_list

    @classmethod
    def from_vocab(cls, vocab_path):
        pass
        # 加载词表
        # 加载词表
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_list = [line[:-1] for line in f.readlines()]
        # 创建tokenizer
        return cls(vocab_list, vocab_path)

    @classmethod
    def build_vocab(cls, sentences, vocab_file):
        # 构建词表（用训练集）
        print(f'构建词表开始')
        vocab_set = set()
        for sentence in tqdm(sentences, desc='构建词表'):
            # 去掉不可见的tolen（' ','   ','\t'等等）
            for word in jieba.lcut(sentence):
                if word.strip() != '':
                    vocab_set.add(word)
        vocab_list = [cls.pad_token] + [cls.unk_token] + list(vocab_set)

        # 保存词表
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for word in vocab_list:
                f.write(word + '\n')
        print(f'构建词表完成')
        print(f'词表总大小：{len(vocab_list)}')


def build_vocab():
    sentences = ['我有一直小狗,我爱宠物']
    Tokenizer.build_vocab(sentences, PROCESSED_DIR / 'vocab.txt')


def from_vocab():
    tokenizer = Tokenizer.from_vocab(vocab_path=PROCESSED_DIR / 'vocab.txt')
    words = tokenizer.encode("我是一名警察", 2)
    print(words)


if __name__ == '__main__':
    ROOT_DIR = Path(__file__).parent.parent
    # 通过__file__这个全局变量获取这个文档的绝对路径。
    # 拿到绝对路径，然后根据根目录去寻找文件路径
    PROCESSED_DIR = ROOT_DIR / 'data' / 'processed'
    # build_vocab()
    from_vocab()
