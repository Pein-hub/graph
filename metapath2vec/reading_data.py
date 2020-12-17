import numpy as np
import torch
from torch.utils.data import Dataset
from download import AminerDataset
np.random.seed(12345)

class DataReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, dataset, min_count, care_type):

        self.negatives = []
        self.discards = []
        self.negpos = 0
        self.care_type = care_type
        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0  # aminer一共有几行
        self.token_count = 0  # 单词数量（包括重复）
        self.word_frequency = dict()  # 词频统计
        self.inputFileName = dataset.fn  # 文件目录

        self.read_words(min_count)  # 统计词频
        self.initTableNegatives()  # 负采样
        self.initTableDiscards()  # ？？？

    def read_words(self, min_count):
        # 词频
        word_frequency = dict()
        for line in open(self.inputFileName, encoding="ISO-8859-1"):
            line = line.split()
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1
                        if self.token_count % 1000000 == 0:
                            print("Read " + str(int(self.token_count / 1000000)) + "M words.")

        # 给每个单词构建从0开始的id
        wid = 0
        for w, c in word_frequency.items():
            if c < min_count:
                # 小于min_count的则不统计进word2id
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1

        # 单词数，即节点数量
        # 单词都是dict保存的，没有重复的
        self.word_count = len(self.word2id)
        print("Total embeddings: " + str(len(self.word2id)))

    def initTableDiscards(self):
        # get a frequency table for sub-sampling. Note that the frequency is adjusted by
        # sub-sampling tricks.
        # word2vec中的降采样subsampling
        # 最高频的词汇，比如in，the，a这些词。这样的词汇通常比其它罕见词提供了更少的信息量。
        # http://d0evi1.com/word2vec-subsampling/
        # 词频越大，ran值就越小。subsampling进行抽样时被抽到的概率就越低。

        t = 0.0001
        # 每个单词的出现频率/一共有多少单词
        f = np.array(list(self.word_frequency.values())) / self.token_count

        # 其实最后还是一个根据词频大小定义的数组，只是多了一些其他变换
        # 当词频f大于一个阈值t时，说明此时词是一个高频词，那么np.sqrt(t / f) + (t / f)肯定是小于1的数，
        # 所以高频词被选中的概率就比较小。
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        # get a table for negative sampling, if word with index 2 appears twice, then 2 will be listed
        # in the table twice.
        # 负样本其实是相对于当时u和v来说的负样本，可以提前先定义好，因为属于正样本的概率特别小。不用后续每次采样

        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        # 负采样的id
        # 负采样要采1y个，然后按ratio分配到各个id词，即每个词要采多少个
        # [wid] * int(c)就表示每个词id要采样的个数，
        # 然后形成一个list,再shuffle一下，就得到负采样的样本
        # 所以负采样是一开始就得到的固定的？？？
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)
        # 每个单词被负采样采到的频率
        self.sampling_prob = ratio

    def getNegatives(self, target, size):  # TODO check equality with target
        # care_type 负采样的类型
        if self.care_type == 0:
            response = self.negatives[self.negpos:self.negpos + size]
            self.negpos = (self.negpos + size) % len(self.negatives) # 分母为1亿，大概率self.negpos + size递增，防止超出长度取余
            # 取到最后，从0开始循环取
            if len(response) != size:
                return np.concatenate((response, self.negatives[0:self.negpos]))
        return response


# -----------------------------------------------------------------------------------------------------------------

class Metapath2vecDataset(Dataset):
    def __init__(self, data, window_size):
        # read in input, window_size and input filename
        self.data = data
        self.window_size = window_size
        self.input_file = open(data.inputFileName, encoding="ISO-8859-1")

    def __len__(self):
        # return the number of walks
        return self.data.sentences_count

    def __getitem__(self, idx):
        # return the list of pairs (center, context, 5 negatives)

        # 一次读取一行数据，1个batch=50，一个batch就是50个lines
        # 一个line就是已一个node为起点的walk序列，而每一个序列长度是不定的，
        # 针对每一个line，构建一个 u，v，neg 三元组，给skipgram训练

        while True:
            # 1行1行读取，1行才是一个序列，包含图的结构
            line = self.input_file.readline()
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()

            if len(line) > 1:
                words = line.split()

                if len(words) > 1:
                    # subsampling
                    # 词频越大，discards 值越小，那么 np.random.rand()<discards的值就越小，被采样的概率就减小
                    # 读取该行word对应的id
                    word_ids = [self.data.word2id[w] for w in words if
                                w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]

                    pair_catch = []
                    for i, u in enumerate(word_ids):
                        for j, v in enumerate(
                                # 采样 i-win，i+win 的窗口数据，都是word的id
                                word_ids[max(i - self.window_size, 0):i + self.window_size]):
                            assert u < self.data.word_count
                            assert v < self.data.word_count
                            if i == j:
                                continue
                            # u是节点，v是windown邻居节点，每次取5个负样本
                            pair_catch.append((u, v, self.data.getNegatives(v, 5)))
                    # u1, v1, [negs5]
                    # u1, v2, [negs5]
                    # u2, v1, [negs5]
                    # u2, v2, [negs5]
                    # 这里都是word的下标
                    return pair_catch


    @staticmethod
    def collate(batches):
        """
        所以我们知道了collate_fn这个函数的输入就是一个list，
        list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果。
        :param batches:
        :return:
        """
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)
