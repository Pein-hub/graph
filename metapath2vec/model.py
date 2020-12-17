import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import os

"""
    u_embedding: Embedding for center word.
    v_embedding: Embedding for neighbor words.
"""


class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v):
        # [126084, 128]
        emb_u = self.u_embeddings(pos_u)
        # [126084, 128]
        emb_v = self.v_embeddings(pos_v)
        # [126084, 5, 128]
        emb_neg_v = self.v_embeddings(neg_v)  # 负采样的embedding层和邻居v是同一个，权重共享

        # torch.mul 对应位相乘
        # torch.mm 矩阵乘法
        # 2个128维对应位相乘再sum，就是内积
        # [126084, 1]
        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        """
        压缩向量到-10，10之间
              | min, if x_i < min
        y_i = | x_i, if min <= x_i <= max
              | max, if x_i > max
        """
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        # 负采样得分
        # torch.squeeze() 这个函数主要对数据的维度进行压缩，去掉维数为1的的维度
        # torch.unsqueeze()这个函数主要是对数据维度进行扩充。给指定位置加上维数为一的维度
        # emb_u.unsqueeze(2)： [126084, 128] -> [126084, 128, 1]
        # emb_neg_v: [126084, 5, 128]
        # bmm: batch的矩阵乘法，[126084, 5, 128] * [126084, 128, 1] = [126084, 5, 1]
        # [126084, 5]  相当于和5个负样本计算得到5次得分
        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        # [126084，1] 5次得分之和
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        # 1次正样本和5个负样本的得分，进行一次训练
        return torch.mean(score + neg_score)

    def save_embedding(self, id2word, output_path):
        file_name = os.path.join(output_path, 'embedding.txt')
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))