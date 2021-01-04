import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import os
import numpy as np

from reading_data import DataReader, Metapath2vecDataset
from model import SkipGramModel
from download import AminerDataset, CustomDataset


class Metapath2VecTrainer:
    def __init__(self, args):
        if args.aminer:
            dataset = AminerDataset(args.path)
        else:
            dataset = CustomDataset(args.path)
        self.data = DataReader(dataset, args.min_count, args.care_type)
        dataset = Metapath2vecDataset(self.data, args.window_size)

        # collate_fn 返回的是Tensor
        # collate_fn 返回的是 all_u, all_v, all_negs 3个tensor数组
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.num_workers, collate_fn=dataset.collate)

        self.output_path = args.output_file
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = args.dim
        self.batch_size = args.batch_size
        self.iterations = args.iterations
        self.initial_lr = args.initial_lr

        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)
        self.device = torch.device("cpu")
        self.skip_gram_model = self.skip_gram_model.to(self.device)
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            print('USE GPU')
            # 单机多GPU训练, 3个GPU训练即可, 机器有8个GPU
            gpu_list = self.get_gpu_memory()
            cuda1 = f'cuda:{gpu_list[0]}'
            cuda2 = f'cuda:{gpu_list[1]}'
            cuda3 = f'cuda:{gpu_list[2]}'
            print(cuda1, cuda2, cuda3)

            # device后续输入和loss使用 cuda1
            self.device = torch.device('cuda:6')
            # 改为GPU
            self.skip_gram_model = self.skip_gram_model.to(self.device)
            # # 改为单机多GPU
            # self.skip_gram_model = torch.nn.DataParallel(
            #     self.skip_gram_model,
            #     device_ids=[cuda1, cuda2, cuda3],
            #     output_device=cuda1)

    def train(self):
        memory = torch.cuda.memory_allocated(6) / (1024 * 1024)
        print(f'训练中: {memory}M')

        optimizer = optim.SparseAdam(self.skip_gram_model.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

        for iteration in range(self.iterations):
            print("\n\n\nIteration: " + str(iteration + 1))

            memory = torch.cuda.memory_allocated(6) / (1024 * 1024)
            print(f'迭代中: {memory}M')

            # 每一次迭代，都重新定义了一个 optimizer， scheduler，
            # 那么每次迭代训练，GPU都会保存当前 optimizer 对应的梯度动量么？ 应该是每次都保存了模型的参数？不应该啊，参数应该是共享的）
            # 每次 iteration 都是重新加载模型参数，更好训练
            # 相当于一次epoch后，已经训练好一个embedding了，
            # 一般 optimizer 是在epoch循环外面定义一次，是否这里反复定义导致GPU使用增加？


            # SparseAdam：基于稀疏张量的Adam
            # optimizer = optim.SparseAdam(self.skip_gram_model.parameters(), lr=self.initial_lr)
            # CosineAnnealingLR：余弦退火来调节学习率
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

            running_loss = 0.0
            # 一共3个循环，每次循环按50一个batch去遍历数据，理论上数据量已经很少了
            # 一个batch是读取一行数据，一行数据就是一个节点的walk路径信息
            for i, sample_batched in enumerate(tqdm(self.dataloader)):

                if len(sample_batched[0]) > 1:
                    # print('-------------------'*5)
                    # print(f'batch data1: {sample_batched[0].size()}')
                    # print(f'batch data2: {sample_batched[1].size()}')
                    # print(f'batch data3: {sample_batched[2].size()}')
                    #
                    # print(f'batch data1: {sample_batched[0].__sizeof__()}M')
                    # print(f'batch data2: {sample_batched[1].__sizeof__()}M')
                    # print(f'batch data3: {sample_batched[2].__sizeof__()}M')
                    #
                    # memory = torch.cuda.memory_allocated(6) / (1024 * 1024)
                    # print(f'GPU1: {memory}M')

                    # (1个batch里面的source node数量) * 1
                    pos_u = sample_batched[0].to(self.device)
                    # (1个batch里面source node对应的line里面的邻居数量) * 1
                    pos_v = sample_batched[1].to(self.device)
                    # （source node 对应的负采样） * 5
                    neg_v = sample_batched[2].to(self.device)

                    # memory = torch.cuda.memory_allocated(6) / (1024 * 1024)
                    # print(f'GPU2: {memory}M')

                    # print(f'Outside u size: {pos_u.size()}')
                    optimizer.zero_grad()
                    loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                    # 多gpu，返回每个gpu的loss，取平均
                    # loss = loss.mean()
                    loss.backward()

                    # memory = torch.cuda.memory_allocated(6) / (1024 * 1024)
                    # print(f'GPU3: {memory}M')

                    optimizer.step()
                    scheduler.step()

                    # memory = torch.cuda.memory_allocated(6) / (1024 * 1024)
                    # print(f'GPU4: {memory}M')

                    # 这个打印的损失是什么意思？
                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if i > 0 and i % 500 == 0:
                        print(" Loss: " + str(running_loss))
                        memory = torch.cuda.memory_allocated(6) / (1024 * 1024)
                        print(f'训练中: {memory}M')
                        print(f'batch data: {sample_batched[0].__sizeof__()}M')

                        if memory > 6666:
                            print('empty cache')
                            torch.cuda.empty_cache()  # 只会释放没有引有的垃圾,

            # DataParallel wrape后，要用 .module 来调用自定义Model的其他函数
            # 每次iteration都save一次
            self.skip_gram_model.save_embedding(self.data.id2word, self.output_path)

    @staticmethod
    def get_gpu_memory():
        """
        返回最大剩余内存的GPU下标
        """
        os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp.txt')
        memory_gpu = [int(x.split()[2]) for x in open('tmp.txt', 'r').readlines()]
        os.system('rm tmp.txt')
        print(f'GPU free memory: {memory_gpu}')
        gpu_list = np.argsort(memory_gpu)[::-1]
        return gpu_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Metapath2vec")
    # parser.add_argument('--input_file', type=str, help="input_file")
    parser.add_argument('--aminer', action='store_true', help='Use AMiner dataset')
    parser.add_argument('--path', default='./input', type=str, help="input_path")
    parser.add_argument('--output_file', default='./output', type=str, help='output_file')
    parser.add_argument('--dim', default=128, type=int, help="embedding dimensions")
    parser.add_argument('--window_size', default=7, type=int, help="context window size")
    parser.add_argument('--iterations', default=3, type=int, help="iterations")
    parser.add_argument('--batch_size', default=50, type=int, help="batch size")
    parser.add_argument('--care_type', default=0, type=int,
                        help="if 1, heterogeneous negative sampling, else normal negative sampling")
    parser.add_argument('--initial_lr', default=0.025, type=float, help="learning rate")
    parser.add_argument('--min_count', default=5, type=int, help="min count")  # 单词出现的次数，小于5次的不考虑
    parser.add_argument('--num_workers', default=16, type=int, help="number of workers")
    args = parser.parse_args()
    m2v = Metapath2VecTrainer(args)
    m2v.train()
