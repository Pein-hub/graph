import numpy as np
import random
import time
import tqdm
import dgl
import sys
import os

num_walks_per_node = 1000
walk_length = 100
path = './input/net_aminer'

def construct_graph():
    paper_ids = []
    paper_names = []
    author_ids = []
    author_names = []
    conf_ids = []
    conf_names = []
    f_3 = open(os.path.join(path, "id_author.txt"), encoding="ISO-8859-1")
    f_4 = open(os.path.join(path, "id_conf.txt"), encoding="ISO-8859-1")
    f_5 = open(os.path.join(path, "paper.txt"), encoding="ISO-8859-1")
    while True:
        z = f_3.readline()
        if not z:
            break
        z = z.strip().split()
        identity = int(z[0])
        author_ids.append(identity)
        author_names.append(z[1])
    while True:
        w = f_4.readline()
        if not w:
            break
        w = w.strip().split()
        identity = int(w[0])
        conf_ids.append(identity)
        conf_names.append(w[1])
    while True:
        v = f_5.readline()
        if not v:
            break
        v = v.strip().split()
        identity = int(v[0])
        paper_name = 'p' + ''.join(v[1:])
        paper_ids.append(identity)
        paper_names.append(paper_name)
    f_3.close()
    f_4.close()
    f_5.close()

    # 原始id重新编码成0开始，从而和names的下标一一对应
    author_ids_invmap = {x: i for i, x in enumerate(author_ids)}
    conf_ids_invmap = {x: i for i, x in enumerate(conf_ids)}
    paper_ids_invmap = {x: i for i, x in enumerate(paper_ids)}

    # 保存pa的起始和结束点，新id
    paper_author_src = []
    paper_author_dst = []

    # 保存pc的起始和结束点，新id
    paper_conf_src = []
    paper_conf_dst = []
    f_1 = open(os.path.join(path, "paper_author.txt"), "r")
    f_2 = open(os.path.join(path, "paper_conf.txt"), "r")

    for x in f_1:
        x = x.split('\t')
        x[0] = int(x[0])
        x[1] = int(x[1].strip('\n'))
        paper_author_src.append(paper_ids_invmap[x[0]])
        paper_author_dst.append(author_ids_invmap[x[1]])
    for y in f_2:
        y = y.split('\t')
        y[0] = int(y[0])
        y[1] = int(y[1].strip('\n'))
        paper_conf_src.append(paper_ids_invmap[y[0]])
        paper_conf_dst.append(conf_ids_invmap[y[1]])
    f_1.close()
    f_2.close()

    hg = dgl.heterograph({
        ('paper', 'pa', 'author') : (paper_author_src, paper_author_dst),
        ('author', 'ap', 'paper') : (paper_author_dst, paper_author_src),
        ('paper', 'pc', 'conf') : (paper_conf_src, paper_conf_dst),
        ('conf', 'cp', 'paper') : (paper_conf_dst, paper_conf_src)})
    return hg, author_names, conf_names, paper_names

#"conference - paper - Author - paper - conference" metapath sampling
def generate_metapath():
    output_path = open(os.path.join(path, "output_path.txt"), "w")
    count = 0

    hg, author_names, conf_names, paper_names = construct_graph()

    for conf_idx in tqdm.trange(hg.number_of_nodes('conf')):
        # metapath：Metapath, specified as a list of edge types.
        # 如果整个路径指定了metapath，就不需要 length参数
        # (num_seeds, len(metapath) + 1)
        # metapath这里是400， 一个metapath相当于一个 walk_length
        traces, _ = dgl.sampling.random_walk(hg,
                                             # walk次数，返回多个以conf_idx为起点的游走序列，下标
                                             [conf_idx] * num_walks_per_node,
                                             # 每个序列走100步，这100步都是cpapc路径走, 一共有400个
                                             metapath=['cp', 'pa', 'ap', 'pc'] * walk_length)

        # The returned traces all have length len(metapath) + 1,
        # where the first node is the starting node itself.
        for tr in traces:

            # 只保存vName和aName，
            # 因为结果是预测v和a的分类，（venus，author），只需要保存这2个节点的graph struct就行了么
            # 其实有p和没有p在graph中都是一样的连线
            # 把paper去掉（因为之前有论文这样干，这里保持队型）
            # 一个 outline 是一个字符串， 包含 201个单词， 1个source word， 100个a和100个v
            outline = ' '.join(
                    (conf_names if i % 4 == 0 else author_names)[tr[i]]
                    for i in range(0, len(tr), 2))  # skip paper

            # 这里直接用print保存的txt文件
            print(outline, file=output_path)
    output_path.close()


if __name__ == "__main__":
    # 相当于每个node随机游走10000次，每次生成一个长度100的序列，序列严格按照元路径类型走
    # 然后只保存a和v的序列信息
    # 这样采集的10000*num_nodes的序列数据，就包含了graph的信息，后面就是输入skipgram训练
    generate_metapath()
