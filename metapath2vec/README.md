metapath2vec
1. metapath如何实现
    - 指定edge type
5. 结合metapath的walk length
    - 一个metapath是一个walk step，如果一个metapath=4，walk length=100，采样长度=400
6. 为什么用sparseAdam：
    - skipgram输入为word embedding，比较sparse
7. 单机多GPU
    - 目前单机8个GPU，1个GPU只有10G显存，amnier数据有12G





