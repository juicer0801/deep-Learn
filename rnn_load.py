import torch
import jieba
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time

# 加载数据
def build_vocab():

    # 1. 记录去重后所有的词，每行文本分词的结果
    unique_works,all_words = [], []

    # 2. 读取数据
    for line in open('data/jaychou_lyrics.txt', 'r', encoding='utf-8'):

        #分词
        words = jieba.lcut(line)

        #保存所有词到all_word
        all_words.append(words)

        #去重并保存
        for word in words:
            if word not in unique_works:
                unique_works.append(word)

    #统计词频（去重后的词数量）
    word_count = len(unique_works)
    print(f"词频统计结果：{word_count}")

    #构建词表
    word_dict = {word: index for index, word in enumerate(unique_works)}
    print(word_dict)

    #保存词表
    with open('data/word_dict.json', 'w', encoding='utf-8') as f:
        json.dump(word_dict, f)

    # 5. 歌词文本用词表索引表示.
    corpus_idx = []

    #遍历每一行分词结果
    for words in all_words:

        #定义变量，记录词索引列表
        tmp = []

        # 6.2 获取每一行的词, 并获取相应的索引.
        for word in words:
            tmp.append(word_dict[word])

        #添加空格
        tmp.append(word_dict[' '])

        #添加到corpus_idx
        corpus_idx.append(tmp)

    return unique_works, corpus_idx, word_dict, word_count

# 构建数据集
class LyricDataset(torch.utils.data.Dataset):

    # 初始化
    def __init__(self, corpus_idx, num_chars):

        # 保存数据集
        self.corpus_idx = corpus_idx

        # 保存每个句子中词的个数
        self.num_chars = num_chars

        #词的数量,不去重
        self.word_count = len(self.corpus_idx)

        #句子数量
        self.number = self.word_count // self.num_chars

    # 获取数据集长度
    def __len__(self):
        return self.number

    # 获取数据
    def __getitem__(self, idx):

        start = min(max(idx,0), self.word_count - self.num_chars - 1)
        end = start + self.num_chars

        x = self.corpus_idx[start:end]
        y = self.corpus_idx[start+1:end+1]

        return torch.tensor(x), torch.tensor(y)

#搭建rnn神经网络
class TextGenerator(nn.Module):

    #初始化
    def __int__(self,unique_word_count):

        super().__init__()

        #词嵌入
        self.embedding = nn.Embedding(unique_word_count, 128)

        #rnn
        self.rnn = nn.RNN(128, 256 ,1)

        #全连接（输出层）
        self.fc = nn.Linear(256, unique_word_count)

    def forward (self,inputs,hidden):

        #初始化
        embd = self.embedding(inputs)

        #rnn处理
        out, hidden = self.rnn(embd.transpose(0,1), hidden)

        #输出层处理
        output = self.fc(output.reshape(shape=(-1, output.shape[-1])))

        #返回结果
        return output, hidden


    def init_hidden(self,bs):
        return torch.zeros(1,bs,256)

#训练模型
def train():

    unique_works, corpus_idx, unique_word_count, word_dict,  = build_vocab()

    lyrics = LyricDataset(corpus_idx,32)

    model = TextGenerator(unique_word_count)


    lyrics_dataloader = DataLoader(lyrics, batch_size=5, shuffle=True)

    # 5. 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 6. 定义优化器.
    ptimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 7. 模型训练.
    # 7.1 定义变量, 记录训练的轮数.
    epochs = 50
    # 7.2 具体的每轮训练动作.
    for epoch in range(epochs):     # epoch: 0, 1, 2, 3...9, 分别表示: 第1轮, 第2轮, ... 第10轮.
        # 7.3 定义变量记录: 本轮开始训练时间, 迭代(批次)次数, 训练总损失.
        start, iter_num, total_loss = time.time(), 0, 0.0
        # 7.4 具体的 本轮 各批次 训练动作.
        # 遍历数据集, 后台会调用 LyricsDataset#__getitem__()方法, 获取到每个样本的数据和标签,
        for x, y in lyrics_dataloader:
            #每次拿5条（batch_size），每条长32个索引

            # 7.5 获取隐藏层初始值.
            hidden = model.init_hidden(5)

            # 7.6 模型计算.
            output, hidden = model(x, hidden)

            # 7.7 计算损失.
            # y的形状: (batch 批次数, seq_len 句子长度, 词向量维度) -> 转成一维向量 -> 每个词的下标索引.
            # output形状为: (seq_len, batch, 词向量维度)
            y = torch.transpose(y, 0, 1).reshape(shape=(-1, ))
            loss = criterion(output, y)

            # 7.8 梯度清零 + 反向传播 + 更新参数.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 7.9 累计损失 和 迭代次数.
            total_loss += loss.item()
            iter_num += 1

        # 7.10 走到这里, 说明 本轮训练结束, 打印本轮的训练信息.
        print(f'epoch: {epoch + 1}, time: {time.time() - start:.2f}s, loss: {total_loss / iter_num:.4f}')

    # 8. 走到这里, 说明多轮训练结束(模型训练结束), 保存即可.
    torch.save(model.state_dict(), './model/text_generator.pth')

