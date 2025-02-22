import logging
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import spacy
from tqdm import tqdm
import re
import time
import os


class Vocab():
    """
    from collections import Counter
    counter = Counter()
    data_iter = [
    "hello world",
    "hello from the other side",
    "hello again",
    "world from the other side"
    ]
    for string_ in data_iter:
        counter.update(string_.split())
    print(counter) # Counter({'hello': 3, 'world': 2, 'from': 2, 'the': 2, 'other': 2, 'side': 2, 'again': 1})
    vocab = Vocab(counter,min_freq=2)
    print(vocab['hello']) # 4
    print(vocab.itos) # ['<unk>', '<pad>', '<bos>', '<eos>', 'hello', 'world', 'from', 'the', 'other', 'side']
    print(vocab['ok']) # 0
    """

    def __init__(self, counter=None, specials=None, min_freq=1):
        """

        Args:
            counter:
            specials: 需要把 <unk> 放在最前面
            min_freq:
        """

        if specials is None:
            specials = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.specials = specials
        self.stoi = {k: v for k, v in zip(specials, range(len(specials)))}
        self.itos = specials[:]

        for c in counter.most_common():
            if c[1] >= min_freq:
                self.itos.append(c[0])
                self.stoi[c[0]] = len(self.itos) - 1

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(self.specials[0]))

    def __len__(self):
        return len(self.itos)


def my_tokenizer():
    # pip install spacy==3.8.0
    # 下载如下两个文件
    # https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.8.0/de_core_news_sm-3.8.0.tar.gz
    # https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0.tar.gz
    # pip install de_core_news_sm-3.8.0.tar.gz
    # pip install en_core_news_sm-3.8.0.tar.gz
    #
    # e.g.
    # text = "Das ist ein Beispiel für die Tokenisierung."
    # tokenizer = my_tokenizer()
    # r = tokenizer['de'](text) # ['Das', 'ist', 'ein', 'Beispiel', 'für', 'die', 'Tokenisierung', '.']

    tokenizer = {}
    de_tokenizer = spacy.load("de_core_news_sm")  # 德语
    en_tokenizer = spacy.load("en_core_web_sm")  # 英语
    tokenizer['de'] = (lambda s: [token.text for token in de_tokenizer(s)])
    tokenizer['en'] = (lambda s: [token.text for token in en_tokenizer(s)])
    return tokenizer


def couple_tokenizer(text):
    """
    tokenizer方法
    :param text: 上联：一夜春风去，怎么对下联？
    :return:
    words: 字粒度： ['上', '联', '：', '一', '夜', '春', '风', '去', '，', '怎', '么', '对', '下', '联', '？']
    """
    if contains_chinese(text):  # 中文
        text = " ".join(text)  # 不分词则是字粒度
    words = text.split()
    return words


def build_vocab(tokenizer, filepath, specials=None, min_freq=1):
    """
    vocab = Vocab(counter, specials=specials)

    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    # ['<unk>', '<pad>', '<bos>', '<eos>', '.', 'a', 'are', 'A', 'Two', 'in', 'men',...]
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；

    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    # {'<unk>': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3, '.': 4, 'a': 5, 'are': 6,...}
    print(vocab.stoi['are'])  # 通过单词返回得到词表中对应的索引
    """
    counter = Counter()
    logging.info(f"# 正在构建词表: {filepath}")
    with open(filepath, encoding='utf8') as f:
        for string_ in tqdm(f):
            counter.update(tokenizer(string_))
        return Vocab(counter, specials, min_freq=min_freq)


def contains_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fff]')  # 匹配中文字符的正则表达式
    return bool(re.search(pattern, text))


def pad_sequence2(sequences, batch_first=False, max_len=None, padding_value=0):
    """
    对一个List中的元素进行padding
    Pad a list of variable length Tensors with ``padding_value``
    a = torch.ones(25)
    b = torch.ones(22)
    c = torch.ones(15)
    pad_sequence2([a, b, c],max_len=None).size()
    torch.Size([25, 3])
        sequences:
        batch_first: 是否把batch_size放到第一个维度
        padding_value:
        max_len :
                当max_len = 50时，表示以某个固定长度对样本进行padding，多余的截掉；
                当max_len=None是，表示以当前batch中最长样本的长度对其它进行padding；
    Returns:
    """
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        if tensor.size(0) < max_len:
            padding_content = [padding_value] * (max_len - tensor.size(0))
            tensor = torch.cat([tensor, torch.tensor(padding_content)], dim=0)
        else:
            tensor = tensor[:max_len]
        out_tensors.append(tensor)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensors


def process_cache(unique_key=None):
    """
    数据预处理结果缓存修饰器
    :param : unique_key
    :return:
    """
    if unique_key is None:
        raise ValueError(
            "unique_key 不能为空, 请指定相关数据集构造类的成员变量，如['min_freq', 'cut_words', 'max_sen_len']")

    def decorating_function(func):
        def wrapper(*args, **kwargs):
            logging.info(f" ## 索引预处理缓存文件的参数为：{unique_key}")
            obj = args[0]  # 获取类对象，因为data_process(self, file_path=None)中的第1个参数为self
            file_path = kwargs['file_path']
            file_dir = f"{os.sep}".join(file_path.split(os.sep)[:-1])
            file_name = "".join(file_path.split(os.sep)[-1].split('.')[:-1])
            paras = f"cache_{file_name}_"
            for k in unique_key:
                paras += f"{k}{obj.__dict__[k]}_"  # 遍历对象中的所有参数
            cache_path = os.path.join(file_dir, paras[:-1] + '.pt')
            start_time = time.time()
            if not os.path.exists(cache_path):
                logging.info(f"缓存文件 {cache_path} 不存在，重新处理并缓存！")
                data = func(*args, **kwargs)
                with open(cache_path, 'wb') as f:
                    torch.save(data, f)
            else:
                logging.info(f"缓存文件 {cache_path} 存在，直接载入缓存文件！")
                with open(cache_path, 'rb') as f:
                    data = torch.load(f,weights_only=True)
            end_time = time.time()
            logging.info(f"数据预处理一共耗时{(end_time - start_time):.3f}s")
            return data

        return wrapper

    return decorating_function


class LoadEnglishGermanDataset():
    def __init__(self, train_file_paths=None, tokenizer=None,
                 batch_size=2, min_freq=1):
        # 根据训练预料建立英语和德语各自的字典
        specials = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.tokenizer = tokenizer()
        self.de_vocab = build_vocab(self.tokenizer['de'], filepath=train_file_paths[0],
                                    specials=specials, min_freq=min_freq)
        self.en_vocab = build_vocab(self.tokenizer['en'], filepath=train_file_paths[1],
                                    specials=specials, min_freq=min_freq)
        self.PAD_IDX = self.de_vocab['<pad>']
        self.BOS_IDX = self.de_vocab['<bos>']
        self.EOS_IDX = self.de_vocab['<eos>']
        self.batch_size = batch_size

    def data_process(self, filepaths):
        """
        将每一句话中的每一个词根据字典转换成索引的形式
        :param filepaths:
        :return:
        """
        raw_de_iter = iter(open(filepaths[0], encoding="utf8"))
        raw_en_iter = iter(open(filepaths[1], encoding="utf8"))
        data = []
        logging.info(f"### 正在将数据集 {filepaths} 转换成 Token ID ")
        for (raw_de, raw_en) in tqdm(zip(raw_de_iter, raw_en_iter), ncols=80):
            de_tensor_ = torch.tensor([self.de_vocab[token] for token in
                                       self.tokenizer['de'](raw_de.rstrip("\n"))], dtype=torch.long)
            en_tensor_ = torch.tensor([self.en_vocab[token] for token in
                                       self.tokenizer['en'](raw_en.rstrip("\n"))], dtype=torch.long)
            data.append((de_tensor_, en_tensor_))
        # [ (tensor([ 9, 37, 46,  5, 42, 36, 11, 16,  7, 33, 24, 45, 13,  4]), tensor([ 8, 45, 11, 13, 28,  6, 34, 31, 30, 16,  4])),
        #   (tensor([22,  5, 40, 25, 30,  6, 12,  4]), tensor([12, 10,  9, 22, 23,  6, 33,  5, 20, 37, 41,  4])),
        #   (tensor([ 8, 38, 23, 39,  7,  6, 26, 29, 19,  4]), tensor([ 7, 27, 21, 18, 24,  5, 44, 35,  4])),
        #   (tensor([ 8, 21,  7, 34, 32, 17, 44, 28, 35, 20, 10, 41,  6, 15,  4]), tensor([ 7, 29,  9,  5, 15, 38, 25, 39, 32,  5, 26, 17,  5, 43,  4])),
        #   (tensor([ 9,  5, 43, 27, 18, 10, 31, 14, 47,  4]), tensor([ 8, 10,  6, 14, 42, 40, 36, 19,  4]))  ]

        return data

    def load_train_val_test_data(self, train_file_paths, val_file_paths, test_file_paths):
        train_data = self.data_process(train_file_paths)
        val_data = self.data_process(val_file_paths)
        test_data = self.data_process(test_file_paths)
        train_iter = DataLoader(train_data, batch_size=self.batch_size,
                                shuffle=True, collate_fn=self.generate_batch)
        valid_iter = DataLoader(val_data, batch_size=self.batch_size,
                                shuffle=False, collate_fn=self.generate_batch)
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=False, collate_fn=self.generate_batch)
        return train_iter, valid_iter, test_iter

    def generate_batch(self, data_batch):
        """
        自定义一个函数来对每个batch的样本进行处理，该函数将作为一个参数传入到类DataLoader中。
        由于在DataLoader中是对每一个batch的数据进行处理，所以这就意味着下面的pad_sequence操作，最终表现出来的结果就是
        不同的样本，padding后在同一个batch中长度是一样的，而在不同的batch之间可能是不一样的。因为pad_sequence是以一个batch中最长的
        样本为标准对其它样本进行padding
        :param data_batch:
        :return:
        """
        de_batch, en_batch = [], []
        for (de_item, en_item) in data_batch:  # 开始对一个batch中的每一个样本进行处理。
            de_batch.append(de_item)  # 编码器输入序列不需要加起止符
            # 在每个idx序列的首位加上 起始token 和 结束 token
            en = torch.cat([torch.tensor([self.BOS_IDX]), en_item, torch.tensor([self.EOS_IDX])], dim=0)
            en_batch.append(en)
        # 以最长的序列为标准进行填充
        de_batch = pad_sequence(de_batch, padding_value=self.PAD_IDX)  # [de_len,batch_size]
        en_batch = pad_sequence(en_batch, padding_value=self.PAD_IDX)  # [en_len,batch_size]
        return de_batch, en_batch

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, src, tgt, device='cpu'):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, device)  # [tgt_len,tgt_len]
        # Decoder的注意力Mask输入，用于掩盖当前position之后的position，所以这里是一个对称矩阵

        src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
        # Encoder的注意力Mask输入，这部分其实对于Encoder来说是没有用的，所以这里全是0

        src_padding_mask = (src == self.PAD_IDX).transpose(0, 1)
        # False表示not masked, True表示masked
        # 用于mask掉Encoder的Token序列中的padding部分,[batch_size, src_len]
        tgt_padding_mask = (tgt == self.PAD_IDX).transpose(0, 1)
        # 用于mask掉Decoder的Token序列中的padding部分,batch_size, tgt_len
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


class LoadCoupletDataset():
    def __init__(self, train_file_paths=None, batch_size=2, min_freq=1, max_len=None):
        # 根据训练预料建立字典，由于都是中文，所以共用一个即可
        specials = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.tokenizer = couple_tokenizer
        self.vocab = build_vocab(self.tokenizer, train_file_paths[0], specials, min_freq)
        self.PAD_IDX = self.vocab['<pad>']
        self.BOS_IDX = self.vocab['<bos>']
        self.EOS_IDX = self.vocab['<eos>']
        self.min_freq = min_freq
        self.max_len = max_len
        self.batch_size = batch_size

    def load_raw_data(self, file_paths=None):
        """
        载入原始的文本
        :param file_paths:
        :return:
        results 里面有两个个元素，分别为一个list，即所有上联集合，和所有下联集合
        [['晚 风 摇 树 树 还 挺','愿 景 天 成 无 墨 迹 '],['晨 露 润 花 花 更 红','万 方 乐 奏 有 于 阗 ' ]]
        """
        results = []
        for i in range(2):
            logging.info(f" ## 载入原始文本 {file_paths[i]}")
            tmp = []
            with open(file_paths[i], encoding="utf8") as f:
                for line in f:
                    line = line.strip()
                    tmp.append(line)
            results.append(tmp)
        return results

    @process_cache(["min_freq", "max_len"])
    def data_process(self, filepaths, file_path=None):
        """
        将每一句话中的每一个词根据字典转换成索引的形式
        :param filepaths:
        :return:
        """
        results = self.load_raw_data(filepaths)
        data = []
        for (raw_in, raw_out) in zip(results[0], results[1]):
            logging.debug(f"原始上联: {raw_in}")
            in_tensor_ = torch.tensor([self.vocab[token] for token in
                                       couple_tokenizer(raw_in.rstrip("\n"))], dtype=torch.long)
            if len(in_tensor_) < 6:
                logging.debug(f"长度过短，忽略: {raw_in}<=>{raw_out}")
                continue
            logging.debug(f"原始上联 token id: {in_tensor_}")
            logging.debug(f"原始下联: {raw_out}")
            out_tensor_ = torch.tensor([self.vocab[token] for token in
                                        couple_tokenizer(raw_out.rstrip("\n"))], dtype=torch.long)
            logging.debug(f"原始下联 token id: {out_tensor_}")

            data.append((in_tensor_, out_tensor_))
        return data

    def load_train_val_test_data(self, train_file_paths, test_file_paths):
        train_data = self.data_process(train_file_paths, file_path=train_file_paths[0])
        test_data = self.data_process(test_file_paths, file_path=test_file_paths[0])
        train_iter = DataLoader(train_data, batch_size=self.batch_size,
                                shuffle=True, collate_fn=self.generate_batch)
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=True, collate_fn=self.generate_batch)
        return train_iter, test_iter

    def generate_batch(self, data_batch):
        """
        自定义一个函数来对每个batch的样本进行处理，该函数将作为一个参数传入到类DataLoader中。
        由于在DataLoader中是对每一个batch的数据进行处理，所以这就意味着下面的pad_sequence2操作，最终表现出来的结果就是
        不同的样本，padding后在同一个batch中长度是一样的，而在不同的batch之间可能是不一样的。因为pad_sequence2是以一个batch中最长的
        样本为标准对其它样本进行padding
        :param data_batch:
        :return:
        """
        in_batch, out_batch = [], []
        for (in_item, out_item) in data_batch:  # 开始对一个batch中的每一个样本进行处理。
            in_batch.append(in_item)  # 编码器输入序列不需要加起止符
            # 在每个idx序列的首位加上 起始token 和 结束 token
            out = torch.cat([torch.tensor([self.BOS_IDX]), out_item, torch.tensor([self.EOS_IDX])], dim=0)
            out_batch.append(out)
        # 以最长的序列为标准进行填充
        in_batch = pad_sequence2(in_batch, max_len=self.max_len,
                                 padding_value=self.PAD_IDX)  # [de_len,batch_size]
        out_batch = pad_sequence2(out_batch, max_len=self.max_len,
                                  padding_value=self.PAD_IDX)  # [en_len,batch_size]
        return in_batch, out_batch

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, src, tgt, device='cpu'):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, device)  # [tgt_len,tgt_len]
        # Decoder的注意力Mask输入，用于掩盖当前position之后的position，所以这里是一个对称矩阵

        src_mask = torch.zeros((src_seq_len, src_seq_len), device=device)
        # Encoder的注意力Mask输入，这部分其实对于Encoder来说是没有用的，所以这里全是0

        src_padding_mask = (src == self.PAD_IDX).transpose(0, 1)
        # 用于mask掉Encoder的Token序列中的padding部分,[batch_size, src_len]
        tgt_padding_mask = (tgt == self.PAD_IDX).transpose(0, 1)
        # 用于mask掉Decoder的Token序列中的padding部分,batch_size, tgt_len
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def make_inference_sample(self, text):
        tokens = [self.vocab.stoi[token] for token in couple_tokenizer(text)]
        num_tokens = len(tokens)
        src = (torch.LongTensor(tokens).reshape(num_tokens, 1))
        return src
