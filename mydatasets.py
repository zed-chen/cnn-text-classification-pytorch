import re
import os
import random
import tarfile
import urllib
# from torchtext import data
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer  # CHANGED
from torchtext.vocab import build_vocab_from_iterator  # CHANGED


# class TarDataset(data.Dataset):
class TarDataset(Dataset):
    """Defines a Dataset loaded from a downloadable tar archive.

    Attributes:
        url: URL where the tar archive can be downloaded.
        filename: Filename of the downloaded tar archive.
        dirname: Name of the top-level directory within the zip archive that
            contains the data files.
    """

    @classmethod
    def download_or_unzip(cls, root):
        path = os.path.join(root, cls.dirname)
        if not os.path.isdir(path):
            tpath = os.path.join(root, cls.filename)
            if not os.path.isfile(tpath):
                print('downloading')
                urllib.request.urlretrieve(cls.url, tpath)
            with tarfile.open(tpath, 'r') as tfile:
                print('extracting')
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tfile, root)
        return os.path.join(path, '')


class MR(TarDataset):

    url = 'https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
    filename = 'rt-polaritydata.tar.gz'
    dirname = 'rt-polaritydata'

    # @staticmethod
    # def sort_key(ex):
    #     return len(ex.text)

    # def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):
    def __init__(self, root='.', split='train', vocab=None, tokenizer=None, dev_ratio=0.1, shuffle=True, random_seed=42):
        """Create an MR dataset instance given a path and fields.

        Arguments:
            root: Root directory to download and store the dataset.
            split: Split to load ('train' or 'dev').
            vocab: Vocabulary object. If None, a new vocabulary is built.
            tokenizer: Tokenizer function. If None, a simple whitespace tokenizer is used.
            dev_ratio: Ratio of training data to use for validation.
            shuffle: Whether to shuffle the data before splitting.
            random_seed: Random seed for shuffling.
        """
        super().__init__()
        self.root = root
        self.split = split
        self.path = self.download_or_unzip(root)

        def clean_str(string):
            """
            文本清洗函数，用于处理文本数据中的特殊字符和格式问题。
            Tokenization/string cleaning for all datasets except for SST.
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip()
        
        # 设置分词器
        if tokenizer is None:
            self.tokenizer = lambda x: clean_str(x).split()  # CHANGED: 移除了text_field相关逻辑
        else:
            self.tokenizer = tokenizer
        
        # 加载数据
        self.examples = []
        label_map = {'neg': 0, 'pos': 1}  # CHANGED: 使用数字标签而非字符串

        # 读取negative样本
        with open(os.path.join(self.path, 'rt-polarity.neg'), 'r', encoding='latin-1') as f:  # CHANGED: 指定编码为latin-1
            neg_examples = [(line.strip(), 0) for line in f]
        # 读取positive样本
        with open(os.path.join(self.path, 'rt-polarity.pos'), 'r', encoding='latin-1') as f:  # CHANGED: 指定编码为latin-1
            pos_examples = [(line.strip(), 1) for line in f]
        
        # 合并样本
        self.examples = neg_examples + pos_examples

        # 分割训练集和验证集
        random.seed(random_seed)
        if shuffle:
            random.shuffle(self.examples)
        
        dev_size = int(len(self.examples) * dev_ratio)
        if split == 'train':
            self.examples = self.examples[dev_size:]
        elif split == 'dev':
            self.examples = self.examples[:dev_size]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'dev'.")

        # 构建词汇表
        if vocab is None:
            def yield_tokens(data_iter):
                for text, _ in data_iter:
                    yield self.tokenizer(text)
            self.vocab = build_vocab_from_iterator(yield_tokens(self.examples), specials=['<pad>', '<unk>']) # CHANGED: 使用新的词汇表构建方法
            self.vocab.set_default_index(self.vocab['<unk>']) # CHANGED: 设置未知词索引
        else:
            self.vocab = vocab

    def __len__(self):  # CHANGED: 新增方法，使数据集可获取长度
        return len(self.examples)

    def __getitem__(self, idx):  # CHANGED: 新增方法，使数据集可索引
        text, label = self.examples[idx]
        tokens = self.tokenizer(text)
        token_ids = self.vocab(tokens)
        return token_ids, label

    def get_vocab(self):  # CHANGED: 新增辅助方法
        return self.vocab

    @classmethod
    def get_tokenizer(cls):  # CHANGED: 新增辅助方法
        """获取用于该数据集的分词器"""
        def clean_str(string):
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip()
        return lambda x: clean_str(x).split()    
    