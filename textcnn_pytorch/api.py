#! /usr/bin/env python
import os
import argparse
import datetime
import torch
# import torchtext.data as data
# import torchtext.datasets as datasets
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset  # CHANGED: 从torchtext.data.functional导入
from torch.utils.data import DataLoader
import torch.nn as nn
from . import model
from . import train
from . import mydatasets
from torchtext.data.utils import get_tokenizer



# # load SST dataset
# def sst(text_field, label_field,  **kargs):
#     train_data, dev_data, test_data = datasets.SST.splits(text_field, label_field, fine_grained=True)
#     text_field.build_vocab(train_data, dev_data, test_data)
#     label_field.build_vocab(train_data, dev_data, test_data)
#     train_iter, dev_iter, test_iter = data.BucketIterator.splits(
#                                         (train_data, dev_data, test_data), 
#                                         batch_sizes=(args.batch_size, 
#                                                      len(dev_data), 
#                                                      len(test_data)),
#                                         **kargs)
#     return train_iter, dev_iter, test_iter 

# TODO: 简化workflow，不要每次都要load数据

class TextCnnModel:
    def __init__(self,lr=0.001, epochs=256, batch_size=64, log_interval=1, test_interval=100, save_interval=500, save_dir='snapshot', early_stop=1000, save_best=True, shuffle=False, dropout=0.5, max_norm=3.0, embed_dim=128,
        kernel_num=100, kernel_sizes='3,4,5', static=False, device=-1, no_cuda=False, snapshot=None, feature_dim=64):
        self.name = 'textcnn'
        parser = argparse.ArgumentParser(description='CNN text classificer')
        self.args = parser.parse_args()
        self.args.lr = lr
        self.args.epochs = epochs
        self.args.batch_size = batch_size
        self.args.log_interval = log_interval
        self.args.test_interval = test_interval
        self.args.save_interval = save_interval
        self.args.save_dir = save_dir
        self.args.early_stop = early_stop
        self.args.save_best = save_best
        self.args.shuffle = shuffle
        self.args.dropout = dropout
        self.args.max_norm = max_norm
        self.args.embed_dim = embed_dim
        self.args.kernel_num = kernel_num
        self.args.kernel_sizes = kernel_sizes
        self.args.static = static
        self.args.device = device
        self.args.no_cuda = no_cuda
        self.args.snapshot = snapshot

        self.tokenizer = get_tokenizer('basic_english')
        self.text_transform = None
        self.label_transform = None
        self.train_iter, self.dev_iter, self.vocab = self._mr(self.tokenizer, self.text_transform, self.label_transform, device=-1)

        self.args.embed_num = len(self.vocab)
        self.args.class_num = 2  # Assume binary classification for MR dataset
        self.args.cuda = (not self.args.no_cuda) and torch.cuda.is_available(); del self.args.no_cuda
        self.args.kernel_sizes = [int(k) for k in self.args.kernel_sizes.split(',')]
        self.args.save_dir = os.path.join(self.args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.args.feature_dim = feature_dim

        self.cnn = model.CNN_Text(self.args)
        if self.args.snapshot is not None:
            self.cnn.load_state_dict(torch.load(self.args.snapshot))
        if self.args.cuda:
            torch.cuda.set_device(self.args.device)
            self.cnn = self.cnn.cuda()
    
    def run(self, predict=None, test=False):
        self.args.predict = predict
        self.args.test = test

        if self.args.predict is not None:
            return train.predict(self.args.predict, self.cnn, self.tokenizer, self.vocab, self.args.cuda)
        elif self.args.test:
            try:
                train.eval(self.dev_iter, self.cnn, self.args)
            except Exception as e:
                print("\nSorry. The test dataset doesn't  exist.\n")
        else:
            print()
            try:
                train.train(self.train_iter, self.dev_iter, self.cnn, self.args)
            except KeyboardInterrupt:
                print('\n' + '-' * 89)
                print('Exiting from training early')


    # load MR dataset
    # def mr(text_field, label_field, **kargs):
    def _mr(self,tokenizer, text_transform, label_transform, **kargs):
        train_data = mydatasets.MR(root='.', split='train')
        dev_data = mydatasets.MR(root='.', split='dev', vocab=train_data.get_vocab())
        # CHANGED: 不再需要to_map_style_dataset，因为新版MR类已经是map-style
        # train_data, dev_data = mydatasets.MR.splits(text_field, label_field)
        # text_field.build_vocab(train_data, dev_data)

        # CHANGED: 使用train_data的迭代器
        def yield_tokens(data_iter):
            for text, _ in data_iter:
                yield tokenizer(text)
        
        # CHANGED: 直接使用train_data的vocab
        vocab = train_data.get_vocab()
        
        def collate_batch(batch):
            label_list, text_list, offsets = [], [], [0]
            for text, label in batch:  # CHANGED: 直接从batch中获取text和label
                label_list.append(label)
                processed_text = torch.tensor(text, dtype=torch.int64)  # CHANGED: 直接使用text，不需要再转换
                text_list.append(processed_text)
                offsets.append(processed_text.size(0))
            label_list = torch.tensor(label_list, dtype=torch.int64)
            offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
            # 使用torch.stack而不是torch.cat来保持批次维度
            text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
            return label_list, text_list, offsets

        train_iter = DataLoader(train_data, batch_size=self.args.batch_size, shuffle=self.args.shuffle, collate_fn=collate_batch)
        dev_iter = DataLoader(dev_data, batch_size=self.args.batch_size, shuffle=False, collate_fn=collate_batch)  # CHANGED: 修改batch_size
        return train_iter, dev_iter, vocab

