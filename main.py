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
import model
import train
import mydatasets


parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()


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


# load MR dataset
# def mr(text_field, label_field, **kargs):
def mr(tokenizer, text_transform, label_transform, **kargs):
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

    train_iter = DataLoader(train_data, batch_size=args.batch_size, shuffle=args.shuffle, collate_fn=collate_batch)
    dev_iter = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)  # CHANGED: 修改batch_size
    return train_iter, dev_iter, vocab


# load data
print("\nLoading data...")
from torchtext.data.utils import get_tokenizer
tokenizer = get_tokenizer('basic_english')
text_transform = None
label_transform = None
train_iter, dev_iter, vocab = mr(tokenizer, text_transform, label_transform, device=-1)
# train_iter, dev_iter, test_iter = sst(text_field, label_field, device=-1, repeat=False)


# update args and print
args.embed_num = len(vocab)
args.class_num = 2  # Assume binary classification for MR dataset
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
args.feature_dim = 64

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))


# model
cnn = model.CNN_Text(args)
if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()
        

# train or predict
if args.predict is not None:
    # label = train.predict(args.predict, cnn, text_field, label_field, args.cuda)
    label = train.predict(args.predict, cnn, tokenizer, vocab, args.cuda)
    # print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
    print(f"[Encoding] {args.predict} -> {label}")
elif args.test:
    try:
        train.eval(dev_iter, cnn, args) 
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else:
    print()
    try:
        train.train(train_iter, dev_iter, cnn, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')

