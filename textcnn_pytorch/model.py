import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable


class CNN_Text(nn.Module):
    
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        feature_dim = args.feature_dim  # 新增：特征向量维度

        self.embed = nn.Embedding(V, D)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)

        # 新增：固定长度的特征层
        self.feature_layer = nn.Linear(len(Ks) * Co, feature_dim)
        # self.fc1 = nn.Linear(len(Ks) * Co, C)
        self.fc1 = nn.Linear(feature_dim, C)

        if self.args.static:
            self.embed.weight.requires_grad = False

    def forward(self, x, offsets=None, return_features=False): # CHANGED: 添加offsets参数
        x = self.embed(x)  # (N, W, D)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)

        # 生成固定长度的特征向量
        features = self.feature_layer(x)  # [batch_size, feature_dim]

        logit = self.fc1(features)  # (N, C)

        if return_features:
            return logit, features.detach().cpu().tolist()[0]
        else:
            return logit
