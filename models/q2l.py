from pyexpat import model
from matplotlib import image
# from natsort import decoder
from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
try:
    from .trans_utils.position_encoding import build_position_encoding
    from .trans_utils.transformer import build_transformer
except ImportError:
    from trans_utils.position_encoding import build_position_encoding
    from trans_utils.transformer import build_transformer


class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d        
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x

class GraphConvolution(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GraphConvolution, self).__init__()
        self.relu = nn.LeakyReLU(0.2)
        self.weight = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, adj, nodes):
        nodes = torch.matmul(nodes, adj)
        nodes = self.relu(nodes)
        nodes = self.weight(nodes)
        nodes = self.relu(nodes)
        return nodes


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Q2L(nn.Module):
    def __init__(self, model, num_classes):
        super(Q2L, self).__init__()
        # backbone
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

        # hyper-parameters
        self.num_classes = num_classes
        self.transformer_dim = 2048
        self.num_queries = num_classes
        self.n_head = 4
        self.num_encoder_layers = 1
        self.num_decoder_layers = 2

        # self.fc = nn.Conv2d(model.fc.in_features, num_classes, (1,1), bias=False)
        self.conv_transform = nn.Conv2d(model.fc.in_features, self.transformer_dim, (1,1))

        # transformer decoder
        self.query_embed = nn.Embedding(self.num_queries, self.transformer_dim)
        self.positional_embedding = build_position_encoding(hidden_dim=self.transformer_dim, mode='learned')
        self.transformer = build_transformer(d_model=self.transformer_dim, nhead=self.n_head,
                                             num_encoder_layers=self.num_encoder_layers,
                                             num_decoder_layers=self.num_decoder_layers)
        # GroupFC
        self.GroupFC = GroupWiseLinear(self.num_classes, self.transformer_dim, bias=True)


    def forward_feature(self, x):
        x = self.features(x)
        return x


    def forward(self, x):
        # backbone
        src = self.forward_feature(x)
        # transformer_encode_decode
        src=self.conv_transform(src)
        pos = self.positional_embedding(src)
        hc = self.transformer(src, mask=None, query_embed=self.query_embed.weight, pos_embed=pos)[0]
        # Gourp_FC
        out = self.GroupFC(hc[-1])
        return out

    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.features.parameters()))
        large_lr_layers = filter(lambda p:id(p) not in small_lr_layers, self.parameters())
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': large_lr_layers, 'lr': lr},
                ]

    def freeze_bn(self):
        print("Freezing bn.")
        for layer in self.features:
            for m in layer.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def freeze(self, layer):
        if layer == 'bn':
            self.freeze_bn()
        else:
            raise NotImplementedError

def main():
    import torchvision
    res101 = torchvision.models.resnet101(pretrained=True)
    image = torch.randn(4, 3, 448, 448)
    model = Q2L(model=res101, num_classes=80)
    decoder = model(image)
    pass

if __name__ == '__main__':
    # print("==")
    # import torch
    # import numpy as np
    # input = [1, 2, 3, 4]
    # # output = torch.Tensor(input, dtype=torch.int16)# .cuda()
    # output = torch.from_numpy(np.array(input, dtype=np.int16)).cuda()

    # print(output)
    main()

