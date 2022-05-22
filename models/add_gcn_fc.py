import torch
import torch.nn as nn
import math
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


class DynamicGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_nodes):
        super(DynamicGraphConvolution, self).__init__()

        self.static_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2))
        self.static_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(in_features*2, num_nodes, 1)
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)

    def forward_static_gcn(self, x):
        x = self.static_adj(x.transpose(1, 2))
        x = self.static_weight(x.transpose(1, 2))
        return x

    def forward_construct_dynamic_graph(self, x):
        ### Model global representations ###
        x_glb = self.gap(x)
        x_glb = self.conv_global(x_glb)
        x_glb = self.bn_global(x_glb)
        x_glb = self.relu(x_glb)
        x_glb = x_glb.expand(x_glb.size(0), x_glb.size(1), x.size(2))
        
        ### Construct the dynamic correlation matrix ###
        x = torch.cat((x_glb, x), dim=1)
        dynamic_adj = self.conv_create_co_mat(x)
        dynamic_adj = torch.sigmoid(dynamic_adj)
        return dynamic_adj

    def dynamic_adj(self, dynamic_adj):


    def forward_dynamic_gcn(self, x, dynamic_adj):

        x = torch.matmul(x, dynamic_adj)
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        """ D-GCN module

        Shape: 
        - Input: (B, C_in, N) # C_in: 1024, N: num_classes
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        out_static = self.forward_static_gcn(x)
        x = x + out_static # residual
        dynamic_adj = self.forward_construct_dynamic_graph(x)
        x = self.forward_dynamic_gcn(x, dynamic_adj)
        return x


class ADD_GCN_FC(nn.Module):
    def __init__(self, model, num_classes):
        super(ADD_GCN_FC, self).__init__()
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
        self.num_classes = num_classes

        self.fc = nn.Conv2d(model.fc.in_features, num_classes, (1,1), bias=False)

        self.conv_transform = nn.Conv2d(2048, 1024, (1,1))
        self.relu = nn.LeakyReLU(0.2)

        self.gcn = DynamicGraphConvolution(1024, 1024, num_classes)
        self.last_linear = GroupWiseLinear(num_class=num_classes, hidden_dim=1024)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward_feature(self, x):
        x = self.features(x)
        return x

    def forward_classification_sm(self, x):
        """ Get another confident scores {s_m}.

        Shape:
        - Input: (B, C_in, H, W) # C_in: 2048
        - Output: (B, C_out) # C_out: num_classes
        """
        x = self.fc(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.topk(1, dim=-1)[0].mean(dim=-1)
        return x

    def forward_sam(self, x):
        """ SAM module

        Shape: 
        - Input: (B, C_in, H, W) # C_in: 2048
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        mask = self.fc(x) 
        mask = mask.view(mask.size(0), mask.size(1), -1) 
        mask = torch.sigmoid(mask)
        mask = mask.transpose(1, 2)

        x = self.conv_transform(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.matmul(x, mask)
        return x

    def forward_dgcn(self, x):
        x = self.gcn(x)
        return x

    def forward(self, x):
        x = self.forward_feature(x)

        out1 = self.forward_classification_sm(x)

        v = self.forward_sam(x) # B*1024*num_classes
        z = self.forward_dgcn(v)
        z = v + z

        out2 = self.last_linear(z.transpose(1, 2))
        return out1, out2

    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.features.parameters()))
        large_lr_layers = filter(lambda p:id(p) not in small_lr_layers, self.parameters())
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': large_lr_layers, 'lr': lr},
                ]

