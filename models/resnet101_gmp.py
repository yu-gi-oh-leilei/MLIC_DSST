import torch
import torch.nn as nn

class ResNet101_GMP(nn.Module):
    def __init__(self, model, num_classes):
        super(ResNet101_GMP, self).__init__()
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

        self.pool = nn.AdaptiveMaxPool2d((1,1))

        self.fc = nn.Linear(2048, num_classes)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward_feature(self, x):
        x = self.features(x)
        return x
   
    def forward(self, x):
        feature = self.forward_feature(x)
        feature = self.pool(feature).view(-1, 2048)
        out = self.fc(feature)
        
        return out

    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.features.parameters()))
        large_lr_layers = filter(lambda p:id(p) not in small_lr_layers, self.parameters())
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': large_lr_layers, 'lr': lr},
                ]

    def get_config_encoder_optim(self, lrp):
        small_lr_layers = list(map(id, self.features.parameters()))
        return [{'params': self.features.parameters(), 'lr': lrp},]

    def get_config_decoder_optim(self, lr):
        small_lr_layers = list(map(id, self.features.parameters()))
        large_lr_layers = filter(lambda p:id(p) not in small_lr_layers, self.parameters())
        return [{'params': large_lr_layers, 'lr': lr},]


    def freeze_bn(self):
        print("Freezing bn.")
        for layer in self.features:
            for m in layer.modules():
                if isinstance(m, nn.BatchNorm2d):
                    # print(' === Freezing bn === ')
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def freeze(self, layer):
        if layer == 'bn':
            self.freeze_bn()
        else:
            raise NotImplementedError

