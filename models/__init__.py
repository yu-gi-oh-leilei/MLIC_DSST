import torch
import torchvision
from .add_gcn import ADD_GCN
from .add_gcn_fc import ADD_GCN_FC
from .resnet101_gmp import ResNet101_GMP
from .resnet101_gap import ResNet101_GAP
from .q2l import Q2L
model_dict = {'ADD_GCN': ADD_GCN, 'ADD_GCN_FC' :ADD_GCN_FC, 'ResNet101_GMP': ResNet101_GMP, 'ResNet101_GAP': ResNet101_GAP, 'Q2L':Q2L}

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

def get_model(num_classes, args):
    if args.model_name in ('ADD_GCN'):
        # name = 'resnet101'
        # dilation = False
        # res101 = getattr(torchvision.models, name)(
        #         replace_stride_with_dilation=[False, False, dilation],
        #         pretrained=True,
        #         norm_layer=FrozenBatchNorm2d)
        res101 = torchvision.models.resnet101(pretrained=True)
        model = model_dict[args.model_name](res101, num_classes)
        return model

    elif args.model_name == 'ResNet101_GAP':
        # name = 'resnet101'
        # dilation = False
        # res101 = getattr(torchvision.models, name)(
        #         replace_stride_with_dilation=[False, False, dilation],
        #         pretrained=True,
        #         norm_layer=FrozenBatchNorm2d)
        res101 = torchvision.models.resnet101(pretrained=True)
        model = model_dict[args.model_name](res101, num_classes)
        return model

    elif args.model_name == 'ResNet101_GMP':
        # name = 'resnet101'
        # dilation = False
        # res101 = getattr(torchvision.models, name)(
        #         replace_stride_with_dilation=[False, False, dilation],
        #         pretrained=True,
        #         norm_layer=FrozenBatchNorm2d)
        res101 = torchvision.models.resnet101(pretrained=True)
        model = model_dict[args.model_name](res101, num_classes)
        return model

    elif args.model_name in ('ADD_GCN_FC'):
        # name = 'resnet101'
        # dilation = False
        # res101 = getattr(torchvision.models, name)(
        #         replace_stride_with_dilation=[False, False, dilation],
        #         pretrained=True,
        #         norm_layer=FrozenBatchNorm2d)
        res101 = torchvision.models.resnet101(pretrained=True)
        model = model_dict[args.model_name](res101, num_classes)
        return model

    elif args.model_name in ('Q2L'):
        # name = 'resnet101'
        # dilation = False
        # res101 = getattr(torchvision.models, name)(
        #         replace_stride_with_dilation=[False, False, dilation],
        #         pretrained=True,
        #         norm_layer=FrozenBatchNorm2d)
        res101 = torchvision.models.resnet101(pretrained=True)
        model = model_dict[args.model_name](res101, num_classes)
        return model