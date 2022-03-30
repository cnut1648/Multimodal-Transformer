import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig

from .out import RegOut, ClfOut, AdaInRegOut, AdaInClfOut
from .resnet import ResNet2Plus1D
from .timesformer import TimeSformer


class ER_Model(nn.Module):
    """
    ER model for video, connect backbone encoder and out head
    """
    def __init__(self, backbone: DictConfig, out: DictConfig, **kwargs):
        super().__init__()

        self.backbone: torch.nn.Module = hydra.utils.instantiate(
            backbone, _recursive_=False
        )
        self.out: torch.nn.Module = hydra.utils.instantiate(
            out, _recursive_=False
        )

    def forward(self, videos, **kwargs):
        x = self.backbone(videos)
        x = self.out(x)
        return x


class PER_Model(nn.Module):
    def __init__(self, model_name, config):
        super().__init__()

        if config['arch'] == 'timesformer':
            self.dim = config['dim']
            self.image_size = config['image_size']
            self.patch_size = config['patch_size']
            self.num_frames = config['num_frames']
            self.depth = config['depth']
            self.heads = config['heads']
            self.dim_head = config['dim_head']
            self.attn_dropout = config['attn_dropout']
            self.ff_dropout = config['ff_dropout']

            self.model = TimeSformer(dim=self.dim, image_size=self.image_size,
                                     patch_size=self.patch_size, num_frames=self.num_frames,
                                     depth=self.depth, heads=self.heads, dim_head=self.dim_head,
                                     attn_dropout=self.attn_dropout, ff_dropout=self.ff_dropout)
            if config['task'] == 'reg':
                self.out = AdaInRegOut(dim=self.dim, dropout=self.ff_dropout)
            else:
                self.out = AdaInClfOut(
                    dim=self.dim, dropout=self.ff_dropout, num_classes=6)
        elif config['arch'] == 'resnet':
            self.dim = config['dim']
            self.dropout = config['dropout']
            self.model = ResNet2Plus1D(dropout_rate=self.dropout)
            if task == 'reg':
                self.out = AdaInRegOut(dim=self.dim, dropout=self.dropout)
            else:
                self.out = AdaInClfOut(
                    dim=self.dim, dropout=self.dropout, num_classes=6)
        else:
            sys.exit('%s not implemented' % model_name)

        # Encoder to generate AdaIN parameters
        num_adain_params = self.get_num_adain_params(self.out)  # 1024
        ks, pw = 3, 1
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=ks, padding=pw),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=ks, padding=pw),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(16, 64, kernel_size=ks, padding=pw),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(64, num_adain_params, kernel_size=ks, padding=pw),
            nn.AdaptiveAvgPool2d(1))

    def forward(self, x):
        mean, std = self.get_adain_params(x)
        self.assign_adain_params(mean, std, self.out)

        x = self.model(x)
        x = self.out(x)

        return x

    def get_adain_params(self, x):
        B, F, C, H, W = x.shape
        x = x.view(B*F, C, H, W)
        x = self.encoder(x).view(B, F, -1)
        mean = x.mean([1])
        std = x.std([1])

        return mean, std

    def assign_adain_params(self, adain_mean, adain_std, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                m.beta = adain_mean[:, :m.num_features]
                m.gamma = adain_std[:, :m.num_features]
                if adain_mean.size(1) > m.num_features:
                    adain_mean = adain_mean[:, m.num_features:]
                    adain_std = adain_std[:, m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += m.num_features
        return num_adain_params


class PersonEnc(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.task = config['task']
        self.dim = config['dim']
        self.dropout = config['dropout']

        ks, pw = 3, 1
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=ks, padding=pw),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=ks, padding=pw),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(16, 64, kernel_size=ks, padding=pw),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(64, self.dim, kernel_size=ks, padding=pw),
            nn.AdaptiveAvgPool2d(1))
        if self.task == 'reg':
            self.out = RegOut(dim=self.dim, dropout=self.dropout)
        else:
            self.out = ClfOut(
                dim=self.dim, dropout=self.dropout, num_classes=8)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)

        return x
