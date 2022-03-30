from typing import List
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

class VideoModel(nn.Module):
    @property
    def hidden_size(self) -> int:
        pass
    @property
    def blocks(self) -> List[nn.Module]:
        pass
    def replace_linear(self, cls: nn.Module, num_out: int):
        """
        ordinal regression, replace last linear
        """
        pass

class MetaTimesformer(VideoModel):
    def __init__(self, img_size=224, patch_size=16, num_classes=400, num_frames=8, attention_type='divided_space_time',  pretrained_model='', **kwargs):
        super().__init__()
        from timesformer.models.vit import TimeSformer
        model = TimeSformer(
            img_size=img_size, patch_size=patch_size, num_classes=num_classes, num_frames=num_frames, attention_type=attention_type,  
            pretrained_model=pretrained_model
        ).model
        self.out = model.head
        del model.head
        self.backbone = model
        # forward of meta timesformer backbone requires a head
        self.backbone.head = nn.Identity()

    def forward(self, videos, **kwargs):
        """
        videos BFCHW, meta needs BCFHW
        """
        x = self.backbone(rearrange(videos, "B F C H W -> B C F H W"))
        x = self.out(x)
        return x
    
    @property
    def hidden_size(self) -> int:
        return self.backbone.num_features

    @property
    def blocks(self) -> List[nn.Module]:
        return self.backbone.blocks
    
    def replace_linear(self, cls: nn.Module, num_out: int):
        self.out = cls(
            self.out.in_features, num_out
        )