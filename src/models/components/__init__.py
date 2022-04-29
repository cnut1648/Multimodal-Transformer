from torch import nn
from typing import List
from .. import TextModule

class ModalityModel(nn.Module):
    @staticmethod
    def load_ckpt():
        ckpt = "/home/ICT2000/jxu/Multimodal-Transformer/selected_ckpt/M2022-IEMOCAP-text-clf/4/ckpt/epoch07-F10.70-acc0.71.ckpt"
        c = TextModule.load_from_checkpoint(ckpt)
        print(c)
        return c
    
    @property
    def hidden_size(self) -> int:
        pass
    @property
    def blocks(self) -> List[nn.Module]:
        """
        normalization blocks
        """
        pass
    def replace_linear(self, cls: nn.Module, num_out: int):
        """
        ordinal regression, replace last linear
        """
        pass