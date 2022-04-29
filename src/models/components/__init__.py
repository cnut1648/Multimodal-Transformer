from torch import nn
from typing import List

class ModalityModel(nn.Module):
    @staticmethod
    def load_ckpt(wandb_path):
        """
        wandb_path: e.g. <project>:<id>
        project eg M2022-IEMOCAP-text-clf
        """
        from src.models.text_module import TextModule
        from src.models.audio_module import AudioModule
        from src.models.video_module import VideoModule
        import wandb, os, json
        
        project, id = wandb_path.split(":")
        api = wandb.Api()
        ckpt_path = None
        for run in api.runs(f"usc_ihp/{project}"):
            if id in run.name:
                config = json.loads(run.json_config)
                # eg '/home/ICT2000/jxu/Multimodal-Transformer/logs/save/M2022-IEMOCAP-text-clf/6/ckpt/epoch06-F10.74-acc0.74.ckpt'
                ckpt_path = config['best_model_path']['value']
                # selected_ckpt/M2022-IEMOCAP-text-clf/6/ckpt/epoch06-F10.74-acc0.74.ckpt
                ckpt_path = "/home/ICT2000/jxu/Multimodal-Transformer/selected_ckpt/" + ckpt_path[ckpt_path.index(project):]
                os.path.exists(ckpt_path), f"{ckpt_path} doesn't exist"
                break
        assert ckpt_path is not None, f"{wandb_path} is not in the wandb"

        module_dict = {
            "text": TextModule,
            "audio": AudioModule,
            "video": VideoModule
        }
        for m, Module in module_dict.items():
            if m in wandb_path:
                break
        
        c = Module.load_from_checkpoint(ckpt_path)
        return c.model
    
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