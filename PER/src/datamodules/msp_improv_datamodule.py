import random
from typing import Optional, Tuple
import os
import numpy as np

import torch
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torch.utils.data.sampler import RandomSampler, SequentialSampler, BatchSampler, Sampler
from torchvision.transforms import transforms
from ..utils.io import sample_frames, load_img
from torch.utils.data.dataloader import default_collate


def get_meta(utterance):
    dialog = utterance.split('-')[2]
    speaker = utterance.split('-')[3]
    session = speaker[-1]
    return dialog, speaker, session

# kfold: [train, val, test]
FOLD_META = {
    1: [[1, 2, 3], 4, 5],
    2: [[2, 3, 4], 5, 1],
    3: [[3, 4, 5], 1, 2],
    4: [[4, 5, 1], 2, 3],
    5: [[5, 1, 2], 3, 4]
}


class MSP_IMPROV_Dataset(Dataset):
    @staticmethod
    def collate_fn(batch):
        return default_collate(list(
            filter(lambda x: x is not None, batch)
        ))
    @staticmethod
    def get_video(frame_dir, num_frames, transform, is_train):
        frame_list = sample_frames(frame_dir, num_frames)

        frames = []
        for frame_path in frame_list:
            frame = load_img(os.path.join(frame_dir, frame_path))
            frame = transform(frame)
            frames.append(frame)
        frames = torch.stack(frames)

        if is_train:
            vflip = random.random() < 0.5
            if vflip:
                torch.flip(frames, [2])
        return frames

    @staticmethod
    def get_text(text_dir):
        with open(text_dir) as f:
            try:
                lines = f.readlines()
            except UnicodeDecodeError:
                # foreign words, ignore
                return

        if len(lines) == 0:
            return

        return lines[0]

    @staticmethod
    def get_batch_sampler(dataset, batch_size: int, shuffle: bool, **kwargs):
        if shuffle == True:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        return BatchSampler(sampler, batch_size, drop_last=False)

class MSP_IMPROV_VideoDataset(MSP_IMPROV_Dataset):
    def __init__(self, task: str, csv_path, data_root, train, num_frames, transform=None):
        self.task = task
        self.csv_path = csv_path
        self.dataset = pd.read_csv(csv_path)
        # data/MSP-IMPROV
        self.data_root = data_root
        self.train = train
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        utterance = self.dataset['utterance_id'][index]
        dialog, speaker, session = get_meta(utterance)
        frame_dir = os.path.join(
            self.data_root, 'session'+session, dialog, utterance)
        frames = type(self).get_video(
            frame_dir, self.num_frames, self.transform, self.train
        )
        
        ret = {
            "videos": frames,
            "speaker": speaker,
        } 

        if self.task == 'reg':
            valence = np.float32(self.dataset['valence'][index])
            arousal = np.float32(self.dataset['arousal'][index])
            return {**ret, "valence": valence, "arousal": arousal}
        else:
            label = int(self.dataset['label'][index])
            return {**ret, "labels": label}

class MSP_IMPROV_TextDataset(MSP_IMPROV_Dataset):
    def __init__(self, task: str, csv_path, data_root, train, **kwargs):
        self.task = task
        self.csv_path = csv_path
        self.dataset = pd.read_csv(csv_path)
        # data/MSP-IMPROV_transcript
        self.data_root = data_root.replace("MSP-IMPROV", "MSP-IMPROV_transcript")
        self.train = train

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        utterance = self.dataset['utterance_id'][index]
        dialog, speaker, session = get_meta(utterance)
        text_dir = os.path.join(
            self.data_root, 'session'+session, dialog, f"{utterance}.txt")
        line = type(self).get_text(text_dir)
        if line is None:
            return

        ret = {
            "text": line,
            "speaker": speaker,
        }

        if self.task == 'reg':
            valence = np.float32(self.dataset['valence'][index])
            arousal = np.float32(self.dataset['arousal'][index])
            return {**ret, "valence": valence, "arousal": arousal}
        else:
            label = int(self.dataset['label'][index])
            return {**ret, "labels": label}

class MSP_IMPROV_AudioDataset(MSP_IMPROV_Dataset):
    def __init__(self, task: str, split: str, fold: int, target: str, data_root: str, 
            spec_aug=False, max_audio_len=8000, audio_format="raw", **kwargs):
        self.task = task
        self.csv_path = csv_path
        # e.g. data/CMU-MOSEI/all_data
        if audio_format == "raw":
            self.data_root = os.path.join(data_root, "raw_audios", split)
        elif audio_format == "fbank":
            self.data_root = os.path.join(data_root, "FilterBank", split)
        else:
            raise NotImplementedError
        # contains text directly
        self.dataset = pd.read_csv(csv_path, sep='\t', quoting=3, engine="python")
        self.spec_aug = spec_aug
        self.max_audio_len = max_audio_len
        self.train = (split == "train")
        self.target = target
        self.audio_format = audio_format


class MSP_IMPROV_VTDataset(MSP_IMPROV_Dataset):
    def __init__(self, task: str, csv_path, data_root, train, num_frames: int, transform, **kwargs):
        self.task = task
        self.csv_path = csv_path
        self.dataset = pd.read_csv(csv_path)
        self.video_root = data_root
        # data/MSP-IMPROV_transcript
        self.text_root = data_root.replace("MSP-IMPROV", "MSP-IMPROV_transcript")
        self.train = train
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        utterance = self.dataset['utterance_id'][index]
        dialog, speaker, session = get_meta(utterance)
        text_dir = os.path.join(
            self.text_root, 'session'+session, dialog, f"{utterance}.txt")
        line = MSP_IMPROV_TextDataset.get_text(text_dir)
        if line is None:
            return
        frame_dir = os.path.join(
            self.video_root, 'session'+session, dialog, utterance)
        frames = MSP_IMPROV_VideoDataset.get_video(
            frame_dir, self.num_frames, self.transform, self.train
        )

        ret = {
            "text": line,
            "videos": frames,
            "speaker": speaker,
        }

        if self.task == 'reg':
            valence = np.float32(self.dataset['valence'][index])
            arousal = np.float32(self.dataset['arousal'][index])
            return {**ret, "valence": valence, "arousal": arousal}
        else:
            label = int(self.dataset['label'][index])
            return {**ret, "labels": label}



class MSP_IMPROV_DataModule(LightningDataModule):
    def __init__(
        self, task: str, modality: str, fold: int,
        # dataset
        # where the csvs are and where the raw data are
        csv_dir: str = "dataset/", data_dir: str = "data/", 
        num_frames: int = 16, image_size: int = 112,
        # dataloader
        batch_size: int = 64, num_workers: int = 0, pin_memory: bool = False,
        # for audio only
        spec_aug: bool = False, max_audio_len: int = 8000, audio_format: str = "raw", use_smart_sampler: bool = False, cache_dir: str = ".",
        **kwargs
    ):
        assert os.path.exists(data_dir)
        assert task in ["clf", "reg"]
        assert fold in list(range(1, 13))
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose([
            transforms.Resize((self.hparams.image_size, self.hparams.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.datasets: dict = {}
        self.modality2ds = {
            "video": MSP_IMPROV_VideoDataset,
            "text": MSP_IMPROV_TextDataset,
            "audio": MSP_IMPROV_AudioDataset,
            "vt": MSP_IMPROV_VTDataset
        }

    def setup(self, stage: Optional[str] = None):
        ds_cls = self.modality2ds[self.hparams.modality]
        for split in ["train", "val", "test"]:
            self.datasets[split] = ds_cls(
                self.hparams.task, 
                os.path.join(self.hparams.csv_dir, "12-fold", str(self.hparams.fold), f"{split}.csv"), self.hparams.data_dir,
                train=(split == "train"), num_frames=self.hparams.num_frames, transform=self.transforms,
                spec_aug=self.hparams.spec_aug, max_audio_len=self.hparams.max_audio_len, audio_format=self.hparams.audio_format
            )
    
    def load_dataloader(self, split: str):
        ds_cls = self.modality2ds[self.hparams.modality]
        batch_sampler = ds_cls.get_batch_sampler(
            self.datasets[split],
            batch_size=self.hparams.batch_size,
            shuffle=(split == "train"),
            split=split,
            cache_dir=self.hparams.cache_dir,
            use_smart_sampler=self.hparams.use_smart_sampler
        )
        return DataLoader(
            self.datasets[split],
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=ds_cls.collate_fn,
            batch_sampler=batch_sampler
        )

    def train_dataloader(self):
        return self.load_dataloader("train")

    def val_dataloader(self):
        return self.load_dataloader("val")

    def test_dataloader(self):
        return self.load_dataloader("test")
