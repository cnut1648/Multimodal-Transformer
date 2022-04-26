import random
from typing import Optional, Tuple
import os
import numpy as np

import torch, librosa
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torch.utils.data.sampler import RandomSampler, SequentialSampler, BatchSampler, Sampler
from torchvision.transforms import transforms
from ..utils.io import sample_frames, load_img
from torch.utils.data.dataloader import default_collate

class MultiDataset(Dataset):
    def __init__(self, task: str, csv_path, data_root, split, modality: str, **kwargs):
        self.task = task
        self.dataset = pd.read_csv(os.path.join(csv_path, "processed", f"kinect_{modality}", f"{split}.csv"), engine="python")
        self.dataset.reset_index(inplace=True)
        self.data_root = data_root
        self.split = split

    def __getitem__(self, index):
        # csv label 0-33
        label = self.dataset['label'][index]
        return {"labels": label}

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def collate_fn(batch):
        return default_collate(list(
            filter(lambda x: x is not None, batch)
        ))

    def get_video(self, utterance, num_frames, transform, is_train):
        frame_dir = os.path.join(
            self.data_root, utterance)
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
    def get_batch_sampler(dataset, batch_size: int, shuffle: bool, **kwargs):
        if shuffle == True:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        return BatchSampler(sampler, batch_size, drop_last=False)

class VideoDataset(MultiDataset):
    def __init__(self, task: str, csv_path, data_root, modality,
           split, num_frames, transform=None, **kwargs):
        super().__init__(task, csv_path, data_root, split, modality)
        self.is_train = (split == "train")
        self.num_frames = num_frames
        self.transform = transform
    
    def __getitem__(self, index):
        utterance = self.dataset['path'][index]
        ret = super().__getitem__(index)
        return {
            **ret, "videos": self.get_video(utterance, self.num_frames, self.transform, self.is_train)
        }

class DataModule(LightningDataModule):
    def __init__(
        self, task: str, modality: str, fold: str,
        # dataset
        # where the csvs are and where the raw data are
        csv_dir: str = "dataset/", data_dir: str = "data/", 
        # for video only
        num_frames: int = 16, image_size: int = 112, 
        # dataloader
        batch_size: int = 64, num_workers: int = 0, pin_memory: bool = False,
        **kwargs
    ):
        # in Drive&Act, data_dir = csv_dir which direct to folder that contains processed/
        assert os.path.exists(data_dir)
        assert os.path.exists(csv_dir)
        assert fold in [1]
        assert task in ["clf"]
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
            "color": VideoDataset,
            "ir": VideoDataset,
            "depth": VideoDataset,
        }

    def setup(self, stage: Optional[str] = None):
        ds_cls = self.modality2ds[self.hparams.modality]
        for split in ["train", "val", "test"]:
            self.datasets[split] = ds_cls(
                self.hparams.task, 
                self.hparams.csv_dir, self.hparams.data_dir, 
                split=split, num_frames=self.hparams.num_frames, transform=self.transforms, modality=self.hparams.modality,
            )
    
    def load_dataloader(self, split: str):
        ds_cls = self.modality2ds[self.hparams.modality]
        batch_sampler = ds_cls.get_batch_sampler(
            self.datasets[split],
            batch_size=self.hparams.batch_size,
            shuffle=(split == "train"),
            split=split,
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