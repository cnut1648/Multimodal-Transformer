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


def get_meta(utterance):
    # eg Ses02F_impro01_F000 or Ses02F_script01_1_F004
    splits = utterance.split('_')
    session, dialog, utteranceid = splits[0], "_".join(splits[1:-1]), splits[-1]
    session, gender = session[-2:]
    # eg 2, F, script01_1, F004
    return session, gender, dialog, utteranceid
    
# kfold: [train, val]
FOLD_META = {
    1: [[1, 2, 3, 4], 5],
    2: [[2, 3, 4, 5], 1],
    3: [[3, 4, 5, 1], 2],
    4: [[4, 5, 1, 2], 3],
    5: [[5, 1, 2, 3], 4]
}

class MultiDataset(Dataset):
    def __init__(self, task: str, csv_path, data_root, folds, **kwargs):
        self.task = task
        if type(folds) is not list:
            folds = [folds]
        self.dataset = pd.read_csv(os.path.join(csv_path, f"post_session{folds[0]}.csv"))
        for fold in folds[1:]:
            self.dataset = self.dataset.append(
                pd.read_csv(os.path.join(csv_path, f"post_session{fold}.csv"))
            )
        self.dataset.reset_index(inplace=True)
        self.data_root = data_root

    def __getitem__(self, index):
        utterance = self.dataset['utterance_id'][index]
        session, gender, dialog, utteranceid = get_meta(utterance)
        ret = {
            # e.g. 2F
            "speaker": f"{session}{gender}",
        }
        if self.task == 'reg':
            valence = np.float32(self.dataset['valence'][index])
            arousal = np.float32(self.dataset['arousal'][index])
            return {**ret, "valence": valence, "arousal": arousal}
        else:
            label = int(self.dataset['label'][index])
            return {**ret, "labels": label}

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def collate_fn(batch):
        return default_collate(list(
            filter(lambda x: x is not None, batch)
        ))

    def get_video(self, utterance, num_frames, transform, is_train):
        session, gender, dialog, utteranceid = get_meta(utterance)
        frame_dir = os.path.join(
            self.data_root, "video", 'Session'+session, utterance) 
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

    def get_text(self, utterance):
        session, gender, dialog, utteranceid = get_meta(utterance)
        text_dir = os.path.join(
            self.data_root, "text", f"Session{session}", f"{utterance}.txt"
        )
        with open(text_dir) as f:
            lines = f.readlines()
        return lines[0]
    
    def get_raw_audio(self, utterance, max_audio_len):
        session, gender, dialog, utteranceid = get_meta(utterance)
        audio_path = os.path.join(
            self.data_root, "audio", f"Session{session}",
            f"Ses0{session}{gender}_{dialog}", f"{utterance}.wav")
        signal, sample_rate = librosa.load(audio_path, sr=None, mono=True)
        if len(signal) > max_audio_len:
            signal = signal[-max_audio_len:]
        return signal

    @staticmethod
    def get_batch_sampler(dataset, batch_size: int, shuffle: bool, **kwargs):
        if shuffle == True:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        return BatchSampler(sampler, batch_size, drop_last=False)

class VideoDataset(MultiDataset):
    def __init__(self, task: str, csv_path, data_root, folds,
           split, num_frames, transform=None, **kwargs):
        super().__init__(task, csv_path, data_root, folds)
        self.is_train = (split == "train")
        self.num_frames = num_frames
        self.transform = transform
    
    def __getitem__(self, index):
        utterance = self.dataset['utterance_id'][index]
        ret = super().__getitem__(index)
        return {
            **ret, "videos": self.get_video(utterance, self.num_frames, self.transform, self.is_train)
        }

class TextDataset(MultiDataset):
    def __getitem__(self, index):
        utterance = self.dataset['utterance_id'][index]
        ret = super().__getitem__(index)
        return {
            **ret, "text": self.get_text(utterance)
        }

class AudioDataset(MultiDataset):
    def __init__(self, task: str, csv_path, data_root, folds,
            split, spec_aug=False, max_audio_len=8000, audio_format="raw", **kwargs):
        super().__init__(task, csv_path, data_root, folds)
        self.spec_aug = spec_aug
        self.max_audio_len = max_audio_len
        self.is_train = (split == "train")
        self.audio_format = audio_format
    def __getitem__(self, index):
        utterance = self.dataset['utterance_id'][index]
        ret = super().__getitem__(index)
        return {
            **ret, "audios": self.get_raw_audio(utterance, self.max_audio_len)
        }
    
    @staticmethod
    def collate_fn(batch, pad_idx=0):
        # for DDP: somehow make batch a dict not list of dict
        if type(batch) is dict:
            batch = [batch]
        batch = list(
            filter(lambda x: x is not None, batch)
        )
        if batch[0]["audios"].ndim == 1:
            # raw
            ret = {
                k: [b[k] for b in batch]
                for k in batch[0]
            }
            ret["labels"] = torch.tensor(ret["labels"])
            return ret
        batch = sorted(batch, key=lambda x: x["audios"].size(0), reverse=True)
        max_seq_sample = max(batch, key=lambda x: x["audios"].size(0))['audios']
        max_seq_size, feat_size = max_seq_sample.shape
        batch_size = len(batch)
        audios = torch.zeros(batch_size, max_seq_size, feat_size)
        for i in range(batch_size):
            audio = batch[i]["audios"]
            seq_length = audio.size(0)
            audios[i].narrow(0, 0, seq_length).copy_(audio)
        return {
            "labels": torch.tensor([x["labels"] for x in batch]).long(),
            "audios": audios,
            "audio_lengths": torch.tensor([len(x["audios"]) for x in batch]).long()
        }

class ATDataset(AudioDataset):
    def __getitem__(self, index):
        utterance = self.dataset['utterance_id'][index]
        ret = super().__getitem__(index)
        return {
            **ret,
            "text": self.get_text(utterance),
        }

class DataModule(LightningDataModule):
    def __init__(
        self, task: str, modality: str, fold: int,
        # dataset
        # where the csvs are and where the raw data are
        csv_dir: str = "dataset/", data_dir: str = "data/", 
        # for video only
        num_frames: int = 16, image_size: int = 112,
        # dataloader
        batch_size: int = 64, num_workers: int = 0, pin_memory: bool = False,
        # for audio only
        spec_aug: bool = False, max_audio_len: int = 8000, audio_format: str = "raw", use_smart_sampler: bool = False, cache_dir: str = ".",
        **kwargs
    ):
        assert os.path.exists(data_dir)
        assert os.path.exists(csv_dir)
        assert task in ["clf", "reg"]
        assert fold in [1, 2, 3, 4, 5]
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
            "video": VideoDataset,
            "text": TextDataset,
            "audio": AudioDataset,
            "at": ATDataset
        }

    def setup(self, stage: Optional[str] = None):
        ds_cls = self.modality2ds[self.hparams.modality]
        train, val = FOLD_META[self.hparams.fold]
        for folds, split in zip([train, val], ["train", "val"]):
            self.datasets[split] = ds_cls(
                self.hparams.task,
                self.hparams.csv_dir, self.hparams.data_dir, folds,
                split=split, num_frames=self.hparams.num_frames, transform=self.transforms,
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
        return self.load_dataloader("val")