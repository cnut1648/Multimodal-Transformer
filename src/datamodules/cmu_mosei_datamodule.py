import random
from typing import List, Optional, Tuple
import os, pickle
import numpy as np
from pathlib import Path
from hydra.utils import get_original_cwd
from torch.utils.data.sampler import RandomSampler, SequentialSampler, BatchSampler, Sampler
import librosa

import torch
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from ..utils.io import sample_frames, load_img
from torch.utils.data.dataloader import default_collate

class CMU_MOSEI_Dataset(Dataset):
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
    def get_batch_sampler(dataset, batch_size: int, shuffle: bool, **kwargs):
        if shuffle == True:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        return BatchSampler(sampler, batch_size, drop_last=False)

class SmartBatchSampler(Sampler):
    split: str
    cache_dir: str = "."
    def __init__(self, data_source, batch_size: int = 32, drop_last: bool = False) -> None:
        super(SmartBatchSampler, self).__init__(data_source)
        self.batch_size = batch_size
        self.data_source = data_source
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, f"{self.split}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self.audio_lengths, self.audio_indices = pickle.load(f)
        else:
            all_audios = []
            all_valid_ids = []
            for i in range(len(data_source)):
                sample = data_source[i]
                if sample is not None:
                    all_audios.append(sample["audios"])
                    all_valid_ids.append(i)
            
            audio_lengths = list(map(lambda x: x.shape[0], all_audios))

            pack_by_length = list(zip(audio_lengths, all_valid_ids))
            sort_by_length = sorted(pack_by_length)
            self.audio_lengths, self.audio_indices = zip(*sort_by_length)
            with open(cache_file, "wb") as f:
                pickle.dump([self.audio_lengths, self.audio_indices], f)
        self.bins = [self.audio_indices[i:i + batch_size] for i in range(0, len(self.audio_indices), batch_size)]
        # start from longest
        self.bins = self.bins[::-1]
        self.drop_last = drop_last
    
    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(list(ids))
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)

class CMU_MOSEI_VideoDataset(CMU_MOSEI_Dataset):
    def __init__(self, task: str, split: str, csv_path, data_root, num_frames, target: str, transform=None, **kwargs):
        self.task = task
        self.csv_path = csv_path
        self.dataset = pd.read_csv(csv_path, sep='\t', quoting=3, engine="python")
        # data/CMU-MOSEI/cropped_frames/<split>
        self.data_root = os.path.join(data_root, "cropped_frames", split)
        self.train = (split == "train")
        self.num_frames = num_frames
        self.transform = transform
        self.target = target

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        # e.g. 9-K1CXCXui4_10
        utterance = self.dataset['utterance_id'][index]
        session, _ = utterance.rsplit("_", 1)
        frame_dir = os.path.join(
            self.data_root, session, utterance)
        assert os.path.exists(frame_dir)
        frames = type(self).get_video(
            frame_dir, self.num_frames, self.transform, self.train
        )
        label = self.dataset[self.target][index]
        if self.target == "label_2" and label == 'None':
            return
        elif self.target == "label_7":
            # shift [-3, -2, -1, 0, 1, 2, 3] to [0, 1, 2, 3, 4, 5, 6]
            label = int(label) + 3
        
        return {
            "videos": frames,
            "labels": int(label)
        } 

class CMU_MOSEI_AudioDataset(CMU_MOSEI_Dataset):
    def __init__(self, task: str, split: str, csv_path, target: str, data_root: str, 
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

    def __len__(self):
        return len(self.dataset)
    
    def load_fbank(self, fbank_path):
        if not os.path.exists(fbank_path):
            return
        # (#frames, #mel=80)
        fbank = torch.load(fbank_path)
        if fbank.size(0) > self.max_audio_len:
            fbank = fbank[-self.max_audio_len:]
        if fbank.std().item() == 0:
            # ignore bad instances
            return
        fbank -= fbank.mean()
        fbank /= fbank.std()

        # use aug
        if self.train and random.random() < 0.5:
            from openspeech.data.audio.augment import SpecAugment
            fbank = SpecAugment(freq_mask_para=27, time_mask_num=4, freq_mask_num=2)(fbank)
        return fbank
    
    def load_raw(self, raw_path):
        signal, sample_rate = librosa.load(raw_path, sr=None, mono=True)
        if len(signal) > self.max_audio_len:
            signal = signal[-self.max_audio_len:]
        return signal
    
    def __getitem__(self, index):
        utterance = self.dataset['utterance_id'][index]
        session, _ = utterance.rsplit("_", 1)
        if self.audio_format == "fbank":
            fbank_path = os.path.join(
                self.data_root, session, f"{utterance}.pt")
            audio = self.load_fbank(fbank_path)
            if audio is None: return

        elif self.audio_format == "raw":
            raw_path = os.path.join(
                self.data_root, session, f"{utterance}.wav")
            audio = self.load_raw(raw_path)

        label = self.dataset[self.target][index]
        if self.target == "label_2" and label == 'None':
            return
        elif self.target == "label_7":
            # shift [-3, -2, -1, 0, 1, 2, 3] to [0, 1, 2, 3, 4, 5, 6]
            label = int(label) + 3

        return {
            "audios": audio,
            "labels": int(label)
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
            return {
                "audios": [b["audios"] for b in batch],
                "labels": torch.tensor([b["labels"] for b in batch]),
            }
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

    @staticmethod
    def get_batch_sampler(dataset, split, batch_size: int, shuffle: bool, cache_dir: str, use_smart_sampler: bool):
        if use_smart_sampler:
            sampler = SmartBatchSampler
            sampler.split = split
            sampler.cache_dir = cache_dir
            return sampler(dataset, batch_size, drop_last=False)
        return CMU_MOSEI_Dataset.get_batch_sampler(dataset, batch_size, shuffle)

class CMU_MOSEI_TextDataset(CMU_MOSEI_Dataset):
    def __init__(self, task: str, csv_path, target: str, **kwargs):
        self.task = task
        self.csv_path = csv_path
        # contains text directly
        self.dataset = pd.read_csv(csv_path, sep='\t', quoting=3, engine="python")
        self.target = target

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        label = self.dataset[self.target][index]
        if self.target == "label_2" and label == 'None':
            return
        elif self.target == "label_7":
            # shift [-3, -2, -1, 0, 1, 2, 3] to [0, 1, 2, 3, 4, 5, 6]
            label = int(label) + 3

        return {
            "text": self.dataset["text"][index],
            "labels": int(label)
        }


class CMU_MOSEI_VTDataset(CMU_MOSEI_Dataset):
    def __init__(self, task: str, csv_path, data_root, train, num_frames: int, transform, target: str, **kwargs):
        self.task = task
        self.csv_path = csv_path
        self.dataset = pd.read_csv(csv_path, sep='\t', quoting=3, engine="python")
        # data/CMU-MOSEI/cropped_frames/<split>
        self.data_root = data_root
        self.train = train
        self.num_frames = num_frames
        self.transform = transform
        self.target = target

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        utterance = self.dataset['utterance_id'][index]
        session, _ = utterance.rsplit("_", 1)
        frame_dir = os.path.join(
            self.data_root, session, utterance)
        assert os.path.exists(frame_dir)
        frames = CMU_MOSEI_VideoDataset.get_video(
            frame_dir, self.num_frames, self.transform, self.train
        )
        label = self.dataset[self.target][index]
        if self.target == "label_2" and label == 'None':
            return
        elif self.target == "label_7":
            # shift [-3, -2, -1, 0, 1, 2, 3] to [0, 1, 2, 3, 4, 5, 6]
            label = int(label) + 3

        return {
            "text": self.dataset["text"][index],
            "videos": frames,
            "labels": int(label)
        }



class CMU_MOSEI_DataModule(LightningDataModule):
    def __init__(
        self, task: str, modality: str,
        # dataset
        # where the csvs are and where the raw data are
        csv_dir: str = "dataset/", data_dir: str = "data/", 
        num_frames: int = 16, image_size: int = 112, num_classes: int = -1,
        # dataloader
        batch_size: int = 64, num_workers: int = 0, pin_memory: bool = False, 
        # for audio only
        spec_aug: bool = False, max_audio_len: int = 8000, audio_format: str = "raw", use_smart_sampler: bool = False, cache_dir: str = ".",
        **kwargs
    ):
        assert os.path.exists(data_dir)
        assert os.path.exists(csv_dir)
        assert task in ["clf", "reg"]
        assert num_classes in [1, 7]
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
            "video": CMU_MOSEI_VideoDataset,
            "text": CMU_MOSEI_TextDataset,
            "audio": CMU_MOSEI_AudioDataset,
            "vt": CMU_MOSEI_VTDataset
        }

    def setup(self, stage: Optional[str] = None):
        ds_cls = self.modality2ds[self.hparams.modality]
        for split in ["train", "val", "test"]:
            target_label = f"label_{self.hparams.num_classes}"
            data_root = self.hparams.data_dir

            self.datasets[split] = ds_cls(
                self.hparams.task, split=split,
                csv_path=os.path.join(self.hparams.csv_dir, f"post_{split}.csv"),
                data_root=data_root, target=target_label,
                num_frames=self.hparams.num_frames, transform=self.transforms,
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
