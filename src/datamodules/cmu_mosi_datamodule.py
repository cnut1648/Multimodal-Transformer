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
    def __init__(self, task: str, csv_path, data_root, split, **kwargs):
        self.task = task
        self.dataset = pd.read_csv(os.path.join(csv_path, f"post_{split}_fulltext.csv"), sep='\t', quoting=3, engine="python")
        self.dataset.reset_index(inplace=True)
        self.data_root = data_root
        self.split = split

    def __getitem__(self, index):
        # label = int(self.dataset['label_7'][index])
        # shift [-3, -2, -1, 0, 1, 2, 3] to [0, 1, 2, 3, 4, 5, 6]
        # label = int(label) + 3
        label = self.dataset['label'][index]
        label2 = self.dataset['label_2'][index]
        # label2 can be none, change it to -1
        if label2 is None or label2 == "None":
            label2 = -1
        return {"labels": label, "labels2": int(label2)}

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def collate_fn(batch):
        return default_collate(list(
            filter(lambda x: x is not None, batch)
        ))

    def get_video(self, utterance, num_frames, transform, is_train):
        session, _ = utterance.rsplit("_", 1)
        frame_dir = os.path.join(
            self.data_root, "video", self.split, session, utterance)
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
        text_dir = os.path.join(
            self.data_root, "text", self.split, f"{utterance}.txt")
        with open(text_dir) as f:
            lines = f.readlines()
        return lines[0]
    
    def get_raw_audio(self, utterance, max_audio_len):
        audio_path = os.path.join(
            self.data_root, "audio", self.split, f"{utterance}.wav")
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


class VideoDataset(MultiDataset):
    def __init__(self, task: str, csv_path, data_root,
           split, num_frames, transform=None, **kwargs):
        super().__init__(task, csv_path, data_root, split)
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
    def __init__(self, task: str, csv_path, data_root, 
            split, spec_aug=False, max_audio_len=8000, audio_format="raw", **kwargs):
        super().__init__(task, csv_path, data_root, split)
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
                k: torch.tensor([b[k] for b in batch if k != "audios"])
                for k in batch[0]
            }
            ret["audios"] = [b['audios'] for b in batch]
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
        self, task: str, modality: str,
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
        for split in ["train", "val", "test"]:
            self.datasets[split] = ds_cls(
                self.hparams.task,
                self.hparams.csv_dir, self.hparams.data_dir, 
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
        return self.load_dataloader("test")