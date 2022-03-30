"""
preprocess CMU MOSEI csv
save new csv that contains intersection of text & video & audio
"""
import os, torch
from pathlib import Path
import librosa
import pandas as pd

pwd = Path(__file__).parent
csv_dir = pwd / "../data/datasets/CMU-MOSEI"
data_root = pwd / "../data/data/CMU-MOSEI"
audio_lens = []
max_audio_len = 4000
frame_lost = 0

for split in ["train", "val", "test"]:
    csv_path = csv_dir / f"{split}.csv"
    dataset = pd.read_csv(csv_path, sep='\t', quoting=3, engine="python")
    bad_indexs = []
    for index in range(len(dataset)):
        utterance: str = dataset['utterance_id'][index]
        session, _ = utterance.rsplit("_", 1)
        """
        text always have
        """
        text_dir = os.path.join(
            data_root, "text", split, session, f"{utterance}.txt")
        if not os.path.exists(text_dir):
            bad_indexs.append(index)
            continue
        with open(text_dir) as f:
            try:
                lines = f.readlines()
            except UnicodeDecodeError:
                # foreign words, ignore
                bad_indexs.append(index)
                continue
        if len(lines) == 0:
            bad_indexs.append(index)
            continue

        """
        video
        """
        frame_dir = os.path.join(
            data_root, "video", split, session, utterance)
        if not os.path.exists(frame_dir):
            bad_indexs.append(index)
            continue
        if len(os.listdir(frame_dir)) < 16:
            bad_indexs.append(index)
            frame_lost += 1
            continue 

        """
        audio fbank
        """
        # fbank_path = os.path.join(
        #     data_root, "FilterBank", split, session, f"{utterance}.pt")
        # if not os.path.exists(fbank_path):
        #     bad_indexs.append(index)
        #     continue
        # fbank = torch.load(fbank_path)
        # if fbank.size(0) > max_audio_len:
        #     fbank = fbank[-max_audio_len:]
        # if fbank.std().item() == 0:
        #     bad_indexs.append(index)
        #     continue

        """
        audio raw
        """
        audio_path = os.path.join(
            data_root, "audio", split, session, f"{utterance}.wav")
        if not os.path.exists(audio_path):
            bad_indexs.append(index)
            continue
        signal, sample_rate = librosa.load(audio_path, sr=None)
        assert sample_rate == 16000
        if len(signal) == 0:
            bad_indexs.append(index)
            continue
        
    bad_indexs = dataset.index.isin(bad_indexs)
    intersection = dataset[~bad_indexs]
    intersection.to_csv(
        str(csv_dir / f"post_{split}.csv"), sep="\t", quoting=3, index=False
    )
print(frame_lost)