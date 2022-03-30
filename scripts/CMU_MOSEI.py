"""
preprocess CMU MOSEI csv
save new csv that contains intersection of text & video & audio
"""
import os, torch
from pathlib import Path
import librosa
import pandas as pd

csv_dir = Path("/home/ICT2000/jxu/PER/data/CMU-MOSEI/all_data")
data_root = Path("/home/ICT2000/jxu/PER/data/CMU-MOSEI/all_data")
max_audio_len = 4000

for split in ["train", "val", "test"]:
    csv_path = csv_dir / f"all_{split}.csv"
    dataset = pd.read_csv(csv_path, sep='\t', quoting=3, engine="python")
    bad_indexs = []
    for index in range(len(dataset)):
        utterance: str = dataset['utterance_id'][index]
        session, _ = utterance.rsplit("_", 1)
        """
        text always have
        """

        """
        video
        """
        frame_dir = os.path.join(
            data_root, "cropped_frames", split, session, utterance)
        if not os.path.exists(frame_dir):
            bad_indexs.append(index)
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
            data_root, "raw_audios", split, session, f"{utterance}.wav")
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