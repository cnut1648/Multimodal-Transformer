"""
preprocess CMU MOSEI csv
save new csv that contains intersection of text & video & audio
"""
import os, torch
import pickle
from pathlib import Path
import librosa
import pandas as pd

pwd = Path(__file__).parent
csv_dir = pwd / "../data/datasets/CMU-MOSI"
data_root = pwd / "../data/data/CMU-MOSI"
audio_lens = []
max_audio_len = 4000
frame_lost = 0

for split in ["train", "val", "test"]:
    csv_path = csv_dir / f"all_{split}.csv"
    dataset = pd.read_csv(csv_path, sep='\t', quoting=3, engine="python")
    with open(f'/home/ICT2000/jxu/Multimodal-Transformer/data/Multimodal-Infomax/datasets/MOSI/{split}.pkl', 'rb') as f:
        mmim = pickle.load(f)
    mmim = {
        utterance: label[0, 0]
        for (_, label, utterance) in mmim
    }
    bad_indexs = []
    wrong_labels = 0
    for index in range(len(dataset)):
        utterance: str = dataset['utterance_id'][index]
        if abs(dataset['label'][index] - mmim[utterance]) > 1e-2:
            wrong_labels += 1

        session, _ = utterance.rsplit("_", 1)
        """
        text always have
        """
        text_dir = os.path.join(
            data_root, "text", split, f"{utterance}.txt")
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
        # frame_dir = os.path.join(
        #     data_root, "video", split, session, utterance)
        # if not os.path.exists(frame_dir):
        #     bad_indexs.append(index)
        #     continue
        # if len(os.listdir(frame_dir)) < 16:
        #     bad_indexs.append(index)
        #     frame_lost += 1
        #     continue 

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
        # audio_path = os.path.join(
        #     data_root, "audio", split, f"{utterance}.wav")
        # if not os.path.exists(audio_path):
        #     bad_indexs.append(index)
        #     continue
        # signal, sample_rate = librosa.load(audio_path, sr=None)
        # assert sample_rate == 16000
        # if len(signal) == 0:
        #     bad_indexs.append(index)
        #     continue
        
    bad_indexs = dataset.index.isin(bad_indexs)
    intersection = dataset[~bad_indexs]
    intersection.to_csv(
        # str(csv_dir / f"post_{split}.csv"), sep="\t", quoting=3, index=False
        str(csv_dir / f"post_{split}_fulltext.csv"), sep="\t", quoting=3, index=False
    )
print(frame_lost)