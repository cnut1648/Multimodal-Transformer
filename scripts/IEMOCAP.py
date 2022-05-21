"""
preprocess MSP IMPROV csv
save new csv that contains intersection of text & video & audio
only keep 
"""
import os, torch
from pathlib import Path
import librosa
import pandas as pd

def get_meta(utterance):
    # eg Ses02F_impro01_F000 or Ses02F_script01_1_F004
    splits = utterance.split('_')
    session, dialog, utteranceid = splits[0], "_".join(splits[1:-1]), splits[-1]
    session, gender = session[-2:]
    # eg 2, F, script01_1, F004
    return session, gender, dialog, utteranceid

ID2LABEL = {
    0: "neu", 1: "sad", 2: "ang", 3: "hap"
}
pwd = Path(__file__).parent
csv_dir = pwd / "../data/datasets/IEMOCAP"
data_root = pwd / "../data/data/IEMOCAP"
audio_lens = []

# can compute stat here
csv_path = csv_dir / f"Session{1}.csv"
dataset = pd.read_csv(csv_path)
for sessionid in [2, 3, 4, 5]:
    csv_path = csv_dir / f"Session{sessionid}.csv"
    dataset = dataset.append( pd.read_csv(csv_path))
dataset.reset_index(inplace=True)

frame_lost = 0
for sessionid in [1, 2, 3, 4, 5]:
    csv_path = csv_dir / f"Session{sessionid}.csv"
    dataset = pd.read_csv(csv_path)
    bad_indexs = []
    for index in range(len(dataset)):
        utterance: str = dataset['utterance_id'][index]
        
        session, gender, dialog, utteranceid = get_meta(utterance)
        """
        text
        """
        text_dir = os.path.join(
            data_root, "text", f"Session{session}", f"{utterance}.txt"
        )
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
        no video
        """
        # frame_dir = os.path.join(
        #     data_root, "video", 'Session'+session, utterance) 
        # if not os.path.exists(frame_dir):
        #     bad_indexs.append(index)
        #     continue
        # if len(os.listdir(frame_dir)) < 16:
        #     frame_lost += 1
        #     bad_indexs.append(index)
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
        audio_path = os.path.join(
            data_root, "audio", f"Session{session}",
            f"Ses0{session}{gender}_{dialog}", f"{utterance}.wav")
        if not os.path.exists(audio_path):
            bad_indexs.append(index)
            continue
        signal, sample_rate = librosa.load(audio_path, sr=None)
        assert sample_rate == 16000
        audio_lens.append( len(signal) )
        if len(signal) == 0:
            bad_indexs.append(index)
            continue
        
    bad_indexs = dataset.index.isin(bad_indexs)
    intersection = dataset[~bad_indexs]
    intersection.to_csv(
        str(csv_dir / f"post_session{sessionid}.csv"), index=False
    )
print(frame_lost)