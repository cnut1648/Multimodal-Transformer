"""
preprocess MSP IMPROV csv
save new csv that contains intersection of text & video & audio
only keep 
"""
import os, torch, soundfile
from pathlib import Path
import librosa
import pandas as pd

def get_meta(utterance):
    # eg MSP-IMPROV-S03H-M01-S-FM03
    _, _, session, gender_session, dialog, _ = utterance.split('-')
    # M01 -> M, 1
    gender, sessionnum = gender_session[0], gender_session[-1]
    # eg S03H, S, M, 1
    return session, dialog, gender, sessionnum

ID2LABEL = {
    0: "neu", 1: "sad", 2: "ang", 3: "hap"
}
pwd = Path(__file__).parent
csv_dir = pwd / "../data/datasets/MSP-IMPROV"
data_root = pwd / "../data/data/MSP-IMPROV"
audio_lens = []

# can compute stat here
csv_path = csv_dir / f"session{1}.csv"
dataset = pd.read_csv(csv_path)
for sessionid in [2, 3, 4, 5, 6]:
    csv_path = csv_dir / f"session{sessionid}.csv"
    dataset = dataset.append( pd.read_csv(csv_path))
dataset.reset_index(inplace=True)

frame_lost = 0
for sessionid in [1, 2, 3, 4, 5, 6]:
    csv_path = csv_dir / f"session{sessionid}.csv"
    dataset = pd.read_csv(csv_path)
    bad_indexs = []
    for index in range(len(dataset)):
        utterance: str = dataset['utterance_id'][index]
        
        session, dialog, _, sessionnum = get_meta(utterance)
        assert int(sessionnum) == sessionid
        """
        text
        """
        text_dir = os.path.join(
            data_root, "text", f'session{sessionid}', session, f"{utterance}.txt"
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
        frame_dir = os.path.join(
            data_root, "video", f'session{sessionid}', session, utterance) 
        if not os.path.exists(frame_dir):
            bad_indexs.append(index)
            continue
        if len(os.listdir(frame_dir)) < 16:
            frame_lost += 1
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
            data_root, "audio", f'session{sessionid}', session,
            dialog, f"{utterance}.wav")
        if not os.path.exists(audio_path):
            bad_indexs.append(index)
            continue
        # msp-improve sample rate 44100, need to convert to 16000
        signal, sample_rate = librosa.load(audio_path, sr=16000)
        assert sample_rate == 16000
        audio_lens.append( len(signal) )
        if len(signal) == 0:
            bad_indexs.append(index)
            continue
        else:
            savepath = os.path.join(data_root, "resampled_audio", f'session{sessionid}', session, dialog)
            os.makedirs(savepath, exist_ok=True)
            soundfile.write(os.path.join(
                savepath, f"{utterance}.wav"
            ), signal, 16000)
        
    bad_indexs = dataset.index.isin(bad_indexs)
    intersection = dataset[~bad_indexs]
    intersection.to_csv(
        str(csv_dir / f"post_session{sessionid}.csv"), index=False
    )
print(frame_lost)