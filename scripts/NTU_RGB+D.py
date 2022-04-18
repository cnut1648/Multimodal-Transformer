"""
NTU RGB+D 60
"""
import os, torch, soundfile
from pathlib import Path
import librosa
import pandas as pd

def get_meta(format: str):
    format = format.replace(".avi", "")
    # eg S001C002P003R002A013
    setup, cameraid, subjectid, replication, label = \
        map(int, [format[1:4], format[5:8], format[9:12], format[13:16], format[17:20]])
    return setup, cameraid, subjectid, replication, label

pwd = Path(__file__).parent
csv_dir = pwd / "../data/datasets/NTU RGB+D"
data_root = pwd / "../data/data/NTU RGB+D"
audio_lens = []

frame_lost = 0
def xsub():
    TRAINIDS = [
        1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
    ]
    train, val = [], []
    for file in os.listdir(data_root / "raw_frames" ):
        setup, cameraid, subjectid, replication, label = get_meta(file)
        if subjectid in TRAINIDS:
            train.append([file, label])
        else:
            val.append([file, label])
    train, val = pd.DataFrame(train, columns=["path", "label"]), pd.DataFrame(val, columns=["path", "label"])
    train.to_csv(csv_dir / "xsub_post_train.csv", index=False)
    val.to_csv(csv_dir / "xsub_post_val.csv", index=False)

def xview():
    TRAINVIEWS = [2, 3]
    train, val = [], []
    s = set()
    for file in os.listdir(data_root / "raw_frames"):
        setup, cameraid, subjectid, replication, label = get_meta(file)
        if cameraid in TRAINVIEWS:
            train.append([file, label])
        else:
            val.append([file, label])
        s.add(cameraid)
    train, val = pd.DataFrame(train, columns=["path", "label"]), pd.DataFrame(val, columns=["path", "label"])
    train.to_csv(csv_dir / "xview_post_train.csv", index=False)
    val.to_csv(csv_dir / "xview_post_val.csv", index=False)

xsub()
xview()