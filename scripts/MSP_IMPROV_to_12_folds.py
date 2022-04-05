"""
preprocess MSP IMPROV csv
run after MSP_IMPROV.py
"""
import os, torch, soundfile
from pathlib import Path
import librosa
import pandas as pd

ID2LABEL = {
    0: "neu", 1: "sad", 2: "ang", 3: "hap"
}
pwd = Path(__file__).parent
csv_dir = pwd / "../data/datasets/MSP-IMPROV"
out_dir = pwd / "../data/datasets/MSP-IMPROV_12fold"
os.makedirs(out_dir, exist_ok=True)

# can compute stat here
csv_path = csv_dir / f"post_session{1}.csv"
dataset = pd.read_csv(csv_path)
for sessionid in [2, 3, 4, 5, 6]:
    csv_path = csv_dir / f"post_session{sessionid}.csv"
    dataset = dataset.append( pd.read_csv(csv_path))
dataset.reset_index(inplace=True)

for fold in range(1, 7):
    for gender in ["M", "F"]:
        partial = dataset[dataset["speaker"] == f"{gender}0{fold}"]
        assert len(partial) > 0
        partial.to_csv(
            str(out_dir / f"post_session{fold}{gender}.csv"), index=False
        )
print()
