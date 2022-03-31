from torchmetrics import F1Score
import numpy as np
from sklearn.metrics import *
import torch


f1 = F1Score()
preds = []
trues = []
for _ in range(30):
    pred = torch.from_numpy(np.random.randint(0, 8, (20, )))
    true = torch.from_numpy(np.random.randint(0, 8, (20, )))
    f1(pred, true)
    preds.append(pred)
    trues.append(true)
print("tm: ", f1.compute())
preds = torch.cat(preds).numpy()
trues = torch.cat(trues).numpy()
print("sklearn: ", f1_score(trues, preds, average="weighted"))
print()