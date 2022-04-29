from typing import Optional
import torch
import numpy as np
from torchmetrics import (
    MeanAbsoluteError, MetricCollection, Metric,
    Accuracy, Precision, Recall, F1Score, ClasswiseWrapper,
    MeanMetric
)
from sklearn.metrics import accuracy_score, f1_score

class MOSEIMetric(Metric):
    """
    compute 2-class metric for CMU-MOSEI
    only compute over instances where [label2 != None, label7 != 0]
    """
    def __init__(self, compute_on_step=True, dist_sync_on_step=False):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)
        self.add_state("true7", default=[], dist_reduce_fx="cat")
        self.add_state("true2", default=[], dist_reduce_fx="cat")
        self.add_state("pred", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target7: torch.Tensor, target2: torch.Tensor):
        """
        all (bsz, ), preds & target7 0-6 int, target2 -1, 0, 1 where -1 means None (exclude from calc)
        """
        assert preds.shape == target7.shape == target2.shape
        assert preds.ndim == 1
        self.true7.append(target7.detach())
        self.true2.append(target2.detach())
        self.pred.append(preds.detach())

    def compute(self):
        if self.true7[0].ndim != 0:
            y_true2, y_true7, y_pred = torch.cat(self.true2), torch.cat(self.true7), torch.cat(self.pred)
        else:
            y_true2, y_true7, y_pred = self.true2, self.true7, self.pred
        # from [0, 6] to [-3, 3]
        y_true7, y_pred = y_true7.cpu().numpy() - 3, y_pred.cpu().numpy() - 3
        y_true2 = y_true2.cpu().numpy()

        corr = np.corrcoef(y_pred, y_true7)[0][1]

        # if label2 is None, map to -1
        exclude = (y_true2 == -1) & (y_true7 == 0)
        y_true7, y_pred = y_true7[~exclude], y_pred[~exclude]
        y_true7 = np.where(y_true7 > 0, 1, 0)
        y_pred = np.where(y_pred > 0, 1, 0)

        return {
            "corr": corr,
            "acc2": accuracy_score(y_true7, y_pred),
            "f12":  f1_score(y_true7, y_pred, average="weighted")
        }

class CCC(Metric):
    """
    ConcordanceCorrelationCoefficient
    """
    def __init__(self, index: int, compute_on_step=True, dist_sync_on_step=False):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)
        self.add_state("true", default=[], dist_reduce_fx="cat")
        self.add_state("pred", default=[], dist_reduce_fx="cat")
        self.index = index

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        both (bsz, 2), only get the `index` one
        """
        assert preds.shape == target.shape
        assert preds.ndim == 2
        self.true.append(target[:, self.index].detach())
        self.pred.append(preds[:, self.index].detach())
    
    def compute(self):
        """
        Returns
        loss : A float in the range [-1,1]. A value of 1 indicates perfect agreement
        between the true and the predicted values.
        """
        if self.true[0].ndim != 0:
            y_true, y_pred = torch.cat(self.true), torch.cat(self.pred)
        else:
            y_true, y_pred = self.true, self.pred
        # both (bsz, )
        y_true, y_pred = y_true.cpu().numpy(), y_pred.cpu().numpy()
        cor=np.corrcoef(y_true,y_pred)[0][1]

        mean_true=np.mean(y_true)
        mean_pred=np.mean(y_pred)

        var_true=np.var(y_true)
        var_pred=np.var(y_pred)

        sd_true=np.std(y_true)
        sd_pred=np.std(y_pred)

        numerator=2*cor*sd_true*sd_pred
        denominator=var_true+var_pred+(mean_true-mean_pred)**2
        return numerator/denominator


class ModuleMetricMixin:
    def __init__(
        self, task: str, num_classes: int,
        dataset: str,
        ordinal_regression: Optional[str] = None,
        **kwargs
    ):
        assert task in ["clf", "reg"]
        assert dataset in ["msp_improv", "cmu_mosei", "iemocap", "cmu_mosi", "ntu_rgb", "drive_and_act"]
        super().__init__()
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch

        # loss function and metric
        dataset_specific_metrics = None
        if task == "reg":
            assert ordinal_regression is None
            assert num_classes == 2
            assert dataset == "msp_improv"
            train_metrics = MetricCollection({
                "valence_CCC": CCC(index=0, compute_on_step=False),
                "arousal_CCC": CCC(index=1, compute_on_step=False),
            }, prefix="train/")
        else:
            additional_metrics = {}
            # optional store classwise metric
            labels = None
            if dataset == "msp_improv": 
                assert ordinal_regression is None
                assert num_classes == 4
                labels = ["neutral", "angry", "sad", "happy"]
            elif dataset in ["cmu_mosei", "cmu_mosi"]: # similar setting for those two
                if num_classes == 7:
                    assert ordinal_regression in [None, "None", "coral", "corn"], f"ordinal regression is {ordinal_regression}, type={type(ordinal_regression)}"
                elif num_classes == 1:
                    # L1 loss
                    assert ordinal_regression is None
                    # temporary change to 7 so that classwise wrapper works
                    num_classes = 7
                else:
                    raise NotImplementedError
                labels = list(map(str, range(-3, 4)))
                additional_metrics = {
                    **additional_metrics,
                    "MAE": MeanAbsoluteError(),
                }
                dataset_specific_metrics = (
                    dataset.replace("cmu_", ""), MOSEIMetric(compute_on_step=False))
            elif dataset == "ntu_rgb":
                assert num_classes == 60
            elif dataset == "drive_and_act":
                assert num_classes == 34
            # iemocap
            else:
                assert ordinal_regression is None
                assert num_classes == 4
                labels = ["neu", "sad", "ang", "hap"]
            
            if labels is not None:
                additional_metrics.update({
                    "classwise_acc": ClasswiseWrapper(Accuracy(num_classes=num_classes, average=None), labels),
                    "classwise_f1": ClasswiseWrapper(F1Score(num_classes=num_classes, average=None), labels),
                })
            train_metrics = MetricCollection({
                # weighted acc, reported in MOSEI
                "WA": Accuracy(num_classes=num_classes, threshold=0.5, average="weighted"),
                # unweight acc, reported in MSP
                "UA": Accuracy(num_classes=num_classes, threshold=0.5, average="micro"),
                "Precision": Precision(num_classes=num_classes, threshold=0.5, average="weighted"),
                "Recall": Recall(num_classes=num_classes, threshold=0.5, average="weighted"),
                # weighted F1
                "WF1": F1Score(num_classes=num_classes, threshold=0.5, average="weighted"),
                **additional_metrics
            }, prefix="train/")

        self.metrics = torch.nn.ModuleDict({
            "train_metrics": train_metrics,
            "valid_metrics": train_metrics.clone(prefix="valid/"),
            "test_metrics": train_metrics.clone(prefix="test/")
        })
        if dataset_specific_metrics is not None:
            metricname, metric = dataset_specific_metrics
            self.metrics.update({
                f"train_{metricname}": metric,
                f"valid_{metricname}": metric.clone(),
                f"test_{metricname}": metric.clone(),
            })

        self.mean_losses = torch.nn.ModuleDict({
            "train_losses": MeanMetric(),
            "valid_losses": MeanMetric(),
        })