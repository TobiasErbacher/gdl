import torch
from typing import NamedTuple

class LossesAndMetrics(NamedTuple):
    train_loss: float
    val_loss: float
    test_loss: float
    train_metric: float
    val_metric: float
    test_metric: float

    def get_fold_metrics(self):
        return torch.tensor([self.train_metric, self.val_metric, self.test_metric])