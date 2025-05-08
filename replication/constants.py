from enum import Enum

from replication.model_classes.model_Gumbel_AP_GCN import get_Gumbel_AP_GCN_configuration
from replication.model_classes.model_Ponder_AP_GCN import get_Ponder_AP_GCN_configuration
from replication.model_classes.model_RL_AP_GCN import get_RL_AP_GCN_configuration
from replication.model_classes.model_spinelli import get_spinelli_configuration


# Do not move the Dataset enum here (or you get a circular dependency)
class Model(Enum):
    SPINELLI = ("Spinelli", get_spinelli_configuration)
    RL_AP_GCN = ("RL-AP-GCN", get_RL_AP_GCN_configuration)
    PONDER_AP_GCN = ("Ponder-AP-GCN", get_Ponder_AP_GCN_configuration)
    Gumbel_AP_GCN = ("Gumbel-AP-GCN", get_Gumbel_AP_GCN_configuration)

    def __init__(self, label: str, get_config):
        self.label = label
        self.get_config = get_config

    @classmethod
    def from_label(cls, label: str):
        return next((item for item in cls if item.label == label), None)


MATPLOTLIBPARAMS = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 14,
    "font.size": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}

ANALYSIS_FIGURE_OUTPUT_PATH = "./analysis/figures/"
