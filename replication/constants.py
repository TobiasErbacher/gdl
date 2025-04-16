# Only put data that is valid for all model classes here

from enum import Enum

from replication.model_classes.model_spinelli import APGCN, APGCNMappingMoved, APGCNNoWeightedSum


class Dataset(Enum):
    CITESEER = ("Citeseer", 1500, "darkgreen")
    CORAML = ("Cora-ML", 1500, "darkorange")
    PUBMED = ("PubMed", 1500, "blue")
    MSACADEMIC = ("MS-Academic", 5000, "pink")  # According to [18] (otherwise not 20 instances per class in the dev set)
    ACOMPUTER = ("A.Computer", 1500, "lightgreen")
    APHOTO = ("A.Photo", 1500, "gold")

    def __init__(self, label: str, num_development: int, plot_color: str):
        self.label = label
        self.num_development = num_development
        self.plot_color = plot_color

    @classmethod
    def from_label(cls, label: str):
        return next((item for item in cls if item.label == label), None)

    def __str__(self):
        return self.label


class Model(Enum):
    SPINELLI = ("spinelli", APGCN)
    MAPPINGMOVED = ("mapping", APGCNMappingMoved)
    NOWEIGHTEDSUM = ("no-weighted-sum", APGCNNoWeightedSum)

    def __init__(self, label: str, model_class):
        self.label = label
        self.model_class = model_class

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
