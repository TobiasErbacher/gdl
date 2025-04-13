from enum import Enum

from replication.AP_GCN_model_and_layer import APGCN, APGCNMappingMoved, APGCNNoWeightedSum


class Dataset(str, Enum):
    CITESEER = 'Citeseer'
    CORAML = 'Cora-ML'
    PUBMED = 'PubMed'
    MSACADEMIC = 'MS-Academic'
    ACOMPUTER = 'A.Computer'
    APHOTO = 'A.Photo'


class Model(str, Enum):
    SPINELLI = 'spinelli'
    MAPPINGMOVED = 'mapping'
    NOWEIGHTEDSUM = 'no-weighted-sum'

    @staticmethod
    def get_model_class(value: str):
        if value == Model.SPINELLI.value:
            return APGCN
        elif value == Model.MAPPINGMOVED.value:
            return APGCNMappingMoved
        elif value == Model.NOWEIGHTEDSUM.value:
            return APGCNNoWeightedSum
        else:
            raise RuntimeError("Invalid model name provided")


def datasetToColorString(dataset: Dataset):
    if dataset == Dataset.CITESEER:
        return "darkgreen"
    elif dataset == Dataset.CORAML:
        return "darkorange"
    elif dataset == Dataset.PUBMED:
        return "blue"
    elif dataset == Dataset.MSACADEMIC:
        return "pink"
    elif dataset == Dataset.ACOMPUTER:
        return "lightgreen"
    elif dataset == Dataset.APHOTO:
        return "yellow"


MATPLOTLIBPARAMS = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}
