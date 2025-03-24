from enum import Enum

from replication.model_and_layer import APGCN, APGCNMappingMoved, APGCNNoWeightedSum


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



