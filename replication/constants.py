from enum import Enum


class Dataset(str, Enum):
    CITESEER = 'Citeseer'
    CORAML = 'Cora-ML'
    PUBMED = 'PubMed'
    MSACADEMIC = 'MS-Academic'
    ACOMPUTER = 'A.Computer'
    APHOTO = 'A.Photo'


class Model(str, Enum):
    SPINELLI = 'spinelli'
