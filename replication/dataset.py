from enum import Enum


class Dataset(Enum):
    CITESEER = ("Citeseer", 1500, "darkgreen")
    CORAML = ("Cora-ML", 1500, "darkorange")
    PUBMED = ("PubMed", 1500, "blue")
    # According to [18] (otherwise not 20 instances per class in the dev set)
    MSACADEMIC = ("MS-Academic", 5000, "pink")
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
