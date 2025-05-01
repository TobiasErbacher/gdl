import torch

from replication.data import get_dataset
from replication.dataset import Dataset

DATASETS = [Dataset.CITESEER, Dataset.CORAML, Dataset.PUBMED, Dataset.MSACADEMIC, Dataset.ACOMPUTER, Dataset.APHOTO]


for dataset_enum in DATASETS:
    dataset = get_dataset(dataset_enum.label)

    print(f"Dataset name: {dataset_enum.label}")

    print(f"Number classes: {dataset.num_classes}")

    print(f"Features: {len(dataset.data.x[0])}")

    num_nodes = len(dataset.data.x)
    print(f"Nodes: {num_nodes}")

    # Division by 2 because the edges are undirected but are bidirectional in the edge list
    print(f"Edges: {len(dataset.data.edge_index[0])//2}")

    # Degree of each node
    degrees = torch.bincount(torch.concat((dataset.data.edge_index[0], dataset.data.edge_index[1])), minlength=num_nodes)

    # Average degree
    average_degree = degrees.float().mean()

    print(f"Average degree: {average_degree:.2f}")
    print("-----------------------------")
