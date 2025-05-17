import torch

from replication.data_loading.data import get_dataset
from replication.dataset import Dataset

DATASETS = [Dataset.CITESEER, Dataset.CORAML, Dataset.PUBMED, Dataset.MSACADEMIC, Dataset.ACOMPUTER, Dataset.APHOTO]


def edge_index_to_dense_adj(edge_index, num_nodes):
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1.0
    return adj


for dataset_enum in DATASETS:
    dataset = get_dataset(dataset_enum.label)

    # Check if adjacency matrix is symmetric (i.e, whether graph is undirected)
    A = edge_index_to_dense_adj(dataset.data.edge_index, len(dataset.data.x))

    is_symmetric = (A == A.T).all().item()
    print("Symmetric adj. matrix:", is_symmetric)

    print(f"Dataset name: {dataset_enum.label}")

    print(f"Number classes: {dataset.num_classes}")

    print(f"Features: {len(dataset.data.x[0])}")

    num_nodes = len(dataset.data.x)
    print(f"Nodes: {num_nodes}")

    # Division by 2 because the edges are undirected but are bidirectional in the edge list
    print(f"Edges: {len(dataset.data.edge_index[0]) // 2}")

    # Degree of each node
    degrees = torch.bincount(torch.concat((dataset.data.edge_index[0], dataset.data.edge_index[1])),
                             minlength=num_nodes)

    # Average degree
    average_degree = degrees.float().mean()

    print(f"Average degree: {average_degree:.2f}")

    # Homophily ratio
    labels = dataset.data.y
    edge_index = dataset.data.edge_index

    same_label = labels[edge_index[0]] == labels[edge_index[1]]

    homophily_ratio = same_label.float().mean().item()
    print(f"Homophily ratio: {homophily_ratio:.2f}")
    print("-----------------------------")
