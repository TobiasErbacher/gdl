# Cooperative Graph Neural Networks

The contents in this folder are adapted from the [github repository](https://github.com/benfinkelshtein/CoGNN/tree/main) of [Finkelshtein et al. (2023)](https://doi.org/10.48550/arXiv.2310.01267).

To reproduce the results we have to use Python 3.9, PyTorch version 2.0.0, Cuda 11.8, PyG version 2.3.0, and torchmetrics.

The datasets that we will be training on are already downloaded in [another folder](https://github.com/TobiasErbacher/gdl/tree/main/replication/data).

Further installation instructions can be found in the [original github repository](https://github.com/benfinkelshtein/CoGNN/tree/main).

The main file of this approach is _main.ipynb_. The other _.py_ files contain some class definitions that are used in the main Jupyter Notebook.

The directory [CodeSplit](https://github.com/TobiasErbacher/gdl/tree/main/Tobias/CoGNN/CodeSplit) contains the authors' code but reorganized into individual class files. Moreover, the folder [Reimplementation](https://github.com/TobiasErbacher/gdl/tree/main/Tobias/CoGNN/Reimplementation) contains a Jupyter Notebook with the commented code.