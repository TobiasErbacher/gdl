# AP-GCN Revisited: Replication and Alternative Approaches for Adaptive Propagation in Graph Neural Networks
Authors: Jonatan Bella, Tobias Erbacher, Jonas Knupp

This repository contains the codebase and LaTeX sources for the replication study of the paper [Adaptive Propagation Graph Convolutional Network](https://arxiv.org/abs/2002.10306). Spinelli et al. provide some code in their [GitHub](https://github.com/spindro/AP-GCN) repository. This study was conducted as part of the **Graph Deep Learning** course at Università della Svizzera italiana in the spring semester 2025.

## Codebase Structure
This section describes the structure of the codebase.

### ./replication
The `./replication` directory contains the Python files necessary to reproduce our experiments. It is fully self-contained, as all models from the `./CoGCN` and `./AP Additional Developments` directories have been integrated into a common framework to enable consistent benchmarking.

### ./AP Additional Developments
The `./AP Additional Developments` directory contains the code for the development of RL-AP-GCN, Ponder-AP-GCN, Gumbel-AP-GCN, as well as our implementation of AP-GCN with an additional set of visualizations.

### ./CoGCN
Inside the `./CoGCN` directory, all the code regarding the adaption of the Co-GCN (Cooperative Graph Neural Networks with GCNs) is located. It contains the simplified version (`.ipynb` and `.py` files) used for integration with the remaining codebase, as well as the directory that was created to contain the code written during the adaptation of the original authors' code.

### ./report
The `./report` directory contains the LaTeX sources and figures for generating the report.