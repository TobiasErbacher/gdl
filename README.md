# Graph Deep Learning: Paper Replication Project
Authors: Jonathan Bella, Tobias Erbacher, Jonas Knupp

This repository contains the codebase and LaTeX sources for the replication study of the paper [Adaptive Propagation Graph Convolutional Network](https://arxiv.org/abs/2002.10306). Spinelli et al. provide some code in their [GitHub](https://github.com/spindro/AP-GCN) repository. This study was conducted as part of the **Graph Deep Learning** course at Università della Svizzera italiana in the spring semester 2025.

<Add notes for running the notebooks!>

The codebase is structured in the following directories:

<Add codebase directory structure!>

## ./replication

First you must log-in in wandb: `wandb login`. The `replicate.py` will use the `AP-GCN` wandb project.

Run `replicate.py` with the following arguments. Note that `number of propagation penalties` times 100 runs are created in wandb.

* --dataset=Citeseer --model=spinelli --prop-penalties 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001
* --dataset=Cora-ML --model=spinelli --prop-penalties 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001
* --dataset=PubMed --model=spinelli --prop-penalties 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001
* --dataset=MS-Academic --model=spinelli --prop-penalties 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001
* --dataset=A.Computer --model=spinelli --prop-penalties 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001
* --dataset=A.Photo --model=spinelli --prop-penalties 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001
