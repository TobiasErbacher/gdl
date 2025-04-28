# Graph Deep Learning: Paper Replication Project
Authors: Jonathan Bella, Tobias Erbacher, Jonas Knupp

This repository contains the codebase and LaTeX sources for the replication study of the paper [Adaptive Propagation Graph Convolutional Network](https://arxiv.org/abs/2002.10306). Spinelli et al. provide some code in their [GitHub](https://github.com/spindro/AP-GCN) repository. This study was conducted as part of the **Graph Deep Learning** course at Università della Svizzera italiana in the spring semester 2025.

<Add notes for running the notebooks!>

The codebase is structured in the following directories:

<Add codebase directory structure!>


* RL-AP-GCN
* Ponder-AP-GCN
* Gumbel-AP-GCN

## ./replication

First you must log-in in wandb: `wandb login`. The `replicate.py` will generate a separate wandb project for each model.

Run `benchmark.py` with the following arguments. Note that `number of propagation penalties` times 100 runs are created in wandb (if running the Spinelli model). The other models generate always 100 runs.

Below are the commands to run the script for the Spinelli model with seven propagation penalties:
* --dataset=Citeseer --model=Spinelli --prop-penalties 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001
* --dataset=Cora-ML --model=Spinelli --prop-penalties 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001
* --dataset=PubMed --model=Spinelli --prop-penalties 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001
* --dataset=MS-Academic --model=Spinelli --prop-penalties 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001
* --dataset=A.Computer --model=Spinelli --prop-penalties 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001
* --dataset=A.Photo --model=Spinelli --prop-penalties 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001

Below are the commands to run the script for the RL-AP-GCN model:
* --dataset=Citeseer --model=RL-AP-GCN 
* --dataset=Cora-ML --model=RL-AP-GCN  
* --dataset=PubMed --model=RL-AP-GCN  
* --dataset=MS-Academic --model=RL-AP-GCN  
* --dataset=A.Computer --model=RL-AP-GCN  
* --dataset=A.Photo --model=RL-AP-GCN  

* Below are the commands to run the script for the Ponder-AP-GCN model:
* --dataset=Citeseer --model=Ponder-AP-GCN 
* --dataset=Cora-ML --model=Ponder-AP-GCN 
* --dataset=PubMed --model=Ponder-AP-GCN 
* --dataset=MS-Academic --model=Ponder-AP-GCN 
* --dataset=A.Computer --model=Ponder-AP-GCN 
* --dataset=A.Photo --model=Ponder-AP-GCN 

* Below are the commands to run the script for the Gumbel-AP-GCN model:
* --dataset=Citeseer --model=Gumbel-AP-GCN 
* --dataset=Cora-ML --model=Gumbel-AP-GCN 
* --dataset=PubMed --model=Gumbel-AP-GCN 
* --dataset=MS-Academic --model=Gumbel-AP-GCN 
* --dataset=A.Computer --model=Gumbel-AP-GCN 
* --dataset=A.Photo --model=Gumbel-AP-GCN 

To execute the analysis scripts the working directory must be set to `./replicate`.
