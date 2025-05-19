# Replication
The `./replication` directory contains the Python files to replicate our experiments. It contains the following directories:
- `analysis`: Once the experiments were performend and are logged on wandb, the analysis scripts can be used to generate various plots and statistics.
- `data`: Contains the datasets as provided by Spinelli et al.
- `data_loading`: Contains code to load the datasets as provided by Spinelli et al.
- `model_classes`: Contains the PyTorch modules for our implementation of AP-GCN, RL-AP-GCN, Ponder-AP-GCN, Gumbel-AP-GCN, and Co-AP-GCN. In addition, each model architecture has classes to integrate it with the `benchmark.py` script.
- `saved_models`: This directory should be kept empty on GitHub because it is only used during experiments to store and retrieve the best model for early stopping.

Most importantly, the folder also includes the `benchmark.py` script. This script is used to perform the experiments described in the report. The results of the experiments are stored on wandb and then the scripts in the `analysis` folder can be used to analyze the results of the experiments.

Before running the `benchmark.py` script, you must log-in to wandb: `wandb login`. Make sure to set the working directory to `./replicate` before running the script. The script has two command line arguments. The argument `--dataset=$DATASET_NAME` specifies for which dataset the experiments should be performed. The argument `--model=$MODEL_NAME` specifies for which model the experiments are performed. Thus, to run all experiments like we did in our report, the script needs to be executed 6*5=30 times.

The `benchmark.py` script will generate a separate wandb project for each model in the entity `AP-GCN`. The runs for specific datasets within a wandb project can be filtered using tags.

Subsequently, we list the commands to execute all the experiments to replicate the results of our report. Each execution of the command generates 100 runs in wandb (20 seeds * 5 random weight initializations).

Below are the commands to run the `benchmark.py` script for our implementation of the AP-GCN as described in the paper by Spinelli et al.:
* --dataset=Citeseer --model=Spinelli 
* --dataset=Cora-ML --model=Spinelli
* --dataset=PubMed --model=Spinelli 
* --dataset=MS-Academic --model=Spinelli 
* --dataset=A.Computer --model=Spinelli
* --dataset=A.Photo --model=Spinelli

Below are the commands to run the `benchmark.py` script for the RL-AP-GCN:
* --dataset=Citeseer --model=RL-AP-GCN 
* --dataset=Cora-ML --model=RL-AP-GCN  
* --dataset=PubMed --model=RL-AP-GCN  
* --dataset=MS-Academic --model=RL-AP-GCN  
* --dataset=A.Computer --model=RL-AP-GCN  
* --dataset=A.Photo --model=RL-AP-GCN  

Below are the commands to run the `benchmark.py` script for the Ponder-AP-GCN:
* --dataset=Citeseer --model=Ponder-AP-GCN 
* --dataset=Cora-ML --model=Ponder-AP-GCN 
* --dataset=PubMed --model=Ponder-AP-GCN 
* --dataset=MS-Academic --model=Ponder-AP-GCN 
* --dataset=A.Computer --model=Ponder-AP-GCN 
* --dataset=A.Photo --model=Ponder-AP-GCN 

Below are the commands to run the `benchmark.py` script for the Gumbel-AP-GCN:
* --dataset=Citeseer --model=Gumbel-AP-GCN 
* --dataset=Cora-ML --model=Gumbel-AP-GCN 
* --dataset=PubMed --model=Gumbel-AP-GCN 
* --dataset=MS-Academic --model=Gumbel-AP-GCN 
* --dataset=A.Computer --model=Gumbel-AP-GCN 
* --dataset=A.Photo --model=Gumbel-AP-GCN

Below are the commands to run the `benchmark.py` script for the Co-GCN:
* --dataset=Citeseer --model=Cooperative-AP-GCN
* --dataset=Cora-ML --model=Cooperative-AP-GCN
* --dataset=PubMed --model=Cooperative-AP-GCN
* --dataset=MS-Academic --model=Cooperative-AP-GCN
* --dataset=A.Computer --model=Cooperative-AP-GCN
* --dataset=A.Photo --model=Cooperative-AP-GCN