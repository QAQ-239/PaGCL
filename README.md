# *Code for PaGCL: Privacy-aware Graph Contrastive Learning against Attribute Inference Attacks*

This is the code of our work - PaGCL

## Setup

- We experiment on CUDA 11.1 and torch 1.12.1.
- unzip dataset, run:

  `unzip data.zip`

- To install requirement for the project using conda:

  `conda env create -f environment.yml`

## Get Start

Train and evaluate the model:

1. To implement our experimental results, run:

   `python train.py`

2. If you need to implement a specific dataset, run:

   `python train.py --DS MUTAG --aug dropnodes --lr 0.001`

Hyperparameter explanation:

- `-- DS` dataset
- `-- aug` methods for graph data augmentation
- `-- lr` the learning rate
- `-- batch_size` batch size in memory during training
