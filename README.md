# CommunitGAN

- This repository is the implementation of CommunityGAN:
> CommunityGAN: Community Detection with Generative Adversarial Nets

## Files in the folder
- `data/`: graph and community data
- `pre_train/`: pre-trained vertex embeddings
  > Note: the dimension of pre-trained vertex embeddings should equal n_emb in src/CommunityGAN/config.py
- `results/`: evaluation results and the learned embeddings of the generator and the discriminator
- `src/`: source codes for CommunityGAN and pre-train model

## Data format

The input file for CommunityGAN (data/community_detection/\*_train.txt): An undirected graph in which vertex IDs start from *0* to *N-1* (*N* is the number of nodes in the graph). Each line contains two node IDs indicating an edge in the graph.

The input file for the pre-train model (data/community_detection/\*_agm.txt): Similar to the input file for CommunityGAN. The only difference is that in this file one edge need occur twice: ```node1 node2``` and ```node2 node1```

The community file (data/community_detection/\*.sampled.cmty.txt): Each line means a community and the numbers indicate the vertices in the community.

## CommunityGAN

#### Requirements
The code of CommunityGAN has been tested running under Python 3.6.1, with the following packages installed (along with their dependencies):

- tensorflow == 1.6.0
- numpy == 1.12.1
- scipy == 1.1.0

#### Basic usage
The basic usage of CommunityGAN is as follow:

```cd src/CommunityGAN```  
```python community_gan.py```

The parameters for CommunityGAN can be changed by editing `src/CommunityGAN/config.py` or passing through command line. An example for running CommunityGAN on the three datasets are written in `scripts/run.py`, which can be called by:

```cd scripts```  
```python run.py```


## Pre-train model

#### Compile
Get into the `src/PreTrain` directory and use `make` command to compile. Has been tested on Ubuntu with g++ 5.3.0, and on Windows with MinGW-w64 5.3.0.

#### Basic usage
A basic usage example of the pre-train model has been written in ```scripts/prepare_pretrain_embedding.py```. The following commands can be used to re-prepare the pre-train embeddings for CommunityGAN on the three datasets:

```cd scripts```  
```python prepare_pretrain_embedding.py```
