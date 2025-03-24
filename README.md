# CTM: Consistency Trajectory Model
This repository contains the codebase for [Consistency Trajectory Model(CTM)](https://arxiv.org/pdf/2310.02279) implemented using PyTorch for conducting experiments on LSUN Bedroom images. 

## Introduction
CTM is an advanced framework to enhance the efficiency and quality of diffusion model sampling. It generalizes consistency models and score-based models, enabling a single neural network to output scores and traverse between any initial and final time along the Probability Flow Ordinary Differential Equation (ODE) in a diffusion process.

__Result Examples__

  ![](sample.png) 


__Evaluation Results__

Evaluation has been done on lsun bedroom validation dataset 
Evaluation metrics used are:<br>
FID(Frechet Inception Distance), Recall and Precision 


| | FID| Recall| Precision| 
|---|---|---|---|
| CTM | 1.89 | 51.88 % | 63.33 %|


## Prerequisites
1.Install docker

2.Use the build.sh and launch.sh scripts provided in the docker folder to create and launch the Docker container

3.Download "edm_bedroom256_ema.pt" as the teacher model and "VIRTUAL_lsun_bedroom256.npz" as the reference path for your tasks. (Need update the links)

## Inference
Download our checkpoint [here](link to released checkpoiny)

[Add inference commands]


## Dataset preparation
1.Download the LSUN Bedroom Dataset: Clone the fyu/lsun repository on GitHub and run their download script

2.Use the LSUN_Bedroom_Download/read_write_lsun.py script to convert images stored in the LMDB database into PNG files and save them to specified directories.

3.Ensure you have the required libraries installed (PIL, lmdb, numpy).
 
## Training
Git clone the CTM github repo. Add correct folder paths and parameters in the below command scripts,then perform the training as given below.

1. For CTM+DSM training, run bash commands/CTM+DSM_command.sh<br>

   Recommendation: at least run bash CTM+DSM for ~20k iterations

2. For CTM+DSM+GAN training, run commands/CTM+DSM+GAN_command.sh<br>

   Recommendation: at least run CTM+DSM+GAN for >=30k iterations

