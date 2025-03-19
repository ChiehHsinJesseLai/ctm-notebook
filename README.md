## CTM Training and Evaluation
This repository contains the codebase for Consistency Trajectory Model(CTM) implemented using PyTorch for conducting experiments on LSUN Bedroom images. The latest code is available in branch "ctm_lsun_bedroom_refactor".
 
 
## Prerequisites 
1.Install docker

2.Use the build.sh and launch.sh scripts provided in the docker folder to create and launch the Docker container

3.Use "edm_bedroom256_ema.pt" as the teacher model and "VIRTUAL_lsun_bedroom256.npz" as the reference path for your tasks.
 
## Dataset
1.Download the LSUN Bedroom Dataset: Clone the fyu/lsun repository on GitHub and run their download script

2.Use the LSUN_Bedroom_Download/read_write_lsun.py script to convert images stored in the LMDB database into PNG files and save them to specified directories.

3.Ensure you have the required libraries installed (PIL, lmdb, numpy).
 
## Training
1. For CTM+DSM training, run commands/CTM+DSM_command.sh
Recommendation: at least run CTM+DSM for ~20k iterations

2. For CTM+DSM+GAN training, run commands/CTM+DSM+GAN_command.sh
Recommendation: at least run CTM+DSM+GAN for >=30k iterations
 
##  Evaluation
The evaluation is activated in the training script cm_train.py. If you want to perform the evaluation separately, use cm/evaluator.py.
 
