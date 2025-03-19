## CTM Training and Evaluation
This repository contains the codebase for Consistency Trajectory Model(CTM) implemented using PyTorch for conducting experiments on LSUN Bedroom images
 
 
## Prerequisites 
1.Install docker. 
2.create and launch container. The build.sh and launch.sh scripts are given in docker folder
3.Use "edm_bedroom256_ema.pt" as teacher model and "VIRTUAL_lsun_bedroom256.npz" as ref_path
 
## Dataset
Use LSUN bedroom dataset. Download the lsun dataset from official site. Then arrange the data using LSUN_Bedroom_Download/read_write_lsun.py 
 
## Training
For CTM+DSM training, run commands/CTM+DSM_command.sh
Recommendation: at least run CTM+DSM for ~20k iterations

For CTM+DSM+GAN training, run commands/CTM+DSM+GAN_command.sh
Recommendation: at least run CTM+DSM+GAN for >=30k iterations
 
##  Evaluation
The evaluation is activated in the training script cm_train.py. If you want to perform the evaluation separately, use cm/evaluator.py.
 
