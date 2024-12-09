#!/bin/bash

#$-l rt_F=1
#$ -l h_rt=168:00:00
#$-j y
#$-cwd

source /etc/profile.d/modules.sh

module load python/3.11/3.11.2
module load cuda/11.7/11.7.1
module load cudnn/8.9/8.9.2
module load nccl/2.14/2.14.3-1
module load intel-mpi/2021.8

mpiexec -n 4 python3.11 cm_train.py --out_dir /home/acf15618av/EighthArticleExperimentalResults/CIFAR10/ablation-V100/CTM/estimate_target/model_target_model_sg_target_model_sg/ --microbatch 16 --ctm_estimate_inner_type=model --ctm_estimate_outer_type=target_model_sg --ctm_target_inner_type=target_model_sg