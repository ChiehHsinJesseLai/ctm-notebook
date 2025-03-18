#!/bin/bash

#$-l rt_AG.small=1
#$ -l h_rt=72:00:00
#$-j y
#$-cwd

source /etc/profile.d/modules.sh

module load python/3.10/3.10.10
module load cuda/12.1/12.1.1
module load cudnn/8.9/8.9.2
module load nccl/2.18/2.18.1-1
module load intel-mpi/2021.8

python3.10 cm_train.py --out_dir /home/acf15618av/EighthArticleExperimentalResults/CIFAR10/ablation/M=3/ --num_heun_step=3 --num_heun_step_random=False