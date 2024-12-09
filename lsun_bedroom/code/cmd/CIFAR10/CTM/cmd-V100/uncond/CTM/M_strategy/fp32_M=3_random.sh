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
module load hpcx/2.12

mpiexec -n 4 python3.11 cm_train.py --out_dir /home/acf15618av/EighthArticleExperimentalResults/CIFAR10/ablation-V100/CTM/M_strategy/fp32_M=3_random/ --microbatch 16 --num_heun_step=3 --use_fp16=False