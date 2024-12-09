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

MODEL_FLAGS="--apply_adaptive_weight=False --microbatch=16 --global_batch_size=128 --self_learn=True --eval_interval=10000 --eval_num_samples=10000 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=1000"

mpiexec -n 4 python3.11 cm_train.py $MODEL_FLAGS --out_dir /home/acf15618av/EighthArticleExperimentalResults/CIFAR10/self_learn/CTM/CTM_self_M=1_dsm_no_adaptive/ --num_heun_step=1 --num_heun_step_random=False