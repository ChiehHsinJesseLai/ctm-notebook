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

MODEL_FLAGS="--start_ema=0.0 --training_mode=cm --ctm_target_inner_type=target_model_sg --diffusion_training=False --num_heun_step_random=False --num_heun_step=1 --save_check_period=100000000 --save_interval=50000 --log_interval=1000 --check_dm_performance=False --eval_fid=True --eval_similarity=False --eval_interval=50000 --microbatch 16 --global_batch_size=256"

mpiexec -n 4 python3.11 cm_train.py $MODEL_FLAGS --out_dir /groups/gce50978/user/dongjun/EighthArticleExperimentalResults/CIFAR10/CM/default/