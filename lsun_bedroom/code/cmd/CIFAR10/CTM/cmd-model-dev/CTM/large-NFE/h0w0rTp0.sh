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

MODEL_FLAGS="--use_fp16=False --target_subtract=True --eval_interval=10000 --eval_num_samples=10000 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=10 --microbatch 8 --global_batch_size 32"

mpiexec -n 4 python3.11 cm_train.py $MODEL_FLAGS --heun_step_multiplier=0.0 --weight_schedule=uniform_g --weight_schedule_multiplier=0.0 --rescaling=True --diffusion_training_frequency=0.0 --diffusion_training=False --out_dir /home/acf15618av/EighthArticleExperimentalResults/CIFAR10/large_NFE/h0w0rTp0/