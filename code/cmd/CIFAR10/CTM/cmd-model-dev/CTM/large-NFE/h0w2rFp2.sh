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

MODEL_FLAGS="--use_fp16=False --target_subtract=True --apply_adaptive_weight=False --eval_interval=10000 --eval_num_samples=10000 --eval_fid=True --eval_similarity=False --check_dm_performance=False --microbatch 8 --global_batch_size 32"

mpiexec -n 4 python3.11 cm_train.py $MODEL_FLAGS --heun_step_multiplier=0.0 --heun_step_strategy=uniform --weight_schedule=uniform_g --weight_schedule_multiplier=2.0 --rescaling=False --diffusion_training_frequency=0.2 --out_dir /home/acf15618av/EighthArticleExperimentalResults/CIFAR10/large_NFE/h0w2rFp2/