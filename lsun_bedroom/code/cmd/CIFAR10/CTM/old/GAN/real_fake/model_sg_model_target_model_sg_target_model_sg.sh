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

python3.10 cm_train.py --out_dir /home/acf15618av/EighthArticleExperimentalResults/CIFAR10/ablation/GAN_model_sg_model_target_model_sg_target_model_sg/  --gan_fake_inner_type=model_sg --gan_fake_outer_type=model --gan_training=True --gan_real_inner_type=target_model_sg --gan_real_free=False --g_learning_period=2 --resume_checkpoint=/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/resume_checkpoint/euler_edm_stop_grad_random_17_M_bs_384_from_edm_dsm_1.0/model050000.pt --microbatch=48 --global_batch_size=144