#!/bin/bash

#$-l rt_F=1
#$ -l h_rt=168:00:00
#$-j y
#$-cwd

# source /etc/profile.d/modules.sh

# module load python/3.11/3.11.2
# module load cuda/11.7/11.7.1
# module load cudnn/8.9/8.9.2
# module load nccl/2.14/2.14.3-1
# module load hpcx/2.12


MODEL_FLAGS="--batch_size=4 --microbatch=2 --gan_different_augment=True --start_ema=0.9999 --save_interval=10 --eval_interval=10 --eval_fid=False --eval_similarity=True --check_dm_performance=True --compute_ema_fids=True --gan_fake_inner_type=model --gan_fake_outer_type=target_model_sg --gan_training=True --g_learning_period=2 --self_learn=True --ref_path=/workspace/ctm/ctm-cifar10_cond/weight/cifar10-32x32.npz --data_dir=/workspace/ctm/cifar10_images_toy --gpu_usage=True"
CKPT_FLAGS="--out_dir /workspace/ctm/test/ctm-cifar10/GAN/uncond/GAN_bs_528_ema_0.9999_diff_aug/ --resume_checkpoint=/workspace/ctm/test/ctm-cifar10/unconditional/model095000.pt"

torchrun /workspace/ctm/test/ctm-cifar10/cm_train.py $MODEL_FLAGS $CKPT_FLAGS

