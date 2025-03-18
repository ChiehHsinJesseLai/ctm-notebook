#!/bin/bash

MODEL_FLAGS="--data_name=lsun_bedroom --microbatch=4 --global_batch_size=512 --lr=0.00001 --class_cond=False --eval_interval=1000 --save_interval=1000 --num_classes=0 --eval_batch=64 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=1 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm False --sanity_check True --num_workers 0 --large_log True --batch_size 30"
CKPT_FLAGS="--out_dir /ctm/lsuncode_refactor --ref_path=/ctm/author_ckpt/VIRTUAL_lsun_bedroom256.npz --teacher_model_path=/ctm/author_ckpt/edm_bedroom256_ema.pt --data_dir=/ctm/dataset/LSUN_Bedroom/train_orig_imgs  --total_training_steps=21000"

export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_LOCAL_RANK=0
export OMPI_COMM_WORLD_SIZE=8

mpiexec -n 8 --allow-run-as-root python3 /home/lsun/ctm/code/cm_train.py $MODEL_FLAGS $CKPT_FLAGS
