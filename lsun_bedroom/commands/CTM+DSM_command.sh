#!/bin/bash

MODEL_FLAGS="--data_name=lsun_bedroom --microbatch=4 --global_batch_size=300 --lr=0.00001 --class_cond=False --eval_interval=1 --save_interval=100 --num_classes=0 --eval_batch=64 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=1 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm False --sanity_check True --num_workers 0 --large_log True --batch_size 30"
CKPT_FLAGS="--out_dir /ctm/ctm_results/Exp13_ctm_bs_1024_8GPUS_DGM_eval_dryrun --ref_path=/ctm/ctm_code/ctm_lsun_bedroom/author_ckpt/VIRTUAL_lsun_bedroom256.npz --teacher_model_path=/ctm/ctm_code/ctm_lsun_bedroom/author_ckpt/edm_bedroom256_ema.pt --data_dir=/ctm/dataset/LSUN_Bedroom/val_orig_imgs --resume_checkpoint=/ctm/ctm_checkpoints/model019000.pt --total_training_steps=21000"

export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_LOCAL_RANK=0
export OMPI_COMM_WORLD_SIZE=2

mpiexec -n 2 --allow-run-as-root python3 /ctm_lsun_bedroom/code/cm_train.py $MODEL_FLAGS $CKPT_FLAGS