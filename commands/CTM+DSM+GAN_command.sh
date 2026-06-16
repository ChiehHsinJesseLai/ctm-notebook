#!/bin/bash

MODEL_FLAGS="--discriminator_weight=1.0 --gan_specific_time=True --microbatch=4 --global_batch_size=512 --lr=0.00001 --data_name=lsun_bedroom --class_cond=False --start_ema=0.999 --gan_different_augment=True --eval_interval=1000 --save_interval=1000 --num_classes=0 --eval_batch=64 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=1 --compute_ema_fids=False --gan_fake_inner_type=model --gan_fake_outer_type=target_model_sg --gan_training=True --g_learning_period=2 --use_MPI=False --num_workers=0 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm False"

CKPT_FLAGS="--out_dir ./results-gan --ref_path=./VIRTUAL_lsun_bedroom256.npz --teacher_model_path=./edm_bedroom256_ema.pt --resume_checkpoint=./results/model021000.pt --data_dir=./LSUN_Bedroom/train_orig_imgs  --total_training_steps=50000"

export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_LOCAL_RANK=0
export OMPI_COMM_WORLD_SIZE=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mpiexec -n 4 --allow-run-as-root python3 /home/lsun/ctm/code/cm_train.py $MODEL_FLAGS $CKPT_FLAGS
