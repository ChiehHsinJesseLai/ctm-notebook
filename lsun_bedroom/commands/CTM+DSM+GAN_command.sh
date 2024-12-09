#!/bin/bash
export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_LOCAL_RANK=0
export OMPI_COMM_WORLD_SIZE=8
# export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_FLAGS="--discriminator_weight=1.0 --gan_specific_time=True --microbatch=3 --global_batch_size=512 --lr=0.00001 --data_name=lsun_bedroom --class_cond=False --start_ema=0.999 --gan_different_augment=True --eval_interval=1000 --save_interval=1000 --num_classes=0 --eval_batch=64 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=1 --compute_ema_fids=False --gan_fake_inner_type=model --gan_fake_outer_type=target_model_sg --gan_training=True --g_learning_period=2 --use_MPI=False --num_workers=0 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm False"
CKPT_FLAGS="--out_dir /ctm/ctm_results/Exp14_ctm_dsm_gan_bs_512_8gpus_5 --ref_path=/ctm/ctm_code/ctm_lsun_bedroom/author_ckpt/VIRTUAL_lsun_bedroom256.npz --resume_checkpoint=/ctm/ctm_checkpoints/model029000.pt --teacher_model_path=/ctm/ctm_code/ctm_lsun_bedroom/author_ckpt/edm_bedroom256_ema.pt --data_dir=/ctm/dataset/LSUN_Bedroom/train_orig_imgs --total_training_steps=51000"

mpiexec -n 8 --allow-run-as-root python3 /ctm_lsun_bedroom/code/cm_train.py $MODEL_FLAGS $CKPT_FLAGS