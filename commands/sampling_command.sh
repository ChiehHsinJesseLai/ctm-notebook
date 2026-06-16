#!/bin/bash
export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_LOCAL_RANK=0
export OMPI_COMM_WORLD_SIZE=1

MODEL_FLAGS="--data_name=lsun_bedroom --class_cond=False --num_classes=0 \
             --eval_fid=True --eval_similarity=False --check_dm_performance=False"

ARCH_FLAGS="--image_size=256 --num_channels=256 --num_res_blocks=2 \
            --use_scale_shift_norm=False --resblock_updown=True \
            --attention_resolutions=32,16,8 --learn_sigma=False \
            --num_head_channels=64 --dropout=0.1 \
            --out_dir=./results \
            --model_path=./ctm_checkpoint.pt \
            --training_mode=ctm \
            --eval_num_samples=1 \
            --batch_size=1 \
            --device_id=0 \
            --stochastic_seed=True \
            --save_format=npz \
            --ind_1=36 \
            --ind_2=20 \
            --use_MPI=True \
            --sampler=exact \
            --sampling_steps=1"

CUDA_VISIBLE_DEVICES=0 
mpiexec -n 1 --allow-run-as-root python /home/sard/CTM_project/CTM/code/image_sample.py $MODEL_FLAGS $ARCH_FLAGS