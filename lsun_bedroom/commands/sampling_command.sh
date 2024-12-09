#!/bin/bash
export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_LOCAL_RANK=0
export OMPI_COMM_WORLD_SIZE=1

MODEL_FLAGS="--data_name=lsun_bedroom --class_cond=False --eval_interval=1000 --save_interval=1000 --num_classes=0 --eval_batch=250 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=100"

CUDA_VISIBLE_DEVICES=0 
mpiexec -n 1 --allow-run-as-root python /home/pranay_sard/CTM_project/CTM/code/image_sample.py $MODEL_FLAGS --class_cond=False --num_classes=0 --out_dir /home/pranay_sard/CTM_project/CTM/LSUN_Bedroom_Output --model_path=/home/pranay_sard/CTM_project/CTM/author_ckpt/cd_bedroom256_lpips.pt --training_mode=edm --class_cond=False --eval_num_samples=6400 --batch_size=800 --device_id=0 --stochastic_seed=True --save_format=npz --ind_1=36 --ind_2=20 --use_MPI=True --sampler=exact --sampling_steps=1 > log.txt