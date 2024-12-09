#!/bin/bash
MODEL_FLAGS="--data_name=afhq --ema_rate=0.999 --class_cond=False --eval_interval=10000000 --save_interval=5000 --num_classes=0 --eval_batch=16 --eval_fid=False --eval_similarity=False --check_dm_performance=False --log_interval=100 --intermediate_samples=False --sample_interval=10 --sampling_batch=2 --use_scale_shift_norm=False --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --ctm_training=False --apply_adaptive_weight=False --diffusion_schedule_sampler=lognormal --training_mode=edm"
CKPT_FLAGS="--data_dir=/home/ubuntu/dataset/AFHQ/train/cat --resume_checkpoint=/home/ubuntu/EighthArticleExperimentalResults/AFHQ/DM/model065000.pt"
export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_LOCAL_RANK=0
export OMPI_COMM_WORLD_SIZE=8

nohup mpiexec -n 8 --allow-run-as-root python cm_train.py $MODEL_FLAGS $CKPT_FLAGS --out_dir /home/ubuntu/EighthArticleExperimentalResults/AFHQ/DM/ --microbatch=5 --global_batch_size=200 --lr=0.0001 > log_train_edm.txt 2>&1 &

docker exec -it ctm-cifar10-best bash
conda activate ctm
cd
cd ..
cd home
cd ubuntu
cd EighthArticle
cd consistency_models-main_ver3/
MODEL_FLAGS="--data_name=afhq --ema_rate=0.999 --class_cond=False --eval_interval=10000000 --save_interval=1000 --num_classes=0 --eval_batch=16 --eval_fid=False --eval_similarity=False --check_dm_performance=False --log_interval=10 --intermediate_samples=False --sample_interval=10 --sampling_batch=2 --use_scale_shift_norm=False --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --training_mode=ctm"
CKPT_FLAGS="--teacher_model_path=/home/ubuntu/EighthArticleExperimentalResults/AFHQ/DM/ema_0.999_065000.pt --data_dir=/home/ubuntu/dataset/AFHQ/train/cat --resume_checkpoint=/home/ubuntu/EighthArticleExperimentalResults/AFHQ/CTM_with_adaptive_/model003000.pt"




CUDA_VISIBLE_DEVICES=4,5,6,7 mpiexec -n 4 --allow-run-as-root python cm_train.py $MODEL_FLAGS $CKPT_FLAGS --out_dir /home/ubuntu/EighthArticleExperimentalResults/AFHQ/CTM_without_adaptive/ --microbatch=2 --global_batch_size=16 --lr=0.00001 --training_mode=ctm --num_heun_step=10 --heun_step_strategy=uniform --save_png=True --save_period=100 --apply_adaptive_weight=False --device_id=0




MODEL_FLAGS="--data_name=afhq --ema_rate=0.999 --class_cond=False --eval_interval=10000000 --save_interval=5000 --num_classes=0 --eval_batch=16 --eval_fid=False --eval_similarity=False --check_dm_performance=False --log_interval=10 --intermediate_samples=False --sample_interval=10 --sampling_batch=2 --use_scale_shift_norm=False --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --training_mode=ctm"
CKPT_FLAGS="--teacher_model_path=/home/ubuntu/EighthArticleExperimentalResults/AFHQ/DM/ema_0.999_065000.pt --data_dir=/home/ubuntu/dataset/AFHQ/train/cat --resume_checkpoint=/home/ubuntu/EighthArticleExperimentalResults/AFHQ/CTM_with_adaptive_M39/model014000.pt"

mpiexec -n 8 --allow-run-as-root python cm_train.py $MODEL_FLAGS $CKPT_FLAGS --out_dir /home/ubuntu/EighthArticleExperimentalResults/AFHQ/CTM_with_adaptive_M39_v2/ --microbatch=2 --global_batch_size=16 --lr=0.00001 --training_mode=ctm --num_heun_step=10 --heun_step_strategy=uniform --apply_adaptive_weight=True --diffusion_training_frequency=0.1 --device_id=0 --use_MPI=True --map_location=cuda