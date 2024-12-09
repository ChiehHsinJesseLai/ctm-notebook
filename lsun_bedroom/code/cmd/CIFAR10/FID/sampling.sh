#!/bin/bash

#$-l rt_F=1
#$ -l h_rt=168:00:00
#$-j y
#$-cwd

qrsh -g gce50978 -l rt_F=1 -l h_rt=12:00:00
source /etc/profile.d/modules.sh

module load python/3.11/3.11.2
module load cuda/11.7/11.7.1
module load cudnn/8.9/8.9.2
module load nccl/2.14/2.14.3-1
module load intel-mpi/2021.8

MODEL_FLAGS="--start_ema=0.999 --save_check_period=1000 --eval_interval=5000 --eval_fid=True --eval_similarity=False --check_dm_performance=False --compute_ema_fids=True --gan_fake_inner_type=model --gan_fake_outer_type=target_model_sg --gan_training=True --g_learning_period=2 --resume_checkpoint=/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/resume_checkpoint/euler_edm_stop_grad_random_17_M_bs_384_from_edm_dsm_1.0/model050000.pt --microbatch=11"

python3.10 image_sample.py $MODEL_FLAGS --out_dir /home/dongjun/EighthArticleExperimentalResults/CIFAR10/good/ --model_path=/home/dongjun/EighthArticleExperimentalResults/CIFAR10/good/ema_0.999_060999.pt --eval_num_samples=16 --batch_size=16 --device_id=0 --sampler=gamma --sampling_steps=2 --stochastic_seed=True --save_format=png --ind_1=5 --ind_2=3

python3.11 image_sample.py $MODEL_FLAGS --out_dir /groups/gce50978/user/dongjun/EighthArticleExperimentalResults/CIFAR10/quite_good --model_path=/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/ablation-V100/GAN/fine_tune_ver3/GAN_bs_264/ema_0.999_065999.pt --eval_num_samples=50000 --batch_size=2000 --device_id=0 --sampler=gamma --sampling_steps=2 --stochastic_seed=True --save_format=npz --ind_1=5 --ind_2=3

nohup python3.11 image_sample.py $MODEL_FLAGS --out_dir /groups/gce50978/user/dongjun/EighthArticleExperimentalResults/CIFAR10/GAN/cond/GAN_bs_528_ema_0.999 --model_path=/groups/gce50978/user/dongjun/EighthArticleExperimentalResults/CIFAR10/GAN/cond/GAN_bs_528_ema_0.999/ema_0.999_039000.pt --eval_num_samples=50000 --batch_size=1000 --stochastic_seed=True --save_format=npz --ind_1=5 --ind_2=3 --class_cond=True --use_MPI=True --sampler=exact --sampling_steps=1 --device_id=0 > log.txt 2>&1 &
nohup python3.11 image_sample.py $MODEL_FLAGS --out_dir /groups/gce50978/user/dongjun/EighthArticleExperimentalResults/CIFAR10/GAN/uncond/GAN_bs_264_ema_0.9999_diff_aug --model_path=/groups/gce50978/user/dongjun/EighthArticleExperimentalResults/CIFAR10/GAN/uncond/GAN_bs_264_ema_0.9999/model081000.pt --eval_num_samples=50000 --batch_size=1000 --stochastic_seed=True --save_format=npz --ind_1=1 --ind_2=0 --class_cond=False --use_MPI=True --sampler=heun --sampling_steps=18 --device_id=1 > log.txt 2>&1 &

python fid_npzs.py --ref=/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/author_ckpt/cifar10-32x32.npz --num_samples=50000 --batch_size=500 --device=cuda:0 --images=/groups/gce50978/user/dongjun/EighthArticleExperimentalResults/CIFAR10/GAN/uncond/GAN_bs_528_ema_0.9999_diff_aug/
python fid_npzs.py --ref=/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/author_ckpt/cifar10-32x32.npz --num_samples=50000 --batch_size=500 --device=cuda:0 --images=/groups/gce50978/user/dongjun/EighthArticleExperimentalResults/CIFAR10/GAN/cond/GAN_bs_528_ema_0.999/
python fid_npzs.py --ref=/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/author_ckpt/cifar10-32x32.npz --num_samples=50000 --images=/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/resume_checkpoint/euler_edm_stop_grad_random_17_M_bs_384_from_edm_dsm_1.0/ --batch_size=500 --device=cuda:0
python fid_npzs.py --ref=/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/author_ckpt/cifar10-32x32.npz --num_samples=50000 --batch_size=500 --device=cuda:0 --images=/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/resume_checkpoint/euler_edm_stop_grad_random_17_M_bs_384_from_edm_dsm_1.0/

MODEL_FLAGS="--start_ema=0.9999 --save_check_period=1000 --eval_interval=5000 --eval_fid=True --eval_similarity=False --check_dm_performance=False --compute_ema_fids=True --gan_fake_inner_type=model --gan_fake_outer_type=target_model_sg --gan_training=True --g_learning_period=2 --resume_checkpoint=/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/resume_checkpoint/euler_edm_stop_grad_random_17_M_bs_384_from_edm_dsm_1.0/model050000.pt --microbatch=11"
python3.11 eval_bpd.py $MODEL_FLAGS --out_dir /groups/gce50978/user/dongjun/EighthArticleExperimentalResults/CIFAR10/GAN/uncond/GAN_bs_264_ema_0.9999_diff_aug --resume_checkpoint=/groups/gce50978/user/dongjun/EighthArticleExperimentalResults/CIFAR10/GAN/uncond/GAN_bs_264_ema_0.9999_diff_aug/ema_0.999_060000.pt --device_id=0 --class_cond=False --use_MPI=True --global_batch_size=100 --batch_size=100
python3.11 eval_bpd.py $MODEL_FLAGS --out_dir /groups/gce50978/user/dongjun/EighthArticleExperimentalResults/CIFAR10/GAN/uncond/GAN_bs_264_ema_0.9999_diff_aug --resume_checkpoint=/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/resume_checkpoint/euler_edm_stop_grad_random_17_M_bs_384_from_edm_dsm_1.0/ema_0.999_050000.pt --device_id=0 --class_cond=False --use_MPI=True --global_batch_size=100 --batch_size=100

python3.11 image_sample.py $MODEL_FLAGS --out_dir /groups/gce50978/user/dongjun/EighthArticleExperimentalResults/CIFAR10/GAN/uncond/GAN_bs_264_ema_0.999 --model_path=/groups/gce50978/user/dongjun/EighthArticleExperimentalResults/CIFAR10/GAN/uncond/GAN_bs_264_ema_0.999/ema_0.999_064000.pt --eval_num_samples=50000 --batch_size=1000 --device_id=0 --sampler=exact --sampling_steps=1 --stochastic_seed=True --save_format=npz --ind_1=5 --ind_2=3 --class_cond=True --use_MPI=True
nohup python3.11 image_sample.py $MODEL_FLAGS --out_dir /groups/gce50978/user/dongjun/EighthArticleExperimentalResults/CIFAR10/GAN/uncond/GAN_bs_264_ema_0.999/ --model_path=/groups/gce50978/user/dongjun/EighthArticleExperimentalResults/CIFAR10/GAN/uncond/GAN_bs_264_ema_0.999/ema_0.999_071000.pt --eval_num_samples=50000 --batch_size=1000 --stochastic_seed=True --save_format=npz --ind_1=1 --ind_2=0 --class_cond=False --use_MPI=True --device_id=0 --sampler=heun --sampling_steps=18 > log.txt 2>&1 &
MODEL_FLAGS="--start_ema=0.999 --save_check_period=1000 --eval_interval=5000 --eval_fid=True --eval_similarity=False --check_dm_performance=False --compute_ema_fids=True --gan_fake_inner_type=model --gan_fake_outer_type=target_model_sg --gan_training=True --g_learning_period=2 --resume_checkpoint=/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/resume_checkpoint/euler_edm_stop_grad_random_17_M_bs_384_from_edm_dsm_1.0/model050000.pt --microbatch=11"
python3.11 image_sample.py $MODEL_FLAGS --out_dir /home/acf15618av/EighthArticleExperimentalResults/CIFAR10/resume_checkpoint/euler_edm_stop_grad_random_17_M_bs_384_from_edm_dsm_1.0/ --model_path=/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/resume_checkpoint/euler_edm_stop_grad_random_17_M_bs_384_from_edm_dsm_1.0/ema_0.999_050000.pt --eval_num_samples=50000 --batch_size=1000 --stochastic_seed=True --save_format=npz --ind_1=5 --ind_2=3 --class_cond=False --use_MPI=True --device_id=2 --sampler=exact --sampling_steps=6
nohup python3.11 image_sample.py $MODEL_FLAGS --out_dir /home/acf15618av/EighthArticleExperimentalResults/CIFAR10/resume_checkpoint/euler_edm_stop_grad_random_17_M_bs_384_from_edm_dsm_1.0/ --model_path=/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/resume_checkpoint/euler_edm_stop_grad_random_17_M_bs_384_from_edm_dsm_1.0/ema_0.999_050000.pt --eval_num_samples=50000 --batch_size=1000 --stochastic_seed=True --save_format=npz --class_cond=False --use_MPI=True --sampler=gamma_multistep --gamma=0.9 --sampling_steps=1 --device_id=1 > log.txt 2>&1 &


nohup python3.11 image_sample.py $MODEL_FLAGS --out_dir /home/acf15618av/EighthArticleExperimentalResults/CIFAR10/EDM/ --model_path=/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/author_ckpt/edm-cifar10-32x32-uncond-vp.pkl --training_mode=edm --eval_num_samples=50000 --batch_size=1000 --stochastic_seed=True --save_format=npz --ind_1=5 --ind_2=3 --class_cond=False --use_MPI=True --sampler=heun --device_id=3 --sampling_steps=3 > log.txt 2>&1 &



scp abci:/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/ablation-V100/CTM/DSM_weight/DSM_turn_off_uniform_g/log.txt /hdd/dongjun/EighthArticleExperimentalResults/CIFAR10/ablation-V100/CTM/DSM_weight/DSM_turn_off_uniform_g/
scp abci:/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/ablation-V100/CTM/DSM_weight/DSM_turn_off/log.txt /hdd/dongjun/EighthArticleExperimentalResults/CIFAR10/ablation-V100/CTM/DSM_weight/DSM_turn_off/
scp abci:/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/ablation-V100/CTM/DSM_weight/DSM_1.0/log.txt /hdd/dongjun/EighthArticleExperimentalResults/CIFAR10/ablation-V100/CTM/DSM_weight/DSM_1.0/
scp abci:/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/ablation-V100/CTM/DSM_weight/DSM_0.1/log.txt /hdd/dongjun/EighthArticleExperimentalResults/CIFAR10/ablation-V100/CTM/DSM_weight/DSM_0.1/
scp abci:/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/ablation-V100/CTM/DSM_weight/DSM_5.0/log.txt /hdd/dongjun/EighthArticleExperimentalResults/CIFAR10/ablation-V100/CTM/DSM_weight/DSM_5.0/
scp abci:/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/ablation-V100/CTM/DSM_weight/DSM_1.0_adaptive/log.txt /hdd/dongjun/EighthArticleExperimentalResults/CIFAR10/ablation-V100/CTM/DSM_weight/DSM_1.0_adaptive/
scp abci:/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/ablation-V100/CTM/DSM_weight/DSM_0.1_adaptive/log.txt /hdd/dongjun/EighthArticleExperimentalResults/CIFAR10/ablation-V100/CTM/DSM_weight/DSM_0.1_adaptive/
scp abci:/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/ablation-V100/CTM/DSM_weight/DSM_5.0_adaptive/log.txt /hdd/dongjun/EighthArticleExperimentalResults/CIFAR10/ablation-V100/CTM/DSM_weight/DSM_5.0_adaptive/

module load hpcx/2.12
MODEL_FLAGS="--start_ema=0.999 --save_check_period=1000 --eval_interval=5000 --eval_fid=True --eval_similarity=False --check_dm_performance=False --compute_ema_fids=True --gan_fake_inner_type=model --gan_fake_outer_type=target_model_sg --gan_training=True --g_learning_period=2 --microbatch=11"
python3.11 eval_bpd.py --training_mode ctm --global_batch_size 100 --resume_checkpoint=/groups/gce50978/user/dongjun/EighthArticleExperimentalResults/CIFAR10/GAN/uncond/GAN_bs_264_ema_0.9999/model081000.pt --out_dir=/groups/gce50978/user/dongjun/EighthArticleExperimentalResults/CIFAR10/GAN/uncond/GAN_bs_264_ema_0.9999 --num_student_elbo=0 --num_teacher_elbo=1 --num_student_nll=1 --device_id=0 --use_MPI=True

python3.11 eval_bpd.py --training_mode edm --global_batch_size 100 --teacher_model_path=/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/author_ckpt/edm-cifar10-32x32-uncond-vp.pkl --out_dir=/groups/gce50978/user/dongjun/EighthArticleExperimentalResults/CIFAR10/GAN/uncond/GAN_bs_264_ema_0.9999 --num_student_elbo=0 --num_teacher_elbo=1 --num_student_nll=1 --device_id=0 --use_MPI=True