#!/bin/bash
source /etc/profile.d/modules.sh
module load singularity/3.5.3
module load openmpi/3.1.6
singularity run --nv /sample/nnabla/container/nvcr-nnabla1.15.0-mpi3.1.6-ib.simg

export OMPI_COMM_WORLD_RANK=0
export OMPI_COMM_WORLD_LOCAL_RANK=0
export OMPI_COMM_WORLD_SIZE=1

MODEL_FLAGS="--apply_adaptive_weight=False --microbatch=16 --global_batch_size=64 --self_learn=True --eval_interval=5000 --eval_num_samples=2000 --eval_fid=True --eval_similarity=False --check_dm_performance=False --log_interval=100"
CKPT_FLAGS="--ref_path=/home/fp084243/EighthArticleExperimentalResults/CIFAR10/author_ckpt/cifar10-32x32.npz --teacher_model_path=/home/fp084243/EighthArticleExperimentalResults/CIFAR10/author_ckpt/edm-cifar10-32x32-uncond-vp.pkl --data_dir=/group/project142/dataset/CIFAR10/train"

python3.7 cm_train.py $MODEL_FLAGS $CKPT_FLAGS --out_dir /home/fp084243/EighthArticleExperimentalResults/CIFAR10/test --microbatch=4 --global_batch_size=4 --eval_interval=101 --eval_num_samples=500 --eval_fid=False --eval_similarity=False --check_dm_performance=False --log_interval=10