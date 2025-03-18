#!/bin/bash

#$-l rt_F=1
#$ -l h_rt=168:00:00
#$-j y
#$-cwd

source /etc/profile.d/modules.sh

module load python/3.11/3.11.2
module load cuda/11.7/11.7.1
module load cudnn/8.9/8.9.2
module load nccl/2.14/2.14.3-1
module load intel-mpi/2021.8

module load hpcx/2.12

mpiexec -n 4 python3.11 cm_train.py --out_dir /home/acf15618av/EighthArticleExperimentalResults/CIFAR10/ablation-V100/CTM/default/ --microbatch 16


mpiexec -n 4 python3.11 cm_train.py --out_dir /home/acf15618av/EighthArticleExperimentalResults/CIFAR10/test/ --microbatch 16 --data_dir=/home/acf15618av/dataset/CIFAR10-cond --class_cond=True

cat $SGE_JOB_HOSTLIST > ./${JOB_ID}_hostfile

export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export RANK=$OMPI_COMM_WORLD_RANK
export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK

HOST=${HOSTNAME:0:5}
NUM_NODES=${NHOSTS}
NUM_GPUS_PER_NODE=4 # 4 for V, 8 for A
MPIOPTS="-N ${NUM_GPUS_PER_NODE} -x MASTER_ADDR=${HOSTNAME}" # --map-by ppr:${NUM_GPUS_PER_NODE}:node
mpirun  ${MPIOPTS} --hostfile ./${JOB_ID}_hostfile python3.11 cm_train.py --out_dir /home/acf15618av/EighthArticleExperimentalResults/CIFAR10/test/ --log_interval=10 --check_dm_performance=False --eval_interval=10000

mpirun -np 2 -n 4 -map-by ppr:1:node -hostfile $SGE_JOB_HOSTLIST python3.11 cm_train.py --out_dir /home/acf15618av/EighthArticleExperimentalResults/CIFAR10/test/ --microbatch=11 --global_batch_size=88 --log_interval=10 --check_dm_performance=False --eval_interval=10000 --use_MPI=True

mpiexec -n 4 python3.11 cm_train.py --out_dir /home/acf15618av/EighthArticleExperimentalResults/CIFAR10/test/ --microbatch=11 --global_batch_size=88 --log_interval=10 --check_dm_performance=False --eval_interval=10000

mpiexec -n 4 python3.11 cm_train.py --out_dir /home/acf15618av/EighthArticleExperimentalResults/CIFAR10/test_/ --eval_interval=2 --microbatch=11 --global_batch_size=132 --gan_training=True --g_learning_period=2 --resume_checkpoint=/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/resume_checkpoint/euler_edm_stop_grad_random_17_M_bs_384_from_edm_dsm_1.0/model050000.pt