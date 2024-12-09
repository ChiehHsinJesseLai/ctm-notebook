#!/bin/bash
#SBATCH --partition=sharedp
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --job-name=ctm_lsun_dataset_dgm
#SBATCH --output=logs/test_logs_1_cont.log
#SBATCH --error=logs/test_logs_1_cont.log
#SBATCH --mem=256G
#SBATCH --gres=gpu:h100:1
#SBATCH --priority=1000
#SBATCH --requeue
singularity exec --bind /group/dgm/sard/dataset:/dataset \
    --nv /docker/lsun_data_1.sif \
    /bin/bash -c "python read_write_lsun_test.py"