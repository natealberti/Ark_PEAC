#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 20           # number of cores 
#SBATCH --mem=64G		# Memory in GiB
#SBATCH -G a100:1       # number of gpus
#SBATCH -t 1-00:00:00   # time in d-hh:mm:ss
#SBATCH --account=class_ImageSummerFall2024

#SBATCH -p general      # partition 
#SBATCH -q public       # QOS
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="%u@asu.edu"
#SBATCH --export=NONE   # Purge the job-submitting shell environment

ENV_INSTALLED=${1:-true}
ENVIRONMENT=${2:-"Ark_PEAC"}
module load mamba/latest
if [ "$ENV_INSTALLED" = false ]; then
	mamba env create -f environment.yml
	ENVIRONMENT=Ark_PEAC
fi
source activate $ENVIRONMENT

# PRETRAINED_WEIGHTS=/scratch/nralbert/research/checkpoints/swin_base_patch4_window7_224.pth             ### Swin-B ImageNet-1K weights
PRETRAINED_WEIGHTS_NATE=/scratch/nralbert/research/checkpoints
PRETRAINED_WEIGHTS_PETER=/scratch/pmousses/PEAC_weights

### Swin-B PEAC weights (ss pretrained on NIH chest x-rays)
WEIGHTS_FILE_NAME=last_swinb_patchsize32_imgsize448.pth
PRETRAINED_WEIGHTS="$PRETRAINED_WEIGHTS_PETER/$WEIGHTS_FILE_NAME"

OUTPUT_DIR=/scratch/$USER/research/Ark/output/debug
mkdir -p $OUTPUT_DIR

~/.conda/envs/$ENVIRONMENT/bin/python main_ark.py --data_set ChestMNIST	\
--opt adamw --lr 1e-4  	\
--batch_size 24 --model swin_base --pretrained_weights $PRETRAINED_WEIGHTS --init peac \
--pretrain_epochs 300  --test_epoch 5 --sched cosine	\
--momentum_teacher 0.9
