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

ENVIRONMENT_NAME=$(grep -E '^name:' environment.yml | awk '{print $2}')
module load mamba/latest
if mamba info --envs | grep -qE "^\s*$ENVIRONMENT_NAME\s"; then :
else
	mamba env create -f environment.yml
fi
source activate $ENVIRONMENT_NAME

# PRETRAINED_WEIGHTS=/scratch/nralbert/research/checkpoints/swin_base_patch4_window7_224.pth             ### Swin-B ImageNet-1K weights
PRETRAINED_WEIGHTS_NATE=/scratch/nralbert/CSE507/PEAC_backup/weights
PRETRAINED_WEIGHTS_PETER=/scratch/pmousses/PEAC_weights

### Swin-B PEAC weights (ss pretrained on NIH chest x-rays)
WEIGHTS_FILE_NAME=last_swinb_patchsize32_imgsize448.pth
PRETRAINED_WEIGHTS="$PRETRAINED_WEIGHTS_PETER/$WEIGHTS_FILE_NAME"

OUTPUT_DIR=/scratch/$USER/research/Ark/output/debug
mkdir -p $OUTPUT_DIR

/home/$USER/.conda/envs/$ENVIRONMENT_NAME/bin/python \
	main_ark.py \
		--data_set ChestMNIST	\
		--opt adamw --lr 0.0005 --warmup-epochs 20     	\
		--batch_size 128 --model swin_base --init peac  \
		--pretrain_epochs 150  --test_epoch 5	\
		--pretrained_weights $PRETRAINED_WEIGHTS	\
		--momentum_teacher 0.9  --projector_features 1376
