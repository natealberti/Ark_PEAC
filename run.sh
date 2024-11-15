#!/bin/bash

ENVIRONMENT=Ark_PEAC
module load mamba/latest
source activate $ENVIRONMENT

# PRETRAINED_WEIGHTS=/scratch/nralbert/research/checkpoints/swin_base_patch4_window7_224.pth             ### Swin-B ImageNet-1K weights
PRETRAINED_WEIGHTS_NATE=/scratch/nralbert/CSE507/PEAC_backup/weights
PRETRAINED_WEIGHTS_PETER=/scratch/pmousses/PEAC_weights

### Swin-B PEAC weights (ss pretrained on NIH chest x-rays)
WEIGHTS_FILE_NAME=last_swinb_patchsize32_imgsize448.pth
PRETRAINED_WEIGHTS="$PRETRAINED_WEIGHTS_PETER/$WEIGHTS_FILE_NAME"

OUTPUT_DIR=/scratch/pmousses/research/Ark/output/debug
mkdir -p $OUTPUT_DIR

~/.conda/envs/$ENVIRONMENT/bin/python main_ark.py --data_set ChestMNIST	\
--opt adamw --lr 0.0005 --warmup-epochs 20     	\
--batch_size 128 --model swin_base --init peac  \
--pretrain_epochs 150  --test_epoch 5	\
--pretrained_weights $PRETRAINED_WEIGHTS	\
--momentum_teacher 0.9  --projector_features 1376
