#!/bin/bash

module load mamba/latest
source activate [ENVIRONMENT]

# PRETRAINED_WEIGHTS=/scratch/nralbert/research/checkpoints/swin_base_patch4_window7_224.pth             ### Swin-B ImageNet-1K weights
PRETRAINED_WEIGHTS=/scratch/nralbert/CSE507/PEAC_backup/weights/last_swinb_patchsize32_imgsize448.pth    ### Swin-B PEAC weights (ss pretrained on NIH chest x-rays)

OUTPUT_DIR=/scratch/nralbert/research/Ark/output/debug

~/.conda/envs/[ENVIRONMENT]/bin/python main_ark.py --data_set ChestXray14	\
--opt sgd --warmup-epochs 20  --lr 0.3     	\
--batch_size 200 --model swin_base --init peac  \
--pretrain_epochs 200  --test_epoch 1	\
--pretrained_weights $PRETRAINED_WEIGHTS	\
--momentum_teacher 0.9  --projector_features 1376
