#!/bin/bash

#SBATCH --workdir=/slurm_storage/jzou/exp/CerCyt/nucleus_mrcnn/
#SBATCH --output=/slurm_storage/jzou/exp/CerCyt/nucleus_mrcnn/slurm_%j.out
#SBATCH --error=/slurm_storage/jzou/exp/CerCyt/nucleus_mrcnn/slurm_%j.error
#SBATCH --job-name=nucleus_mrcnn
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx1

export LD_LIBRARY_PATH=/slurm_storage/public/cuda9.0/lib64

# env
which python
python /slurm_storage/jzou/programs/CerCyt/cercyt/nucleus_mrcnn.py\
    --weights /slurm_storage/jzou/programs/CerCyt/models/mask_rcnn_coco.h5\
    --dataset /slurm_storge/jzou/datasets/data-science-bowl-2018\
    --subset stage1_train
