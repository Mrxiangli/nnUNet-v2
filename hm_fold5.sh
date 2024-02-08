#!/bin/bash
module load cuda/11.7.0
module load cudnn/cuda-11.7_8.6
module load use.own
module load conda-env/nnunetv1-py3.8.5
source ~/.bashrc
cd /scratch/gilbreth/li2068/nnUNet_v2/nnunetv2/run/
#cd /scratch/gilbreth/li2068/nnUNet_v2/nnunetv2/inference/
#python /scratch/gilbreth/li2068/roto_unet/nnunetv2/inference/predict_from_raw_data.py
python /scratch/gilbreth/li2068/nnUNet_v2/nnunetv2/run//run_training.py 301 3d_fullres 3 -num_gpus=1