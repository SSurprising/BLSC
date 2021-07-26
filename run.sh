
#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name='BLSC'\
    --data_path='/home/zjm/Data/HaN_OAR_256/' \
    --loss_image_path='/home/zjm/Project/segmentation/BLSC/loss_show/' \
    --save_path='/home/zjm/Project/segmentation/BLSC/checkpoints/' \
    --val_subset='subset5' \
    --repeat_times='ResumeFromFirstLrStep' \
    --epochs=1000 \
    --learning_rate=1e-4 \
    --num_organ=22 \
    --load_all_image \
    --resume_training \
    --resume_model='/home/zjm/Project/segmentation/BLSC/checkpoints/BLSC/best_BLSC1.pth'\
    --MultiStepLR 20 320 670
