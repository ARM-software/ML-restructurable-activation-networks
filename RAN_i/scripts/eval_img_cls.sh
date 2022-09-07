# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0

#model ran_i_tiny
python -m torch.distributed.launch --nproc_per_node=8 train_eval_img_cls_imagenet.py \
--model ran_i_tiny \
--batch_size 80  \
--model_ema true --model_ema_eval true --drop_path 0.1 \
--data_path ~/workspace/pytorch/datasets/imagenet_data/ \
--eval true \
--resume ../public_checkpoints/img_cls_imagenet/RAN_i_Tiny_82.03.pth \
--log_dir ../logs_outputs/ran_i_tiny \
--output_dir ../logs_outputs/ran_i_tiny 2>&1 | tee -i ran_i_tiny_eval.log


#model ran_i_small
python -m torch.distributed.launch --nproc_per_node=8 train_eval_img_cls_imagenet.py \
--model ran_i_small \
--batch_size 80  \
--model_ema true --model_ema_eval true --drop_path 0.2 \
--data_path ~/workspace/pytorch/datasets/imagenet_data/ \
--eval true \
--resume ../public_checkpoints/img_cls_imagenet/RAN_i_Small_82.63.pth \
--log_dir ../logs_outputs/ran_i_small \
--output_dir ../logs_outputs/ran_i_small 2>&1 | tee -i ran_i_small_eval.log


#model ran_i_base
python -m torch.distributed.launch --nproc_per_node=8 train_eval_img_cls_imagenet.py \
--model ran_i_base \
--batch_size 80  \
--model_ema true --model_ema_eval true --drop_path 0.4 \
--data_path ~/workspace/pytorch/datasets/imagenet_data/ \
--eval true \
--resume ../public_checkpoints/img_cls_imagenet/RAN_i_Base_83.61.pth \
--log_dir ../logs_outputs/ran_i_base \
--output_dir ../logs_outputs/ran_i_base 2>&1 | tee -i ran_i_base_eval.log


