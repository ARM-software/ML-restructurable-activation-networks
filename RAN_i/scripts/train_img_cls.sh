# SPDX-License-Identifier: BSD-3-Clause

echo "WARNING: RUNS TRAINING ON ALL RAN_i MODELS (TINY, SMALL, BASE) SEQUENTIALLY. COMMENT OUT OTHERS IF YOU ONLY WANT TO RUN ONE OF THEM"


#model ran_i_tiny
python -m torch.distributed.launch --nproc_per_node=8 train_eval_img_cls_imagenet.py \
--model ran_i_tiny --drop_path 0.1 \
--batch_size 80 --lr 4e-3 --update_freq 4 \
--model_ema true --model_ema_eval true \
--data_path ~/workspace/pytorch/datasets/imagenet_data/ \
--epochs 300 \
--log_dir ../logs_outputs/ran_i_tiny \
--output_dir ../logs_outputs/ran_i_tiny 2>&1 | tee -i ran_i_tiny_300e.log

#model ran_i_small
python -m torch.distributed.launch --nproc_per_node=8 train_eval_img_cls_imagenet.py \
--model ran_i_small --drop_path 0.2 \
--batch_size 80 --lr 4e-3 --update_freq 4 \
--model_ema true --model_ema_eval true \
--data_path ~/workspace/pytorch/datasets/imagenet_data/ \
--epochs 300 \
--log_dir ../logs_outputs/ran_i_small \
--output_dir ../logs_outputs/ran_i_small 2>&1 | tee -i ran_i_small_300e.log

#model ran_i_base
python -m torch.distributed.launch --nproc_per_node=8 train_eval_img_cls_imagenet.py \
--model ran_i_base --drop_path 0.4 \
--batch_size 80 --lr 4e-3 --update_freq 4 \
--model_ema true --model_ema_eval true \
--data_path ~/workspace/pytorch/datasets/imagenet_data/ \
--epochs 300 \
--log_dir ../logs_outputs/ran_i_base \
--output_dir ../logs_outputs/ran_i_base 2>&1 | tee -i ran_i_base_300e.log
