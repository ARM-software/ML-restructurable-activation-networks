# SPDX-License-Identifier: BSD-3-Clause

#model ran_i_small backbone
python -m torch.distributed.launch --nproc_per_node=8 --use_env train_eval_obj_det_coco.py --dataset coco --backbone-model-name ran_i_s --backbone-model-weights ../public_checkpoints/img_cls_imagenet/RAN_i_Small_82.63.pth --epochs 26 --lr-steps 16 22 --world-size=8 2>&1 | tee od_train_ran_i_s_backbone.log
