# SPDX-License-Identifier: BSD-3-Clause

#model ran_i_small backbone
python -m torch.distributed.launch --nproc_per_node=8 --use_env train_eval_obj_det_coco.py --dataset coco --backbone-model-name ran_i_s  --epochs 26 --lr-steps 16 22 --world-size=8 --test-only --resume ../public_checkpoints/obj_det_coco/RAN_i_S_backbone_detector.pth 2>&1 | tee od_eval_only_ran_i_s_backbone.log
