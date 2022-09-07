# SPDX-License-Identifier: Apache-2.0
# Copyright 2022, Arm Limited and/or its affiliates
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

MODEL_DIR="../logs_350epochs/RAN_e"
DATA_DIR="/home/ubuntu/2159_imagenet/"
INDX="./index_file"

export TF_XLA_FLAGS="--tf_xla_cpu_global_jit"
echo $TF_XLA_FLAGS

# horovodrun -np 8 bash ./scripts/bind.sh --cpu=exclusive --ib=single -- python3 main.py \
horovodrun -np 8 python3 main.py \
  --mode "train_and_eval" \
  --arch "ran_e_C-b0" \
  --model_dir $MODEL_DIR \
  --data_dir $DATA_DIR \
  --use_xla \
  --activation hard_swish \
  --enable_tensorboard \
  --write_model_weights \
  --augmenter_name autoaugment \
  --weight_init fan_out \
  --lr_decay cosine \
  --max_epochs 350 \
  --train_batch_size 96 \
  --eval_batch_size 96 \
  --log_steps 100 \
  --save_checkpoint_freq 50 \
  --lr_init 0.005 \
  --batch_norm syncbn \
  --mixup_alpha 0.0 \
  --weight_decay 5e-6 \
  --epsilon 0.001 \
  --resume_checkpoint 2>&1 | tee -i RAN_e.log
