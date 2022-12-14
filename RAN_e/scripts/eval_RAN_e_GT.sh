# SPDX-License-Identifier: Apache-2.0
# Copyright 2022, Arm Limited and/or its affiliates.
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

# horovodrun -np 8 bash ./scripts/bind.sh --cpu=exclusive --ib=single -- python3 main.py \
python3 main.py \
  --mode "eval" \
  --arch "ran_e_GT-b0" \
  --model_dir $MODEL_DIR \
  --data_dir $DATA_DIR \
  --use_xla \
  --augmenter_name autoaugment \
  --max_epochs 1 \
  --eval_batch_size 96 \
  --log_steps 100 \
  --save_checkpoint_freq 5 \
  --lr_init 0.005