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

import tensorflow as tf
from typing import Any, Dict, Optional, Text, Tuple

from model.layers import get_activation
from model.blocks import conv2d_block 
import numpy as np

__all__ = ['mb_conv_block']

def mb_conv_block(inputs: tf.Tensor,
                  block: dict,
                  config: dict,
                  prefix: Text = None):
  """Mobile Inverted Residual Bottleneck.

  Args:
    inputs: the Keras input to the block
    block: BlockConfig, arguments to create a Block
    config: ModelConfig, a set of model parameters
    prefix: prefix for naming all layers

  Returns:
    the output of the block
  """
  use_se = config['use_se']
  activation = get_activation(config['activation'])
  drop_connect_rate = config['drop_connect_rate']
  data_format = tf.keras.backend.image_data_format()
  use_depthwise = block['conv_type'] != 'no_depthwise'
  prefix = prefix or ''
  io = block['io']

  filters = block['input_filters'] * block['expand_ratio']

  block_exp = block['expand_ratio']

  #Get the dimensions for all convolutions
  # residual = True if all(s == 1 for s in block['strides']) and block['input_filters'] == block['output_filters'] else False
  downsample_ibn = True if block['input_filters'] == 128 and block['input_filters'] == block['output_filters'] else False
  if downsample_ibn: 
    inC = block['input_filters']
    outC = int(block['output_filters'] / 2)
    block_exp = 6  # NB hard wiring here for the single 128->128 case which actually does downsample
  else:
    inC = block['input_filters']
    outC = block['output_filters']
  if io == 1:
    intermediate_dim = inC*block_exp
  else:
    intermediate_dim = int(np.maximum(inC, outC))*block_exp

  x = inputs

  if block['fused_conv']:
    # If we use fused mbconv, skip expansion and use regular conv.
    x = conv2d_block(x,
                     intermediate_dim,
                     config,
                     kernel_size=block['kernel_size'],
                     strides=block['strides'],
                     activation=activation,
                     name=prefix + 'fused')
  else:
    if block['expand_ratio'] != 1:
      # Expansion phase
      kernel_size = (1, 1) if use_depthwise else (3, 3)
      x = conv2d_block(x,
                       intermediate_dim,
                       config,
                       kernel_size=kernel_size,
                       activation=activation,
                       name=prefix + f'expand__{intermediate_dim}')

    # Depthwise Convolution
    if use_depthwise:
      x = conv2d_block(x,
                       conv_filters=None,
                       config=config,
                       kernel_size=block['kernel_size'],
                       strides=block['strides'],
                       activation=activation,
                       depthwise=True,
                       name=prefix + 'depthwise')

  # Squeeze and Excitation phase
  if use_se:
    assert block['se_ratio'] is not None
    assert 0 < block['se_ratio'] <= 1
    num_reduced_filters = max(1, int(
        block['input_filters'] * block['se_ratio']
    ))

    if data_format == 'channels_first':
      se_shape = (filters, 1, 1)
    else:
      se_shape = (1, 1, filters)

    se = tf.keras.layers.GlobalAveragePooling2D(name=prefix + 'se_squeeze')(x)
    se = tf.keras.layers.Reshape(se_shape, name=prefix + 'se_reshape')(se)

    se = conv2d_block(se,
                      num_reduced_filters,
                      config,
                      use_bias=True,
                      use_batch_norm=False,
                      activation=activation,
                      name=prefix + 'se_reduce')
    se = conv2d_block(se,
                      filters,
                      config,
                      use_bias=True,
                      use_batch_norm=False,
                      activation='sigmoid',
                      name=prefix + 'se_expand')
    x = tf.keras.layers.multiply([x, se], name=prefix + 'se_excite')

  # Output phase
  x = conv2d_block(x,
                   outC,
                   config,
                   activation=None,
                   name=prefix + f'project__{outC}')

  # SECOND DOWNSAMPLED IBN
  if downsample_ibn:
    inC = int(block['output_filters'] / 2)
    outC = block['output_filters']
    intermediate_dim = int(np.maximum(inC, outC))*block_exp #There is no residual in the initial few layers. Later on we can use max expansion.

    x = conv2d_block(x,
                    intermediate_dim,
                    config,
                    kernel_size=kernel_size,
                    activation=activation,
                    name=prefix + f'expand2__{intermediate_dim}')

    # Depthwise Convolution
    if use_depthwise:
      x = conv2d_block(x,
                       conv_filters=None,
                       config=config,
                       kernel_size=block['kernel_size'],
                       strides=block['strides'],
                       activation=activation,
                       depthwise=True,
                       name=prefix + 'depthwise2')
    # Output phase
    x = conv2d_block(x,
                   outC,
                   config,
                   activation=None,
                   name=prefix + f'project2__{outC}')

  # Add identity so that quantization-aware training can insert quantization
  # ops correctly.
  x = tf.keras.layers.Activation(get_activation('identity'),
                                 name=prefix + 'id')(x)

  if (block['id_skip']
      and all(s == 1 for s in block['strides'])
      and block['input_filters'] == block['output_filters']):
    if drop_connect_rate and drop_connect_rate > 0:
      # Apply dropconnect
      # The only difference between dropout and dropconnect in TF is scaling by
      # drop_connect_rate during training. See:
      # https://github.com/keras-team/keras/pull/9898#issuecomment-380577612
      x = tf.keras.layers.Dropout(drop_connect_rate,
                                  noise_shape=(None, 1, 1, 1),
                                  name=prefix + 'drop')(x)

    x = tf.keras.layers.add([x, inputs], name=prefix + 'add')

  return x