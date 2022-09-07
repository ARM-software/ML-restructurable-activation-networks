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
from model.layers import get_activation, get_batch_norm

__all__ = ['collapsed_block']

def collapsed_block(inputs: tf.Tensor,
                    block: dict,
                    config: dict,
                    prefix: Text = None):
  return CollapsedBlock(block, config, name=prefix + 'COLLAPSED_BLOCK_WRAPPER')(inputs)


class CollapsedBlock(tf.keras.layers.Layer):
  def __init__(self, block: dict, config: dict, *args, **kwargs):
    super().__init__(*args, **kwargs)
    data_format = tf.keras.backend.image_data_format()

    batch_norm = get_batch_norm(config['batch_norm'])
    bn_momentum = config['bn_momentum']
    bn_epsilon = config['bn_epsilon']
    self.drop_connect_rate = config['drop_connect_rate']
    data_format = tf.keras.backend.image_data_format()
    self.bn_axis = 1 if data_format == 'channels_first' else -1

    self.activation = get_activation(config['activation'])  # hswish or relu
    # self.activation = tf.nn.relu6

    self.residual = True if all(s == 1 for s in block['strides']) and block['input_filters'] == block['output_filters'] else False
    self.block_cfg = block

    # Defines if an additional lower dimensional IBN should follow
    self.downsample_ibn = True if block['input_filters'] == 128 and self.residual else False  # single use case here

    if self.downsample_ibn:
      outC = int(block['output_filters'] / 2)  # Downsample to half the output filters for the extra downsample IBN
    else:
      outC = block['output_filters']

    # OPERATORS
    self.preceding_bn = batch_norm(axis=self.bn_axis,
                                  momentum=bn_momentum,
                                  epsilon=bn_epsilon,
                                  name='bn_preceding')

    # COLLAPSED BLOCK
    self.collapsed_conv1, self.collapsed_bn1, _ = local_conv2d(
        outC,
        config,
        kernel_size=(3,3),
        strides=block['strides'],
        activation=None,
        name=f"collapsed_conv1__{outC}",
    )

    if self.downsample_ibn:
      outC = block['output_filters']

      self.collapsed_conv2, self.collapsed_bn2, _ = local_conv2d(
            outC,
            config,
            kernel_size=(3,3),
            strides=block['strides'],
            activation=None,
            name=f"collapsed_conv2__{outC}",
          )

  def call(self, inputs):
    x = inputs

    x = self.preceding_bn(x)
    x = self.activation(x)
    x = self.collapsed_conv1(x)
    x = self.collapsed_bn1(x)

    if self.downsample_ibn:
        x = self.activation(x)
        x = self.collapsed_conv2(x)
        x = self.collapsed_bn2(x)

    x = tf.keras.layers.add([x, inputs], name=self.name + 'add') if self.residual else x

    return x


LOCAL_CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_in',
        # Note: this is a truncated normal distribution
        'distribution': 'normal'
    }
}

def local_conv2d(conv_filters: Optional[int],
                 config: dict,
                 kernel_size: Any = (1, 1),
                 strides: Any = (1, 1),
                 use_batch_norm: bool = True,
                 use_bias: bool = False,
                 activation: Any = None,
                 depthwise: bool = False,
                 padding: Text = 'same',
                 name: Text = None):
  """Identical to blocks.conv2d_block except initialisation without input is possible.

      Return format: conv, bn, act
  """
  batch_norm = get_batch_norm(config['batch_norm'])
  bn_momentum = config['bn_momentum']
  bn_epsilon = config['bn_epsilon']
  data_format = tf.keras.backend.image_data_format()
  weight_decay = config['weight_decay']

  name = name or ''

  # Collect args based on what kind of conv2d block is desired
  init_kwargs = {
      'kernel_size': kernel_size,
      'strides': strides,
      'use_bias': use_bias,
      'padding': padding,
      'name': name + '_conv2d',
      'kernel_regularizer': tf.keras.regularizers.l2(weight_decay),
      'bias_regularizer': tf.keras.regularizers.l2(weight_decay),
  }
  LOCAL_CONV_KERNEL_INITIALIZER['config']['mode'] = config['weight_init']

  if depthwise:
    conv2d = tf.keras.layers.DepthwiseConv2D
    init_kwargs.update({'depthwise_initializer': LOCAL_CONV_KERNEL_INITIALIZER})
  else:
    conv2d = tf.keras.layers.Conv2D
    init_kwargs.update({'filters': conv_filters,
                        'kernel_initializer': LOCAL_CONV_KERNEL_INITIALIZER})

  conv = conv2d(**init_kwargs)
  bn = None
  act = None

  if use_batch_norm:
    bn_axis = 1 if data_format == 'channels_first' else -1
    bn = batch_norm(axis=bn_axis,
                   momentum=bn_momentum,
                   epsilon=bn_epsilon,
                   name=name + '_bn')

  if activation is not None:
    act = tf.keras.layers.Activation(activation,
                                   name=name + '_activation')

  return conv, bn, act