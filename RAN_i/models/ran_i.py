# SPDX-License-Identifier: MIT AND Apache-2.0
# Copyright 2022, Arm Limited and/or its affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from models.convnext import ConvNeXt, Block, LayerNorm
from timm.models.registry import register_model


class RAN_i(ConvNeXt):
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: list = [3, 3, 9, 3],
        dims: list = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        head_init_scale: float = 1.0,
        wm: float = 1.0,
        dm: float = 1.0
    ):
        """A RAN-scaled version of ConvNeXt, modified with depth and width multipliers. Includes NN-Mass calculation.

        Args:
            in_chans (int, optional): Number of input image channels. Defaults to 3.
            num_classes (int, optional): Number of classes for classification head. Defaults to 1000.
            depths (list, optional): Number of blocks at each stage, multiplied later by dm. Defaults to [5, 5, 14, 5].
            dims (list, optional): Feature dimension at each stage, multiplied later by wm. Defaults to [62, 124, 248, 496].
            drop_path_rate (float, optional): Stochastic depth rate. Defaults to 1.0.
            layer_scale_init_value (float, optional): Init value for Layer Scale. Defaults to 1e-6.
            head_init_scale (float, optional): Init scaling value for classifier weights and biases. Defaults to 1.0.
            wm (float, optional): Width scaling multiplier for RAN_i. Defaults to 1.0.
            dm (float, optional): Depth scaling multiplier for RAN_i. Defaults to 1.0.
        """
        super().__init__(in_chans, num_classes, depths, dims, drop_path_rate, layer_scale_init_value, head_init_scale)

        # Recalculate Depths and Dims using d, w multipliers
        depths = [round(d * dm) for d in depths]
        dims = [round(w * wm) for w in dims]
        self.nn_mass = 0.0

        # Standard ConvNeXt initialization from here on out
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            # Add NN-Mass calculation
            for _ in range(depths[i]):
                self.nn_mass += 2.0 * dims[i]
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)


@register_model
def ran_i_tiny(pretrained_weights_path: str = None, pretrained=False) -> RAN_i:
    """Returns an RAN_i-T instance. If a file is provided, loads the weights.

    NOTE: Weights must be saved under the "model" key in the weights file.

    Args:
        pretrained_weights_path (str, optional): Path to the weights file for RAN_i-T. Defaults to None.

    Returns:
        RAN_i: The RAN_i-T model.
    """
    model = RAN_i(wm=0.666, dm=1.65)
    if pretrained_weights_path:
        checkpoint = torch.load(pretrained_weights_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def ran_i_small(pretrained_weights_path: str = None, pretrained=False) -> RAN_i:
    """Returns an RAN_i-S instance. If a file is provided, loads the weights.

    NOTE: Weights must be saved under the "model" key in the weights file.

    Args:
        pretrained_weights_path (str, optional): Path to the weights file for RAN_i-S. Defaults to None.

    Returns:
        RAN_i: The RAN_i-S model.
    """
    model = RAN_i(wm=0.789, dm=1.65)
    if pretrained_weights_path:
        checkpoint = torch.load(pretrained_weights_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def ran_i_base(pretrained_weights_path: str = None, pretrained=False) -> RAN_i:
    """Returns an RAN_i-B instance. If a file is provided, loads the weights.

    NOTE: Weights must be saved under the "model" key in the weights file.

    Args:
        pretrained_weights_path (str, optional): Path to the weights file for RAN_i-B. Defaults to None.

    Returns:
        RAN_i: The RAN_i-B model.
    """
    model = RAN_i(wm=0.9105, dm=2.30)
    if pretrained_weights_path:
        checkpoint = torch.load(pretrained_weights_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

if __name__ == "__main__":
    ran_i_s_pretrained = ran_i_small("RAN_i_Small_82.63.pth")
    ran_i_s_pretrained(torch.rand([1,3,224,224]))
