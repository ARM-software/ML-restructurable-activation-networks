# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0

import torch
import torch.nn.functional as F
import torch.nn as nn

from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork
from torchvision.models.detection.rpn import RPNHead as RPNPooling
from torchvision.models.detection.roi_heads import RoIHeads as ObjDetHead
from obj_det_coco.thepreprocessor import TheDetectorMandatoryPreprocessor


class TheDetector(GeneralizedRCNN):
    def __init__(
        self,
        preprocessor: TheDetectorMandatoryPreprocessor = TheDetectorMandatoryPreprocessor(),
        backbone: nn.Module = None,
        num_classes: int = None,
        rpn_anchor_sizes: tuple = ((32, 64, 128, 256, 512),),
        rpn_anchor_aspect_ratios: tuple = ((0.5, 1.0, 2.0),),
        rpn_nms_thresh: float = 0.7,
        rpn_batch_size_per_image: int = 256,
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.5,
        box_detections_per_img: int = 100,
        box_batch_size_per_image: int = 256,
        head_weights=None,
    ):
        """Builds a slim 2-stage object detector for use with RAN_i/ConvNeXt et al backbones.

        INFERENCE:
            Remember to call `model.eval()`
            Input should be of type `List[FloatTensor[C, H, W]]`, one tensor per image.
            Images should be in `[0, 1]` float range.
            Images can have different sizes, as the preprocessor will resize everything.
            The model returns a List[Dict], with one `dict` per predicted image.
            Each dict contains:
                "boxes" (`FloatTensor[N, 4]`): predicted bounding boxes (NMS thresholded) in xmin, ymin, xmax, ymax format.
                "labels" (`Int64Tensor[N]`): predicted class label for each box.
                "scores" (`Tensor[N]`): confidence scores for each box.

        TRAINING:
            Remember to call `model.train()`
            Input should be passed in as a tuple of lists, where each list is of size N,
                where the first element is input images of type `List[FloatTensor[C, H, W]]`, one tensor per image.
                where the second element is targets of type `List[Dict]` with one `Dict` per image.
                Each dict contains:
                "boxes" (`FloatTensor[N, 4]`): ground truth bounding boxes in xmin, ymin, xmax, ymax format.
                "labels" (`Int64Tensor[N]`): ground truth class label for each box.
            The model returns a Dict[Tensor] containing losses for the RPN and head.

        Args:
            preprocessor (TheDetectorMandatoryPreprocessor, optional): Preprocessor to ensure images are the same size. Defaults to TheDetectorMandatoryPreprocessor().
            backbone (nn.Module, optional): Backbone model. Should have an out_channels attribute. Defaults to None.
            num_classes (int, optional): Number of dataset classes (MS COCO is 91). Defaults to None.
            rpn_anchor_sizes (tuple, optional): Anchorbox sizes, a la Faster-RCNN paper. Defaults to ((32, 64, 128, 256, 512),).
            rpn_anchor_aspect_ratios (tuple, optional): Anchorbox aspect ratios, a la Faster-RCNN paper. Defaults to ((0.5, 1.0, 2.0),).
            rpn_nms_thresh (float, optional): NMS threshold for the RPN. Defaults to 0.7.
            rpn_batch_size_per_image (int, optional): For training only. Number of proposed RoIs used to calculate the loss. Defaults to 256.
            box_score_thresh (float, optional): For inference only. Minimum classification score needed to return. Defaults to 0.05.
            box_nms_thresh (float, optional): For inference only. NMS threshold for the head. Defaults to 0.5.
            box_detections_per_img (int, optional): Maximum number of predictions allowed per image. Defaults to 100.
            box_batch_size_per_image (int, optional): For training only. Number of bounding boxes used to calculate the loss. Defaults to 256.
            head_weights (_type_, optional): Pre-trained head weights. Defaults to None.
        """
        if not backbone:
            raise ValueError("Need backbone!")
        if not num_classes:
            raise ValueError("Needs num_classes!")
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels specifying the number of output channels from the backbone feature map"
            )

        # Should automatically be provided by the get_backbone() functionality of RAN_i and ConvNeXt
        out_channels = backbone.out_channels
        rpn_anchor_generator = AnchorGenerator(sizes=rpn_anchor_sizes, aspect_ratios=rpn_anchor_aspect_ratios)
        rpn_pooling = RPNPooling(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        # Use Faster-RCNN values: minimum IoU between the anchor and the GT box so that they can beconsidered as positive during training of the RPN.
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3

        # proportion of positive anchors in a mini-batch during trainingof the RPN
        rpn_positive_fraction = 0.5

        # Number of proposals to keep before applying NMS. Using Faster-RCNN values.
        rpn_pre_nms_top_n = {"training": 2000, "testing": 1000}

        # Number of proposals to keep after applying NMS. Using Faster-RCNN values.
        rpn_post_nms_top_n = {"training": 2000, "testing": 1000}

        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_pooling,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=0.0,
        )

        # Use setup from PyTorch's ResNet50-FPN-Faster-RCNN
        roi_alignment = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=7,
            sampling_ratio=2
        )
        resolution = roi_alignment.output_size[0]
        intermediate_representation_size = 512  # TODO add to params
        head_part_one = HeadPartOne(out_channels * resolution**2, intermediate_representation_size)
        head_part_two = HeadPartTwo(intermediate_representation_size, num_classes)

        #  minimum IoU between the proposals and the GT box so that they can be considered as positive during training of the classification head
        box_fg_iou_thresh = 0.5
        box_bg_iou_thresh = 0.5

        # proportion of positive proposals in a mini-batch during training of the classification head
        box_positive_fraction = 0.25

        objdet_head = ObjDetHead(
            roi_alignment,
            head_part_one,
            head_part_two,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            head_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )

        super().__init__(backbone, rpn, objdet_head, preprocessor)


class HeadPartOne(nn.Module):
    def __init__(self, in_channels: int, representation_size: int):
        """First part of the head, to flatten the features and send them to the predictor.

        Args:
            in_channels (int): Number of input channels.
            representation_size (int): Desired intermediate feature representation size. Change to whatever you'd like to reduce size.
        """
        super().__init__()

        self.fc1 = nn.Linear(in_channels, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


class HeadPartTwo(nn.Module):
    """
    Second part of the head, classification and regression, prediction layers.

    Args:
        in_channels (int): Number of input channels
        num_classes (int): Number of dataset classes (for COCO, this is 91)
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.class_scores = nn.Linear(in_channels, num_classes)
        self.bbox_predictor = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        x = x.flatten(1)
        scores = self.class_scores(x)
        bboxes = self.bbox_predictor(x)

        return scores, bboxes

if __name__ == "__main__":
    from models.ran_i import ran_i_small
    newNet = TheDetector(backbone=ran_i_small().get_backbone(), num_classes=91)
