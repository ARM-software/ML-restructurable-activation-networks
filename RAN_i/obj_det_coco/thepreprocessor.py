# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0

from typing import List, Dict, Tuple, Optional

import torch
from torchvision.models.detection.transform import GeneralizedRCNNTransform

class ListOfResizedImages:
    def __init__(self, tensors: torch.Tensor, original_image_sizes: List[Tuple[int, int]]):
        """Convenience class designed to store a batch of images that have been resized, along with their original image sizes.

        Args:
            tensors (torch.Tensor): Resized image tensors.
            original_image_sizes (List[Tuple[int, int]]): a list of the original image size of each image
        """
        self.tensors = tensors
        self.image_sizes = original_image_sizes

    def to(self, device: torch.device):
        """Send contained image tensors to requested PyTorch device.

        Args:
            device (torch.device): PyTorch device.

        Returns:
            ListOfResizedImages: The new batch, with the image tensors on the PyTorch device.
        """
        cast_tensor = self.tensors.to(device)
        return ListOfResizedImages(cast_tensor, self.image_sizes)

class TheDetectorMandatoryPreprocessor(GeneralizedRCNNTransform):
    def __init__(
        self,
        min_size: int=800,
        max_size: int=1200,
        normalize_mean: List[float]=[0.485, 0.456, 0.406],
        normalize_stdev: List[float]=[0.229, 0.224, 0.225],
    ):
        """Modified version of the PyTorch GeneralizedRCNNTransform, handling preprocessing of images.

        Args:
            min_size (int, optional): Minimum size to which images are resized. Defaults to 800.
            max_size (int, optional): Maximum size to which images are resized. Defaults to 1200.
            normalize_mean (List[float], optional): Mean values to which images are normalized. Defaults to ImageNet [0.485, 0.456, 0.406].
            normalize_stdev (List[float], optional): Stdev values to which images are normalized. Defaults to ImageNet [0.229, 0.224, 0.225].
        """
        super().__init__(min_size, max_size, normalize_mean, normalize_stdev)

    def forward(self, images: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]]=None) -> Tuple[ListOfResizedImages, Optional[List[Dict[str, torch.Tensor]]]]:
        """Performs preprocessing on a single batch (list of images, list of targets)

        Args:
            images (List[torch.Tensor]): Input images.
            targets (List[Dict[str, torch.Tensor]], optional): Input targets. Defaults to None.

        Returns:
            Tuple[ListOfResizedImages, Optional[List[Dict[str, torch.Tensor]]]]: Transformed batch.
        """
        if targets:
            targets = [{k: v for k,v in t.items()} for t in targets]
            
        for i in range(len(images)):
            image = images[i]
            curr_target = targets[i] if targets else None

            if image.dim() != 3:
                raise ValueError(f"Images is expected to be a list of 3d tensors of shape [C, H, W], got {image.shape}")
            
            # Perform image normalization using GeneralizedRCNNTransform.normalize()
            normalized_image = self.normalize(image)
            
            # Resize image using GeneralizedRCNNTransform.resize()
            normalized_resized_image, resized_target = self.resize(normalized_image, curr_target)
            images[i] = normalized_resized_image
            if targets and resized_target:
                targets[i] = resized_target

        # Update image_sizes
        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images, size_divisible=self.size_divisible)
        image_sizes_list: List[Tuple[int, int]] = []
        for image_size in image_sizes:
            torch._assert(
                len(image_size) == 2,
                f"Input tensors expected to have in the last two elements H and W, instead got {image_size}",
            )
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ListOfResizedImages(images, image_sizes_list)
        return image_list, targets
