"""PyTorch dataset class for IMAC image dataset."""


import os
import torch
from torch.utils.data import Dataset
from torch import Tensor
import torchvision
from torchvision.transforms import Compose, Resize, RandomCrop, ToTensor, Normalize
import pandas as pd
from PIL import Image
from typing import Dict, Tuple, Any
from climur.utils.constants import CLIP_IMAGE_SIZE


class IMACImages(Dataset):
    """Dataset class for IMAC image dataset.

    Attributes:
        metadata (DataFrame): Metadata.
        root (str): Path of top-level root directory of dataset.
        transform (torchvision Compose object): Image augmentations transform.
        n_views (int): Number of augmented views.
        eval (bool): Selects whether to get images in evaluation mode.
        preprocess (torchvision Compose object): Image preprocessing transform for evaluation mode.
        emotion_tags (list): Emotion tags vocabulary.
    """

    def __init__(self, root: str, metadata_file_name: str, augment_params: Dict = None, eval: bool = False, preprocess: Any = None) -> None:
        """Initialization.

        Args:
            root (str): Path of root directory of dataset.
            metadata_file_name (str): Name of metadata file.
            augment_params (dict): Audio augmentation paramaters.
            eval (bool): Selects whether to get images in evaluation mode.
            preprocess (torchvision Compose object): Image preprocessing transform for evaluation mode.
        
        Returns: None
        """

        # validate parameters:
        assert augment_params is not None or preprocess is not None, "Either augment_params or preprocess must be not None."
        if eval:
            assert augment_params is None, "Not allowed to apply augmentations when in evaluation mode."
            assert preprocess is not None, "Need preprocessing when in evaluation mode."

        # save parameters:
        self.root = root
        self.eval = eval
        self.preprocess = preprocess

        # load metadata:
        self.metadata = pd.read_csv(os.path.join(self.root, metadata_file_name))
        # get emotion tags:
        self.emotion_tags = self.metadata["label"].unique().tolist()

        # set up image augmentations:
        if augment_params is not None:
            self.n_views = augment_params["n_views"]
            self.transform = Compose([
                Resize(size=CLIP_IMAGE_SIZE, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                RandomCrop(size=CLIP_IMAGE_SIZE),
                _convert_image_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            self.transform = None
    
    def __len__(self) -> int:
        """Gets length of dataset.

        Args: None

        Returns:
            dataset_len (int): Length of dataset.
        """

        dataset_len = self.metadata.shape[0]

        return dataset_len
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, str]:
        """Gets an image and its emotion tag.

        Args:
            idx (int): Item index.
        
        Returns:
            if not eval:
                image (Tensor): Multiple augmented views of image.
                    shape: (n_views, image_channels, image_height, image_width)
            else:
                image (Tensor): Preprocessed image.
                    shape: (image_channels, image_height, image_width)
            tag (str): Emotion tag.
        """

        # get image file_path:
        subdir_name = self.metadata.loc[idx, "subdir_name"]
        file_name = self.metadata.loc[idx, "file_name"]
        file_path = os.path.join(self.root, subdir_name, file_name)

        # load image in training mode:
        if not self.eval:
            # apply augmentations if selected:
            if self.transform is not None:
                image_augments_list = []
                for _ in range(self.n_views):
                    with Image.open(file_path) as image_file:
                        image_augment = self.transform(image_file)
                    # image_augment = self.transform(image)
                    image_augments_list.append(image_augment)
                image = torch.stack(image_augments_list)
            # only preprocess image:
            else:
                if self.preprocess is not None:
                    with Image.open(file_path) as image_file:
                        image = self.preprocess(image_file)
                    # insert n_views dimension for compatibility:
                    image = image.unsqueeze(dim=0)
                else:
                    raise RuntimeError("Both self.transform and self.preprocess are None.")
        
        # load image in evaluation mode:
        else:
            if self.preprocess is not None:
                with Image.open(file_path) as image_file:
                    image = self.preprocess(image_file)
            else:
                raise RuntimeError("In evaluation mode and could not preprocess image.")
        
        # get emotion tag:
        tag = self.metadata.loc[idx, "label"]
        
        return image, tag


def _convert_image_to_rgb(image: Any) -> Any:     # TODO: Deal with this.

    return image.convert("RGB")

