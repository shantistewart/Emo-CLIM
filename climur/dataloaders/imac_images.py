"""PyTorch dataset class for IMAC image dataset."""


import os
from torch.utils.data import Dataset
from torch import Tensor
import torchvision
import pandas as pd
from PIL import Image
from typing import Tuple, Any


class IMACImages(Dataset):
    """Dataset class for IMAC image dataset.

    Attributes:
        metadata (DataFrame): Metadata.
        root (str): Path of top-level root directory of dataset.
        preprocess (torchvision transform): torchvision preprocessing transform.
        emotion_tags (list): Emotion tags vocabulary.
    """

    def __init__(self, root: str, metadata_file_name: str, preprocess: Any = None) -> None:
        """Initialization.

        Args:
            root (str): Path of root directory of dataset.
            metadata_file_name (str): Name of metadata file.
            preprocess (torchvision transform): torchvision preprocessing transform.
        
        Returns: None
        """

        # save parameters:
        self.root = root
        self.preprocess = preprocess

        # load metadata:
        self.metadata = pd.read_csv(os.path.join(self.root, metadata_file_name))
        # get emotion tags:
        self.emotion_tags = self.metadata["label"].unique().tolist()
    
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
            image (Tensor): Image (preprocessed if preprocess is not None).
                shape: (image_channels, image_height, image_width)
            tag (str): Emotion tag.
        """

        # get image file_path:
        subdir_name = self.metadata.loc[idx, "subdir_name"]
        file_name = self.metadata.loc[idx, "file_name"]
        file_path = os.path.join(self.root, subdir_name, file_name)

        # load image:
        if self.preprocess is not None:
            with Image.open(file_path) as image_file:
                image = self.preprocess(image_file)
        else:
            image = torchvision.io.read_image(file_path)
        
        # get emotion tag:
        tag = self.metadata.loc[idx, "label"]

        return image, tag

