"""PyTorch dataset class for retrieving images and audio clips."""


import torch
from torch.utils.data import Dataset
from torch import Tensor
import random
from typing import Union, Dict, List
from climur.utils.constants import IMAGE2AUDIO_TAG_MAP


class Multimodal(Dataset):
    """Wrapper dataset class for retrieving images and audio clips.

    Attributes:
        image_dataset (PyTorch Dataset): Image dataset.
        audio_dataset (PyTorch Dataset): Audio dataset.
        length (int): Effective length of dataset (since __getitem__(idx) ignores idx, length can be arbitrarily set).
        image_dataset_len (int): Effective length of image dataset (disregarding images with unused labels).
        audio_dataset_len (int): Effective length of audio dataset (disregarding audio clips with unused labels).
        n_classes (int): Number of (common) emotion tag classes.
        image2audio_tag_map (dict): Dictionary mapping image dataset emotion tags to audio dataset emotion tags.
        image_tags (list): Image dataset emotion tags.
        audio_tags (list): Audio dataset emotion tags.
        label2idx (Dict): Dictionary mapping emotion labels to class label indices:
            Note: audio dataset emotion tags are used as emotion labels.
        idx2label (Dict): Dictionary mapping class label indices to emotion labels:
    """

    def __init__(self, image_dataset: Dataset, audio_dataset: Dataset, length: int = None, image2audio_tag_map: Dict = IMAGE2AUDIO_TAG_MAP, label2idx: Dict = None) -> None:
        """Initialization.

        Args:
            image_dataset (PyTorch Dataset): Image dataset.
            audio_dataset (PyTorch Dataset): Audio dataset.
            length (int): Effective length of dataset.
            image2audio_tag_map (dict): Dictionary mapping image dataset emotion tags to audio dataset emotion tags.
            label2idx (Dict): Dictionary mapping emotion labels to class label indices (if None, default is created):
        
        Returns: None
        """

        # validate params:
        assert set(list(image2audio_tag_map.keys())) <= set(image_dataset.emotion_tags), "image2audio_tag_map contains emotion tags not in image dataset."
        assert set(list(image2audio_tag_map.values())) <= set(audio_dataset.emotion_tags), "image2audio_tag_map contains emotion tags not in audio dataset."
        if label2idx is not None:
            assert set(list(label2idx.keys())) == set(list(image2audio_tag_map.values())), "Error with keys of label2idx."
        
        # save parameters:
        self.image_dataset = image_dataset
        self.audio_dataset = audio_dataset
        self.length = length

        # save things related to emotion tags:
        self.image2audio_tag_map = image2audio_tag_map
        self.n_classes = len(image2audio_tag_map)
        self.image_tags = list(image2audio_tag_map.keys())
        self.audio_tags = list(image2audio_tag_map.values())

        # create label2idx map:
        if label2idx is None:
            self.label2idx = {label: idx for idx, label in enumerate(self.audio_tags)}
        else:
            self.label2idx = label2idx
        # also create inverse mapping (for later use):
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

        # create dictionary mapping emotion tags to image file names:
        self.tag2image = {}
        for tag in self.image_tags:
            self.tag2image[tag] = image_dataset.metadata[image_dataset.metadata["label"] == tag]
        # create dictionary mapping emotion tags to audio file names:
        self.tag2audio = {}
        for tag in self.audio_tags:
            self.tag2audio[tag] = audio_dataset.metadata[audio_dataset.metadata["label"] == tag]

        # compute effective image dataset length:
        self.image_dataset_len = 0
        for df in self.tag2image.values():
            self.image_dataset_len += df.shape[0]
        # compute effective audio dataset length:
        self.audio_dataset_len = 0
        for df in self.tag2audio.values():
            self.audio_dataset_len += df.shape[0]
    
    def __len__(self) -> int:
        """Gets effective length of dataset.

        Args: None

        Returns:
            dataset_len (int): Effective length of dataset.
        """

        if self.length is not None:
            dataset_len = self.length
        # if length not specified, use default value:
        else:
            dataset_len = min(self.image_dataset_len, self.audio_dataset_len)
        
        return dataset_len
    
    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, int]]:
        """Randomly retrieves an (image, audio clip) pair and their emotion labels.

        Args:
            idx (int): Item index (not used due to random selection).
        
        Returns:
            item (dict): Item dictionary with keys/values:
                "image": (Tensor) Multiple augmented views of image.
                    shape: (n_views, image_channels, image_height, image_width)
                "image_label" (int): Image emotion label index.
                "audio": (Tensor) Multiple augmented views of audio clip.
                    shape: (n_views, audio_clip_length)
                "audio_label" (int): Audio emotion label index.
        """

        # randomly select emotion tag for image:
        image_tag = random.choice(self.image_tags)
        # randomly select an image labeled with selected emotion tag:
        image_idx = int(self.tag2image[image_tag].sample(n=1, axis="index").index[0])
        image, tag = self.image_dataset[image_idx]
        assert tag == image_tag, "Image has incorrect emotion label."

        # randomly select emotion tag for audio clip:
        audio_tag = random.choice(self.audio_tags)
        # randomly select an audio clip labeled with selected emotion tag:
        audio_idx = int(self.tag2audio[audio_tag].sample(n=1, axis="index").index[0])
        audio, tag = self.audio_dataset[audio_idx]
        assert tag == audio_tag, "Audio clip has incorrect emotion label."

        # return as an item dictionary (note: image emotion tag is mapped to equivalent audio emotion tag):
        item = {
            "image": image,
            "image_label": self.label2idx[self.image2audio_tag_map[image_tag]],
            "audio": audio,
            "audio_label": self.label2idx[audio_tag]
        }

        return item
    
    def collate_fn(self, batch: List) -> Dict:
        """Custom collate function for Multimodal dataset class.

        Args:
            batch (list): List of batch items.
        
        Returns:
            batch_dict (dict): Batch dictionary with keys/values:
                "image": (Tensor) Augmented images.
                    shape: (batch_size * n_views, image_channels, image_height, image_width)
                "image_label" (Tensor): Image emotion label indices.
                    shape: (batch_size * n_views, )
                "audio": (Tensor) Augmented audio clips.
                    shape: (batch_size * n_views, audio_clip_length)
                "audio_label" (int): Audio emotion label indices.
                    shape: (batch_size * n_views, )
        """

        # sanity check:
        assert set(list(batch[0].keys())) == {"image", "image_label", "audio", "audio_label"}, "Item keys are unexpected."

        # unpack batch:
        batch_size = len(batch)
        image_views_list = [item["image"] for item in batch]     # list element shape: (n_views, image_channels, image_height, image_width)
        image_labels_orig = [item["image_label"] for item in batch]
        audio_views_list = [item["audio"] for item in batch]     # list element shape: shape: (n_views, audio_clip_length)
        audio_labels_orig = [item["audio_label"] for item in batch]

        # unroll multiple views of images:
        images_list = []
        n_image_views = image_views_list[0].size(dim=0)
        for image_views in image_views_list:
            for k in range(n_image_views):
                images_list.append(image_views[k])
        # convert list to tensor:
        images = torch.stack(images_list, dim=0)     # shape: (batch_size * n_views, image_channels, image_height, image_width)
        assert len(tuple(images.size())) == 4 and images.size(dim=0) == batch_size * n_image_views, "Error with shape of unrolled images."

        # unroll multiple views of audio clips:
        audios_list = []
        n_audio_views = audio_views_list[0].size(dim=0)
        for audio_views in audio_views_list:
            for k in range(n_audio_views):
                audios_list.append(audio_views[k])
        # convert list to tensor:
        audios = torch.stack(audios_list, dim=0)     # shape: (batch_size * n_views, audio_clip_length)
        assert len(tuple(audios.size())) == 2 and audios.size(dim=0) == batch_size * n_audio_views, "Error with shape of unrolled audio clips."


        """
        # convert to tensors:
        image_views = torch.stack(image_views_list, dim=0)     # shape: (batch_size, n_views, image_channels, image_height, image_width)
        audio_views = torch.stack(audio_views_list, dim=0)     # shape: (batch_size, n_views, audio_clip_length)
        # remove n_views dimension:
        n_image_views = image_views.size(dim=1)
        n_audio_views = audio_views.size(dim=1)
        images = image_views.view(batch_size * n_image_views, -1)     # (batch_size, n_views, image_channels, image_height, image_width) -> (batch_size * n_views, image_channels, image_height, image_width)
        audios = audio_views.view(batch_size * n_audio_views, -1)     # shape: (batch_size, n_views, audio_clip_length) -> (batch_size * n_views, audio_clip_length)
        """


        # repeat image labels to account for multiple views of images:
        image_labels = []
        for label in image_labels_orig:
            for k in range(n_image_views):
                image_labels.append(label)
        
        # repeat audio labels to account for multiple views of audio clips:
        audio_labels = []
        for label in audio_labels_orig:
            for k in range(n_audio_views):
                audio_labels.append(label)
        
        # convert labels to tensors:
        image_labels = torch.tensor(image_labels)
        audio_labels = torch.tensor(audio_labels)

        # create batch dictionary:
        batch_dict = {
            "image": images,
            "image_label": image_labels,
            "audio": audios,
            "audio_label": audio_labels
        }

        return batch_dict

