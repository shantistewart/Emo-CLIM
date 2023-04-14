"""PyTorch dataset class for retrieving images and audio clips."""


from torch.utils.data import Dataset
from torch import Tensor
import random
from typing import Union, Dict


# default manual mapping from image dataset emotion tags to audio dataset emotion tags:
IMAGE2AUDIO_TAG_MAP = {
    "excitement": "exciting",
    "contentment": "happy",
    "amusement": "funny",     # TODO: not totally sure about this one (amusement may be closer to "entertaining" than "funny")
    "anger": "angry",
    "fear": "scary",
    "sadness": "sad"
}


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
            image2audio_tag_map (dict): Dictinoary mapping image dataset emotion tags to audio dataset emotion tags.
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
                "image": (Tensor) Raw image.
                    shape: (image_channels, image_height, image_width)
                "image_label" (int): Image emotion label index.
                "audio": (Tensor) Raw audio clip.
                    shape: (audio_clip_length, )
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

