"""PyTorch dataset class for retrieving images and audio clips."""


from torch.utils.data import Dataset
from torch import Tensor
import numpy as np
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
        n_classes (int): Number of (common) emotion tag classes.
        image2audio_tag_map (dict): Dictionary mapping image dataset emotion tags to audio dataset emotion tags.
        image_tags (list): Image dataset emotion tags.
        audio_tags (list): Audio dataset emotion tags.
        tag_pairs (dict): All possible pairs of (image emotion tag, audio emotion tag).
        same_tag_prob (float): Probability that retrieved image and audio clip have the same emotion tag.
    """

    def __init__(self, image_dataset: Dataset, audio_dataset: Dataset, image2audio_tag_map: Dict = IMAGE2AUDIO_TAG_MAP, same_tag_prob: float = 0.5) -> None:
        """Initialization.

        Args:
            image_dataset (PyTorch Dataset): Image dataset.
            audio_dataset (PyTorch Dataset): Audio dataset.
            image2audio_tag_map (dict): Dictinoary mapping image dataset emotion tags to audio dataset emotion tags.
            same_tag_prob (float): Probability that retrieved image and audio clip have the same emotion tag.
        
        Returns: None
        """

        # validate params:
        if same_tag_prob < 0.0 or same_tag_prob > 1.0:
            raise ValueError("same_tag_prob not in [0, 1].")
        assert set(list(image2audio_tag_map.keys())) <= set(image_dataset.emotion_tags), "image2audio_tag_map contains emotion tags not in image dataset."
        assert set(list(image2audio_tag_map.values())) <= set(audio_dataset.emotion_tags), "image2audio_tag_map contains emotion tags not in image dataset."

        # save params:
        self.image2audio_tag_map = image2audio_tag_map
        self.same_tag_prob = same_tag_prob
        # save things related to emotion tags:
        self.n_classes = len(image2audio_tag_map)
        self.image_tags = list(image2audio_tag_map.keys())
        self.audio_tags = list(image2audio_tag_map.values())

        # save (original) datasets:
        self.image_dataset = image_dataset
        self.audio_dataset = audio_dataset

        # create dictionary mapping emotion tags to image file names:
        self.tag2image = {}
        for tag in self.image_tags:
            self.tag2image[tag] = image_dataset.metadata[image_dataset.metadata["label"] == tag]
        # create dictionary mapping emotion tags to audio file names:
        self.tag2audio = {}
        for tag in self.audio_tags:
            self.tag2audio[tag] = audio_dataset.metadata[audio_dataset.metadata["label"] == tag]
        
        # create all possible pairs of (image emotion tag, audio emotion tag):
        self.tag_pairs = {
            "same": [],
            "different": []
        }
        for image_tag in self.image_tags:
            for audio_tag in self.audio_tags:
                # if same tag:
                if self.image2audio_tag_map[image_tag] == audio_tag:
                    self.tag_pairs["same"].append((image_tag, audio_tag))
                # else different tag:
                else:
                    self.tag_pairs["different"].append((image_tag, audio_tag))
        assert len(self.tag_pairs["same"]) + len(self.tag_pairs["different"]) == len(self.image_tags) * len(self.audio_tags), "Error creating multimodal tag pairs."
    
    def __len__(self) -> int:
        """Gets effective length of dataset.

        Args: None

        Returns:
            dataset_len (int): Minimum of effective sizes of image and audio dataset.
        """

        # compute effective image dataset length:
        image_dataset_len = 0
        for df in self.tag2image.values():
            image_dataset_len += df.shape[0]
        
        # compute effective audio dataset length:
        audio_dataset_len = 0
        for df in self.tag2audio.values():
            audio_dataset_len += df.shape[0]
        
        dataset_len = min(image_dataset_len, audio_dataset_len)

        return dataset_len
    
    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, str]]:
        """Randomly retrieves an (image, audio clip) pair and their emotion tags.

        Args:
            idx (int): Item index (not used due to random selection).
        
        Returns:
            item (dict): Item dictionary with keys/values:
                "image": (Tensor) Raw image.
                    shape: (image_channels, image_height, image_width)
                "image_label" (str): Image emotion label.
                "audio": (Tensor) Raw audio clip.
                    shape: (audio_clip_length, )
                "audio_label" (str): Audio emotion label.
        """

        # randomly choose if image and audio clip will have the same emotion tag (with probability same_tag_prob):
        same_tag = np.random.choice([True, False], p=[self.same_tag_prob, 1 - self.same_tag_prob])

        # randomly select a (image emotion tag, audio emotion tag) pair:
        if same_tag:
            (image_tag, audio_tag) = random.choice(self.tag_pairs["same"])
        else:
            (image_tag, audio_tag) = random.choice(self.tag_pairs["different"])
        
        # randomly select an image labeled with selected emotion tag:
        image_idx = int(self.tag2image[image_tag].sample(n=1, axis="index").index[0])
        image, tag = self.image_dataset[image_idx]
        assert tag == image_tag, "Image has incorrect emotion label."

        # randomly select an audio clip labeled with selected emotion tag:
        audio_idx = int(self.tag2audio[audio_tag].sample(n=1, axis="index").index[0])
        audio, tag = self.audio_dataset[audio_idx]
        assert tag == audio_tag, "Audio clip has incorrect emotion label."

        # return as an item dictionary (note: image emotion tag is mapped to equivalent audio emotion tag):
        item = {
            "image": image,
            "image_label": self.image2audio_tag_map[image_tag],
            "audio": audio,
            "audio_label": audio_tag
        }

        return item

