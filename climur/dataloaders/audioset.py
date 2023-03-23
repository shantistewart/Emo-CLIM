"""PyTorch dataset class for AudioSet music mood subset."""


import os
import torch
from torch.utils.data import Dataset
from torch import Tensor
import torchaudio
import pandas as pd
from typing import Dict, Tuple


# expected sampling rate:
SAMPLE_RATE = 16000
# dictionary mapping original emotion tag names to shorter names:
EMOTION_TAGS_MAP = {
    "Happy music": "happy",
    "Funny music": "funny",
    "Sad music": "sad",
    "Tender music": "tender",
    "Exciting music": "exciting",
    "Angry music": "angry",
    "Scary music": "scary"
}
# dictionary mapping emotion tags to label indices:
EMOTION_TAG2IDX = {
    "happy": 0,
    "funny": 1,
    "sad": 2,
    "tender": 3,
    "exciting": 4,
    "angry": 5,
    "scary": 6
}


class AudioSetMood(Dataset):
    """Dataset class for AudioSet music mood subset.

    Attributes:
        metadata (DataFrame): Metadata.
        root (str): Path of top-level root directory of dataset.
        clip_length_sec (float): Target length of audio clips in seconds.
        sample_rate (int): Sampling rate.
        emotion_tags_map (Dict): Dictionary mapping original emotion tag names to shorter names:
        emotion_tags (list): Emotion tags vocabulary.
        emotion_tag2idx (Dict): Dictionary mapping emotion tags to label indices.
        audio_dir_name (str): Name of subdirectory containing audio files.
    """

    def __init__(self, root: str, subset: str, clip_length_sec: float, sample_rate: int = SAMPLE_RATE, emotion_tags_map: Dict = EMOTION_TAGS_MAP, emotion_tag2idx: Dict = EMOTION_TAG2IDX, audio_dir_name: str = "audio_files") -> None:
        """Initialization.

        Args:
            root (str): Path of root directory of dataset.
            subset (str): Dataset subset ("train" or "test").
            clip_length_sec (float): Target length of audio clips in seconds.
            sample_rate (int): Sampling rate.
            emotion_tags_map (Dict): Dictionary mapping original emotion tag names to shorter names:
            emotion_tag2idx (Dict): Dictionary mapping emotion tags to label indices.
            audio_dir_name (str): Name of subdirectory containing audio files.
        
        Returns: None
        """

        # validate dataset subset:
        if subset != "train" and subset != "test":
            raise ValueError("Invalid dataset subset.")
        # save parameters:
        self.root = root
        self.sample_rate = sample_rate
        self.emotion_tags_map = emotion_tags_map
        self.emotion_tag2idx = emotion_tag2idx
        self.audio_dir_name = audio_dir_name

        # convert clip length to samples:
        self.clip_length = int(self.sample_rate * clip_length_sec)

        # load metadata:
        self.metadata = pd.read_csv(os.path.join(self.root, f"labels_{subset}.csv"))
        # get emotion tags:
        orig_emotion_tags = self.metadata["label"].unique().tolist()
        self.emotion_tags = [self.emotion_tags_map[tag] for tag in orig_emotion_tags]
        assert set(self.emotion_tags) == set(list(self.emotion_tags_map.values())), "Error with emotion_tags_map."
    
    def __len__(self) -> int:
        """Gets length of dataset.

        Args: None

        Returns:
            dataset_len (int): Length of dataset.
        """

        dataset_len = self.metadata.shape[0]

        return dataset_len
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Gets an audio clip and its emotion label.

        Args:
            idx (int): Item index.
        
        Returns:
            audio (Tensor): Raw audio clip.
                shape: (clip_length, )
            label_idx (Tensor): Emotion label index.
        """

        # get audio file path:
        orig_subset = self.metadata.loc[idx, "orig_subset"]
        file_name = self.metadata.loc[idx, "file_name"]
        file_path = os.path.join(self.root, self.audio_dir_name, orig_subset, file_name)
        # load audio:
        audio, sample_rate = torchaudio.load(file_path)
        audio = audio.squeeze(dim=0)
        assert sample_rate == self.sample_rate, "Unexpected sampling rate."

        # crop to target clip length:
        length = audio.size(dim=0)
        if length < self.clip_length:
            raise RuntimeError("Audio clip is too short")
        # TODO: randomly (?) crop to target clip length:

        # get emotion tag:
        orig_tag = self.metadata.loc[idx, "label"]
        tag = self.emotion_tags_map[orig_tag]
        # map to label index:
        label_idx = torch.tensor(self.emotion_tag2idx[tag], dtype=int)

        return audio, label_idx

