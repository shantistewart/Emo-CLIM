"""PyTorch dataset class for AudioSet music mood subset."""


import os
from torch.utils.data import Dataset
from torch import Tensor
import torchaudio
import pandas as pd
import numpy as np
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


class AudioSetMood(Dataset):
    """Dataset class for AudioSet music mood subset.

    Attributes:
        metadata (DataFrame): Metadata.
        root (str): Path of top-level root directory of dataset.
        clip_length (int): Target length of audio clips in samples.
        sample_rate (int): Sampling rate.
        emotion_tags (list): Emotion tags vocabulary.
        audio_dir_name (str): Name of subdirectory containing audio files.
    """

    def __init__(self, root: str, subset: str, clip_length_samples: int, sample_rate: int = SAMPLE_RATE, emotion_tags_map: Dict = EMOTION_TAGS_MAP, audio_dir_name: str = "audio_files") -> None:
        """Initialization.

        Args:
            root (str): Path of root directory of dataset.
            subset (str): Dataset subset ("train" or "test").
            clip_length_samples (int): Target length of audio clips in samples.
            sample_rate (int): Sampling rate.
            emotion_tags_map (Dict): Dictionary mapping original emotion tag names to shorter names.
            audio_dir_name (str): Name of subdirectory containing audio files.
        
        Returns: None
        """

        # validate dataset subset:
        if subset != "train" and subset != "test":
            raise ValueError("Invalid dataset subset.")
        # save parameters:
        self.root = root
        self.clip_length = clip_length_samples
        self.sample_rate = sample_rate
        self.audio_dir_name = audio_dir_name

        # load metadata:
        self.metadata = pd.read_csv(os.path.join(self.root, f"labels_{subset}.csv"))
        orig_emotion_tags = self.metadata["label"].unique().tolist()
        # map original emotion tag names to shorter names:
        for idx in range(self.metadata.shape[0]):
            self.metadata.loc[idx, "label"] = emotion_tags_map[self.metadata.loc[idx, "label"]]
        # get new emotion tags:
        self.emotion_tags = self.metadata["label"].unique().tolist()
        assert len(self.emotion_tags) == len(orig_emotion_tags), "Error with mapping original emotion tag names to shorter names."
        assert set(self.emotion_tags) == set(list(emotion_tags_map.values())), "Error with mapping original emotion tag names to shorter names."
    
    def __len__(self) -> int:
        """Gets length of dataset.

        Args: None

        Returns:
            dataset_len (int): Length of dataset.
        """

        dataset_len = self.metadata.shape[0]

        return dataset_len
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, str]:
        """Gets an audio clip and its emotion tag.

        Args:
            idx (int): Item index.
        
        Returns:
            audio (Tensor): Raw audio clip.
                shape: (clip_length, )
            tag (str): Emotion tag.
        """

        # get audio file path:
        orig_subset = self.metadata.loc[idx, "orig_subset"]
        file_name = self.metadata.loc[idx, "file_name"]
        file_path = os.path.join(self.root, self.audio_dir_name, orig_subset, file_name)
        # load audio:
        audio, sample_rate = torchaudio.load(file_path)
        audio = audio.squeeze(dim=0)
        assert sample_rate == self.sample_rate, "Unexpected sampling rate."

        # check that audio clip is long enough:
        length = audio.size(dim=0)
        if length < self.clip_length:
            raise RuntimeError("Audio clip is too short.")
        # randomly crop to target clip length:
        start_idx = np.random.randint(low=0, high=length - self.clip_length + 1)
        end_idx = start_idx + self.clip_length
        audio = audio[start_idx : end_idx]
        assert audio.size(dim=0) == self.clip_length, "Error with cropping audio clip."

        # get emotion tag:
        tag = self.metadata.loc[idx, "label"]

        return audio, tag

