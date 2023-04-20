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
        eval (bool): Selects whether to get audio clips in evaluation mode.
        overlap_ratio (float): Overlap ratio between adjacent chunks for evalutation mode (must be in range [0, 0.9]).
        emotion_tags (list): Emotion tags vocabulary.
        audio_dir_name (str): Name of subdirectory containing audio files.
        audio_model (str): Name of audio backbone model being used (only used for CLAP).
    """

    def __init__(self, root: str, metadata_file_name: str, clip_length_samples: int, sample_rate: int = SAMPLE_RATE, eval: bool = False, overlap_ratio: float = 0.0, emotion_tags_map: Dict = EMOTION_TAGS_MAP, audio_dir_name: str = "audio_files", audio_model: str = "ShortChunk") -> None:
        """Initialization.

        Args:
            root (str): Path of root directory of dataset.
            metadata_file_name (str): Name of metadata file.
            clip_length_samples (int): Target length of audio clips in samples.
            sample_rate (int): Sampling rate.
            eval (bool): Selects whether to get audio clips in evaluation mode.
            overlap_ratio (float): Overlap ratio between adjacent chunks for evalutation mode (must be in range [0, 0.9]).
            emotion_tags_map (Dict): Dictionary mapping original emotion tag names to shorter names.
            audio_dir_name (str): Name of subdirectory containing audio files.
            audio_model (str): Name of audio backbone model being used (only used for CLAP).
        
        Returns: None
        """

        # validate overlap ratio:
        if overlap_ratio < 0.0 or overlap_ratio > 0.9:
            raise ValueError("Invalid overlap ratio value.")
        
        # save parameters:
        self.root = root
        self.clip_length = clip_length_samples
        self.sample_rate = sample_rate
        self.eval = eval
        self.overlap_ratio = overlap_ratio
        self.audio_dir_name = audio_dir_name
        self.audio_model = audio_model

        # load metadata:
        orig_metadata = pd.read_csv(os.path.join(self.root, metadata_file_name))
        # filter out audio clips that are too short:
        if self.audio_model == "CLAP":
            # CLAP uses 48000 Hz as the input
            self.metadata = orig_metadata[orig_metadata["length_samples"] >= self.clip_length//3]
        else:
            self.metadata = orig_metadata[orig_metadata["length_samples"] >= self.clip_length]
        self.metadata = self.metadata.reset_index(drop=True)

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
            if not eval:
                audio (Tensor): Randomly cropped raw audio clip.
                    shape: (clip_length, )
            else:
                audio_chunks (Tensor): Raw audio clip chunked with a sliding window.
                    shape: (n_chunks, clip_length)
            tag (str): Emotion tag.
        """

        # get audio file path:
        orig_subset = self.metadata.loc[idx, "orig_subset"]
        file_name = self.metadata.loc[idx, "file_name"]
        file_path = os.path.join(self.root, self.audio_dir_name, orig_subset, file_name)
        # load audio:
        audio, sample_rate = torchaudio.load(file_path)
        if self.audio_model == "CLAP":
            audio = torchaudio.functional.resample(
                audio, 
                orig_freq=sample_rate, 
                new_freq=48000
            )
            sample_rate = 48000
        audio = audio.squeeze(dim=0)
        assert sample_rate == self.sample_rate, "Unexpected sampling rate."

        # check that audio clip is long enough:
        length = audio.size(dim=0)
        if length < self.clip_length:
            raise RuntimeError("Audio clip is too short.")
        
        if not self.eval:
            # randomly crop to target clip length:
            start_idx = np.random.randint(low=0, high=length - self.clip_length + 1)
            end_idx = start_idx + self.clip_length
            audio = audio[start_idx : end_idx]
            assert audio.size(dim=0) == self.clip_length, "Error with cropping audio clip."
        
        else:
            # split audio clip into chunks:
            step = int(np.around((1 - self.overlap_ratio) * self.clip_length))
            audio_chunks = audio.unfold(dimension=0, size=self.clip_length, step=step)
            # sanity check shape:
            assert len(tuple(audio_chunks.size())) == 2 and audio_chunks.size(dim=-1) == self.clip_length, "Error with shape of chunked audio clip."
        
        # get emotion tag:
        tag = self.metadata.loc[idx, "label"]

        # CLAP input requires audio load in int16 format
        if self.audio_model == "CLAP":
            audio = np.clip(audio, a_min=-1., a_max=1.)
            audio = (audio * 32767.).int()
            audio = (audio / 32767.0).float()
        
        if not self.eval:
            return audio, tag
        else:
            return audio_chunks, tag

