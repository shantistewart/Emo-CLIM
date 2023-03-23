"""Script for testing dataset classes and dataloaders."""


from torch.utils.data import DataLoader
from climur.dataloaders.audioset import AudioSetMood


# dataset path:
DATA_ROOT = "/proj/systewar/datasets/audioset_music_mood"
# script options:
subset = "train"
clip_length_sec = 10.0
sample_rate = 16000
example_idx = 9
batch_size = 8


if __name__ == "__main__":
    print("\n")

    # create dataset:
    audioset_dataset = AudioSetMood(
        root=DATA_ROOT,
        subset=subset,
        clip_length_sec=clip_length_sec,
        sample_rate=sample_rate
    )


    # DATASET TESTS:

    # test __len__() method:
    print("Testing __len__() method...")
    assert len(audioset_dataset) == audioset_dataset.metadata.shape[0], "Error with __len__() method."

    # test __getitem__() method:
    print("\nTesting __getitem__() method...")
    audio, label_idx = audioset_dataset[example_idx]
    assert len(tuple(audio.size())) == 1 and audio.size(dim=0) == int(sample_rate * clip_length_sec), "Error with audio shape."


    print("\n")

