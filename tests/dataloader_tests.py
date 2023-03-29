"""Script for testing dataset classes and dataloaders."""


from climur.dataloaders.audioset import AudioSetMood
from climur.dataloaders.imac_images import IMACImages


# dataset paths:
AUDIOSET_DATA_ROOT = "/proj/systewar/datasets/audioset_music_mood"
IMAC_IMAGES_DATA_ROOT = "/proj/systewar/datasets/IMAC/image_dataset"

# script options:
subset = "train"
example_idx = 9
# for AudioSet:
clip_length_sec = 10.0
sample_rate = 16000


if __name__ == "__main__":
    print("\n")

    # AudioSetMood CLASS TESTS:
    print("Testing AudioSetMood class...")

    # create dataset:
    audioset_dataset = AudioSetMood(
        root=AUDIOSET_DATA_ROOT,
        subset=subset,
        clip_length_sec=clip_length_sec,
        sample_rate=sample_rate
    )

    # test __len__() method:
    print("Testing __len__() method...")
    assert len(audioset_dataset) == audioset_dataset.metadata.shape[0], "Error with __len__() method."

    # test __getitem__() method:
    print("Testing __getitem__() method...")
    audio, tag = audioset_dataset[example_idx]
    assert len(tuple(audio.size())) == 1 and audio.size(dim=0) == int(sample_rate * clip_length_sec), "Error with audio shape."


    # IMACImages CLASS TESTS:
    print("\n\nTesting IMACImages class...")

    # create dataset:
    imac_dataset = IMACImages(
        root=IMAC_IMAGES_DATA_ROOT,
        subset=subset
    )

    # test __len__() method:
    print("Testing __len__() method...")
    assert len(imac_dataset) == imac_dataset.metadata.shape[0], "Error with __len__() method."

    # test __getitem__() method:
    print("Testing __getitem__() method...")
    image, tag = imac_dataset[example_idx]
    assert len(tuple(image.size())) == 3 and image.size(dim=0) == 3, "Error with image shape."


    print("\n")

