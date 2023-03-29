"""Script for testing dataset classes and dataloaders."""


from climur.dataloaders.multimodal import Multimodal
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
# for multimodal dataset:
same_tag_prob = 0.5


if __name__ == "__main__":
    print("\n")

    # IMACImages CLASS TESTS:
    print("Testing IMACImages class...")

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


    # AudioSetMood CLASS TESTS:
    print("\n\nTesting AudioSetMood class...")

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


    # Multimodal CLASS TESTS:
    print("\n\n\nTesting Multimodal class...")
    multimodal_dataset = Multimodal(
        image_dataset=imac_dataset,
        audio_dataset=audioset_dataset,
        same_tag_prob=same_tag_prob
    )

    # test __len__() method:
    print("\nTesting __len__() method...")
    print("Size of dataset: {}".format(len(multimodal_dataset)))

    # test __getitem__() method:
    print("\nTesting __getitem__() method...")
    item = multimodal_dataset[-100]

    assert type(item) == dict and len(item) == 4, "Error with item dictionary."
    assert len(tuple(item["image"].size())) == 3 and item["image"].size(dim=0) == 3, "Error with image shape."
    assert len(tuple(item["audio"].size())) == 1 and item["audio"].size(dim=0) == int(sample_rate * clip_length_sec), "Error with audio shape."
    assert type(item["image_label"]) == str and type(item["audio_label"]) == str, "Error with data type of emotion tags."

    print()
    print("image shape: {}".format(tuple(item["image"].size())))
    print("image emotion label: {}".format(item["image_label"]))
    print("audio shape: {}".format(tuple(item["audio"].size())))
    print("audio emotion label: {}".format(item["audio_label"]))


    print("\n")

