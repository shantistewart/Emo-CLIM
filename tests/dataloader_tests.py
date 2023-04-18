"""Script for testing dataset classes and dataloaders."""


from climur.dataloaders.imac_images import IMACImages
from climur.dataloaders.audioset import AudioSetMood
from climur.dataloaders.multimodal import Multimodal


# dataset paths:
IMAC_IMAGES_DATA_ROOT = "/proj/systewar/datasets/IMAC/image_dataset"
IMAC_IMAGES_METADATA_FILE = "metadata_train.csv"
AUDIOSET_DATA_ROOT = "/proj/systewar/datasets/audioset_music_mood"
AUDIOSET_METADATA_FILE = "new_split_metadata_files/metadata_train.csv"

# script options:
example_idx = 9
# for AudioSet:
sample_rate = 16000
clip_length_sec = 5.0
clip_length_samples = int(clip_length_sec * sample_rate)
# for multimodal dataset:
effective_length = 10000


if __name__ == "__main__":
    print("\n")

    # IMACImages CLASS TESTS:
    print("Testing IMACImages class...")

    # create dataset:
    imac_dataset = IMACImages(
        root=IMAC_IMAGES_DATA_ROOT,
        metadata_file_name=IMAC_IMAGES_METADATA_FILE,
        preprocess=None
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
        metadata_file_name=AUDIOSET_METADATA_FILE,
        clip_length_samples=clip_length_samples,
        sample_rate=sample_rate
    )

    # test __len__() method:
    print("Testing __len__() method...")
    assert len(audioset_dataset) == audioset_dataset.metadata.shape[0], "Error with __len__() method."

    # test __getitem__() method:
    print("Testing __getitem__() method...")
    audio, tag = audioset_dataset[example_idx]
    assert len(tuple(audio.size())) == 1 and audio.size(dim=0) == clip_length_samples, "Error with audio shape."


    # Multimodal CLASS TESTS:
    print("\n\n\nTesting Multimodal class...")
    multimodal_dataset = Multimodal(
        image_dataset=imac_dataset,
        audio_dataset=audioset_dataset,
        length=effective_length
    )

    # test __len__() method:
    print("\nTesting __len__() method...")
    print("Length of dataset: {}".format(len(multimodal_dataset)))
    print("Effective length of image dataset: {}".format(multimodal_dataset.image_dataset_len))
    print("Effective length of audio dataset: {}".format(multimodal_dataset.audio_dataset_len))

    # test __getitem__() method:
    print("\nTesting __getitem__() method...")
    item = multimodal_dataset[-100]

    assert type(item) == dict and len(item) == 4, "Error with item dictionary."
    assert len(tuple(item["image"].size())) == 3 and item["image"].size(dim=0) == 3, "Error with image shape."
    assert len(tuple(item["audio"].size())) == 1 and item["audio"].size(dim=0) == clip_length_samples, "Error with audio shape."
    assert type(item["image_label"]) == int and type(item["audio_label"]) == int, "Error with data type of emotion label indices."

    print()
    print("image shape: {}".format(tuple(item["image"].size())))
    print("image emotion label: {}".format(multimodal_dataset.idx2label[item["image_label"]]))
    print("audio shape: {}".format(tuple(item["audio"].size())))
    print("audio emotion label: {}".format(multimodal_dataset.idx2label[item["audio_label"]]))


    print("\n")

