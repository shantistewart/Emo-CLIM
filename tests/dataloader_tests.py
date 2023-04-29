"""Script for testing dataset classes and dataloaders."""


import torch
from climur.dataloaders.imac_images import IMACImages
from climur.dataloaders.audioset import AudioSetMood
from climur.dataloaders.multimodal import Multimodal
from climur.utils.constants import CLIP_IMAGE_SIZE


# dataset paths:
IMAC_IMAGES_DATA_ROOT = "/proj/systewar/datasets/IMAC/image_dataset"
IMAC_IMAGES_METADATA_FILE = "metadata_train.csv"
AUDIOSET_DATA_ROOT = "/proj/systewar/datasets/audioset_music_mood"
AUDIOSET_METADATA_FILE = "new_split_metadata_files/metadata_train.csv"

# script options:
example_idx = 9
batch_size = 16
# for AudioSet:
sample_rate = 16000
clip_length_sec = 5.0
clip_length_samples = int(clip_length_sec * sample_rate)
overlap_ratio = 0.75     # only used for evaluation mode
# for multimodal dataset:
effective_length = 10000

# data augmentation options:
audio_augment_params = {
    "n_views": 2,
    "gaussian_noise": {
        "prob": 0.8,
        "min_snr": 5.0,
        "max_snr": 40.0
    }
}
image_augment_params = {
    "n_views": 2
}


if __name__ == "__main__":
    print("\n")

    # IMACImages CLASS TESTS:
    print("Testing IMACImages class...")

    # create dataset:
    imac_dataset = IMACImages(
        root=IMAC_IMAGES_DATA_ROOT,
        metadata_file_name=IMAC_IMAGES_METADATA_FILE,
        augment_params=image_augment_params,
        eval=False,
        preprocess=None
    )

    # test __len__() method:
    print("Testing __len__() method...")
    assert len(imac_dataset) == imac_dataset.metadata.shape[0], "Error with __len__() method."

    # test __getitem__() method:
    print("Testing __getitem__() method...")
    image, tag = imac_dataset[example_idx]
    if image_augment_params is not None:
        assert tuple(image.size()) == (image_augment_params["n_views"], 3, CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE), "Error with image shape."
        assert not torch.equal(image[0], image[1]), "Different augmented views are equal."
    else:
        assert tuple(image.size()) == (1, 3, CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE), "Error with image shape."
    
    # TODO: Test IMACImages class in eval mode.


    # AudioSetMood CLASS TESTS:
    print("\n\nTesting AudioSetMood class...")

    # create dataset:
    audioset_dataset = AudioSetMood(
        root=AUDIOSET_DATA_ROOT,
        metadata_file_name=AUDIOSET_METADATA_FILE,
        clip_length_samples=clip_length_samples,
        sample_rate=sample_rate,
        augment_params=audio_augment_params,
        eval=False
    )

    # test __len__() method:
    print("Testing __len__() method...")
    assert len(audioset_dataset) == audioset_dataset.metadata.shape[0], "Error with __len__() method."

    # test __getitem__() method:
    print("Testing __getitem__() method...")
    audio, tag = audioset_dataset[example_idx]
    if audio_augment_params is not None:
        assert tuple(audio.size()) == (audio_augment_params["n_views"], clip_length_samples), "Error with audio shape."
        assert not torch.equal(audio[0], audio[1]), "Different augmented views are equal."
    else:
        assert tuple(audio.size()) == (1, clip_length_samples), "Error with audio shape."


    # AudioSetMood CLASS EVALUATION MODE TESTS:
    print("\nTesting AudioSetMood class in evaluation mode...")

    # create dataset:
    audioset_dataset_eval = AudioSetMood(
        root=AUDIOSET_DATA_ROOT,
        metadata_file_name=AUDIOSET_METADATA_FILE,
        clip_length_samples=clip_length_samples,
        sample_rate=sample_rate,
        eval=True,
        overlap_ratio=overlap_ratio
    )

    # test __len__() method:
    print("Testing __len__() method...")
    assert len(audioset_dataset_eval) == audioset_dataset_eval.metadata.shape[0], "Error with __len__() method."

    # test __getitem__() method:
    print("Testing __getitem__() method...")
    audio_chunks, tag = audioset_dataset_eval[example_idx]
    assert len(tuple(audio_chunks.size())) == 3 and audio_chunks.size(dim=1) == 1 and audio_chunks.size(dim=-1) == clip_length_samples, "Error with audio shape."



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
    if image_augment_params is not None:
        assert tuple(item["image"].size()) == (image_augment_params["n_views"], 3, CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE), "Error with image shape."
        assert not torch.equal(item["image"][0], item["image"][1]), "Different augmented views are equal."
    else:
        assert tuple(item["image"].size()) == (1, 3, CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE), "Error with image shape."
    if audio_augment_params is not None:
        assert tuple(item["audio"].size()) == (audio_augment_params["n_views"], clip_length_samples), "Error with audio shape."
        assert not torch.equal(item["audio"][0], item["audio"][1]), "Different augmented views are equal."
    else:
        assert tuple(item["audio"].size()) == (1, clip_length_samples), "Error with audio shape."
    assert type(item["image_label"]) == int and type(item["audio_label"]) == int, "Error with data type of emotion label indices."

    print()
    print("image shape: {}".format(tuple(item["image"].size())))
    print("image emotion label: {}".format(multimodal_dataset.idx2label[item["image_label"]]))
    print("audio shape: {}".format(tuple(item["audio"].size())))
    print("audio emotion label: {}".format(multimodal_dataset.idx2label[item["audio_label"]]))


    # Multimodal CLASS COLLATE FUNCTION TESTS:
    print("\n\n\nTesting Multimodal class's custom collate function...")

    # create example batch:
    batch_input = [multimodal_dataset[-100] for _ in range(batch_size)]
    # test custom collate function:
    batch = multimodal_dataset.collate_fn(batch_input)

    assert type(batch) == dict and len(batch) == 4, "Error with batch dictionary."
    if image_augment_params is not None:
        assert tuple(batch["image"].size()) == (batch_size * image_augment_params["n_views"], 3, CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE), "Error with images shape."
        assert tuple(batch["image_label"].size()) == (batch_size * image_augment_params["n_views"], ), "Error with image labels shape."
    else:
        assert tuple(batch["image"].size()) == (batch_size, 3, CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE), "Error with images shape."
        assert tuple(batch["image_label"].size()) == (batch_size, ), "Error with image labels shape."
    if audio_augment_params is not None:
        assert tuple(batch["audio"].size()) == (batch_size * audio_augment_params["n_views"], clip_length_samples), "Error with audio clips shape."
        assert tuple(batch["audio_label"].size()) == (batch_size * audio_augment_params["n_views"], ), "Error with audio labels shape."
    else:
        assert tuple(batch["audio"].size()) == (batch_size, clip_length_samples), "Error with audio clips shape."
        assert tuple(batch["audio_label"].size()) == (batch_size, ), "Error with audio labels shape."
    
    print()
    print("images shape: {}".format(tuple(batch["image"].size())))
    print("image emotion labels shape: {}".format(tuple(batch["image_label"].size())))
    print("audio clips shape: {}".format(tuple(batch["audio"].size())))
    print("audio emotion labels shape: {}".format(tuple(batch["audio_label"].size())))


    print("\n")

