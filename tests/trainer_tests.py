"""Script for testing trainers (PyTorch Lightning LightningModule classes)."""


import os
import torch
from torch.utils.data import DataLoader
import torchinfo
import clip

from climur.dataloaders.imac_images import IMACImages
from climur.dataloaders.audioset import AudioSetMood
from climur.dataloaders.multimodal import Multimodal
from climur.models.image_backbones import CLIPModel
from climur.models.audio_backbones import HarmonicCNNEmbeddings
from climur.models.audio_model_components import HarmonicCNN
from climur.trainers.image2music import Image2Music


# dataset paths:
IMAC_IMAGES_DATA_ROOT = "/proj/systewar/datasets/IMAC/image_dataset"
IMAC_IMAGES_METADATA_FILE = "metadata_train.csv"
AUDIOSET_DATA_ROOT = "/proj/systewar/datasets/audioset_music_mood"
AUDIOSET_METADATA_FILE = "metadata_unbalanced_train.csv"

# image constants:
IMAGE_CHANNELS = 3
IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224
IMAGE_EMBED_SIZE = 512
# audio constants:
SAMPLE_RATE = 16000
AUDIO_CLIP_LENGTH = 5 * SAMPLE_RATE     # 5.0 seconds
AUDIO_EMBED_SIZE = 256

# script options:
# device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")     # TODO: Figure out why dtype error occurs with GPU.
device = torch.device("cpu")
# for audio backbone model:
pretrained_audio_backbone_path = "/proj/systewar/pretrained_models/music_tagging/msd/harmonic_cnn/best_model.pth"
last_layer_embed = "layer7"
shrink_freq = True
shrink_time = True
pool_type = "max"
# for full model:
joint_embed_dim = 128
# for training:
batch_size = 16
optimizer = "Adam"
learn_rate = 0.001
verbose = True


if __name__ == "__main__":
    print("\n")


    # ---------------
    # BACKBONE MODELS
    # ---------------

    if verbose:
        print("\nSetting up backbone models...")
    
    # load CLIP model:
    orig_clip_model, image_preprocess_transform = clip.load("ViT-B/32", device=device)
    # create CLIP wrapper model:
    image_backbone = CLIPModel(orig_clip_model)
    image_backbone.to(device)

    # load pretrained full Harmonic CNN model:
    full_hcnn_model = HarmonicCNN()
    full_hcnn_model.load_state_dict(torch.load(pretrained_audio_backbone_path, map_location=device))
    full_hcnn_model.to(device)
    # create HarmonicCNNEmbeddings model:
    sample_audio_input = torch.rand((batch_size, AUDIO_CLIP_LENGTH))
    sample_audio_input = sample_audio_input.to(device)
    audio_backbone = HarmonicCNNEmbeddings(
        full_hcnn_model,
        sample_input=sample_audio_input,
        last_layer=last_layer_embed,
        shrink_freq=shrink_freq,
        shrink_time=shrink_time,
        pool_type=pool_type
    )
    audio_backbone.to(device)


    # ------------
    # DATA LOADERS
    # ------------

    if verbose:
        print("\nSetting up datasets and data loaders...")
    
    # create image dataset:
    image_dataset = IMACImages(
        root=IMAC_IMAGES_DATA_ROOT,
        metadata_file_name=IMAC_IMAGES_METADATA_FILE,
        preprocess=image_preprocess_transform
    )
    # create audio dataset:
    audio_dataset = AudioSetMood(
        root=AUDIOSET_DATA_ROOT,
        metadata_file_name=AUDIOSET_METADATA_FILE,
        clip_length_samples=AUDIO_CLIP_LENGTH,
        sample_rate=SAMPLE_RATE
    )
    # create multimodal dataset:
    multimodal_dataset = Multimodal(
        image_dataset=image_dataset,
        audio_dataset=audio_dataset
    )
    if verbose:
        print("Dataset size: {}".format(len(multimodal_dataset)))
    
    # create dataloader:
    dataloader = DataLoader(
        multimodal_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    # test example batch:
    batch_example = next(iter(dataloader))
    assert type(batch_example) == dict, "Example batch is of incorrect data type.."
    assert len(batch_example) == 4, "Example batch has incorrect size."
    for key, value in batch_example.items():
        if key == "image_label" or key == "audio_label":
            assert tuple(value.size()) == (batch_size, ), "Error with shape of {}".format(key)
        elif key == "image":
            assert tuple(value.size()) == (batch_size, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), "Error with shape of {}".format(key)
        elif key == "audio":
            assert tuple(value.size()) == (batch_size, AUDIO_CLIP_LENGTH), "Error with shape of {}".format(key)
        else:
            raise RuntimeError("Unexpected key in example batch dictionary.")
    

    # ----------
    # FULL MODEL
    # ----------

    if verbose:
        print("\nSetting up full model...")
    
    # create full model:
    hparams = {
        "optimizer": optimizer,
        "learn_rate": learn_rate
    }
    full_model = Image2Music(
        image_backbone=image_backbone,
        audio_backbone=audio_backbone,
        joint_embed_dim=joint_embed_dim,
        hparams=hparams,
        freeze_image_backbone=True,
        freeze_audio_backbone=True
    )
    full_model.to(device)


    # -----
    # TESTS
    # -----

    # test forward() method:
    if verbose:
        print()
        print("\nTesting forward() method...\n")
    # unpack batch:
    images = batch_example["image"].to(device)
    audios = batch_example["audio"].to(device)
    if verbose:
        print("Input images size: {}".format(tuple(images.size())))
        print("Input audio clips size: {}".format(tuple(audios.size())))
    
    # test forward pass:
    image_embeds, audio_embeds = full_model.forward(images, audios)
    if verbose:
        print("Image embeddings size: {}".format(tuple(image_embeds.size())))
        print("Audio embeddings size: {}".format(tuple(audio_embeds.size())))
    assert tuple(image_embeds.size()) == (batch_size, joint_embed_dim), "Error with shape of image embeddings."
    assert tuple(audio_embeds.size()) == (batch_size, joint_embed_dim), "Error with shape of image embeddings."

    # test configure_optimizers() method:
    if verbose:
        print("\nTesting configure_optimizers() method...")
    optimizer = full_model.configure_optimizers()


    print("\n")

