"""Script for testing trainers (PyTorch Lightning LightningModule classes)."""


import os
import torch
from torch.utils.data import DataLoader
from torch import Tensor
import torchinfo
from copy import deepcopy
import clip

from climur.dataloaders.imac_images import IMACImages
from climur.dataloaders.audioset import AudioSetMood
from climur.dataloaders.multimodal import Multimodal
from climur.models.image_backbones import CLIPModel
from climur.models.audio_model_components import ShortChunkCNN_Res, HarmonicCNN
from climur.models.vcmr_trainer import VCMR
from climur.models.audio_backbones import ShortChunkCNNEmbeddings, HarmonicCNNEmbeddings, SampleCNNEmbeddings
from climur.trainers.image2music import Image2Music
from climur.utils.constants import (
    CLIP_IMAGE_SIZE,
    SHORTCHUNK_INPUT_LENGTH,
    HARMONIC_CNN_INPUT_LENGTH,
    SAMPLE_CNN_INPUT_LENGTH,
    CLIP_EMBED_DIM,
    SHORTCHUNK_EMBED_DIM,
    HARMONIC_CNN_EMBED_DIM,
    SAMPLE_CNN_EMBED_DIM,
    SAMPLE_CNN_DEFAULT_PARAMS,
    VCMR_DEFAULT_PARAMS
)


# dataset paths:
IMAC_IMAGES_DATA_ROOT = "/proj/systewar/datasets/IMAC/image_dataset"
IMAC_IMAGES_METADATA_FILE = "metadata_train.csv"
AUDIOSET_DATA_ROOT = "/proj/systewar/datasets/audioset_music_mood"
AUDIOSET_METADATA_FILE = "new_split_metadata_files/metadata_train.csv"

# image constants:
IMAGE_CHANNELS = 3
IMAGE_EMBED_DIM = CLIP_EMBED_DIM
# audio constants:
SAMPLE_RATE = 16000

# script options:
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
# for audio backbone model:
audio_backbone_name = "HarmonicCNN"     # or "ShortChunk" or "SampleCNN"
if audio_backbone_name == "ShortChunk":
    pretrained_audio_backbone_path = "/proj/systewar/pretrained_models/music_tagging/msd/short_chunk_resnet/best_model.pth"
    audio_clip_length = SHORTCHUNK_INPUT_LENGTH     # ~3.69 seconds
    audio_embed_dim = SHORTCHUNK_EMBED_DIM
elif audio_backbone_name == "HarmonicCNN":
    pretrained_audio_backbone_path = "/proj/systewar/pretrained_models/music_tagging/msd/harmonic_cnn/best_model.pth"
    audio_clip_length = HARMONIC_CNN_INPUT_LENGTH     # 5.0 seconds
    audio_embed_dim = HARMONIC_CNN_EMBED_DIM
elif audio_backbone_name == "SampleCNN":
    pretrained_audio_backbone_path = "/proj/systewar/pretrained_models/VCMR/multimodal/multimodal_model_1.ckpt"
    audio_clip_length = SAMPLE_CNN_INPUT_LENGTH     # ~6.15 seconds
    audio_embed_dim = SAMPLE_CNN_EMBED_DIM
last_layer_embed = "layer7"
pool_type = "max"

# for full model:
output_embed_dim = 128
multi_task = False
base_proj_hidden_dim = 256
base_proj_dropout = 0.2
base_proj_output_dim = 128
task_proj_dropout = 0.5
normalize_image_embeds = True
normalize_audio_embeds = True

# for training:
loss_temperature = 0.07
loss_weights = {
    "image2image": 0.25,
    "audio2audio": 0.25,
    "image2audio": 0.25,
    "audio2image": 0.25
}
batch_size = 32
n_batches = 100
optimizer = "Adam"
learn_rate = 0.001
verbose = True

# data augmentation options:
audio_augment_params = {
    "n_views": 2,
    "gaussian_noise": {
        "prob": 0.8,
        "min_snr": 5.0,     # in dB
        "max_snr": 40.0     # in dB
    },
    "background_noise": {
        "sounds_path": "/proj/systewar/datasets/NSynth/nsynth-train/audio",
        "prob": 0.8,
        "min_snr": 3.0,     # in dB
        "max_snr": 30.0     # in dB
    }
}
image_augment_params = {
    "n_views": 2
}


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

    # set up audio backbone model:
    sample_audio_input = torch.rand((2, audio_clip_length))
    if audio_backbone_name == "ShortChunk":
        # load pretrained full Short-Chunk CNN ResNet model:
        full_audio_backbone = ShortChunkCNN_Res()
        full_audio_backbone.load_state_dict(torch.load(pretrained_audio_backbone_path, map_location=torch.device("cpu")))
        # create wrapper model:
        audio_backbone = ShortChunkCNNEmbeddings(
            full_audio_backbone,
            sample_input=sample_audio_input,
            last_layer=last_layer_embed,
            pool_type=pool_type
        )
    
    elif audio_backbone_name == "HarmonicCNN":
        # load pretrained full Harmonic CNN model:
        full_audio_backbone = HarmonicCNN()
        full_audio_backbone.load_state_dict(torch.load(pretrained_audio_backbone_path, map_location=torch.device("cpu")))
        # create wrapper model:
        audio_backbone = HarmonicCNNEmbeddings(
            full_audio_backbone,
            sample_input=sample_audio_input,
            last_layer=last_layer_embed,
            pool_type=pool_type
        )
    
    elif audio_backbone_name == "SampleCNN":
        # create (empty) SampleCNNEmbeddings model:
        sample_cnn = SampleCNNEmbeddings(
            params=SAMPLE_CNN_DEFAULT_PARAMS
        )
        # load pretrained VCMR model:
        vcmr_model = VCMR.load_from_checkpoint(
            pretrained_audio_backbone_path,
            map_location=torch.device("cpu"),
            encoder=sample_cnn,
            video_params=VCMR_DEFAULT_PARAMS
        )
        # extract SampleCNNEmbeddings component:
        audio_backbone = deepcopy(vcmr_model.encoder)
    
    else:
        raise ValueError("{} model not supported".format(audio_backbone_name))
    
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
        augment_params=image_augment_params,
        eval=False
    )
    # create audio dataset:
    audio_dataset = AudioSetMood(
        root=AUDIOSET_DATA_ROOT,
        metadata_file_name=AUDIOSET_METADATA_FILE,
        clip_length_samples=audio_clip_length,
        sample_rate=SAMPLE_RATE,
        augment_params=audio_augment_params,
        eval=False
    )
    effective_length = n_batches * batch_size
    # create multimodal dataset:
    multimodal_dataset = Multimodal(
        image_dataset=image_dataset,
        audio_dataset=audio_dataset,
        length = effective_length
    )
    if verbose:
        print("Effective dataset length: {}".format(len(multimodal_dataset)))
    
    # create dataloader:
    dataloader = DataLoader(
        multimodal_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=multimodal_dataset.collate_fn,
        drop_last=True
    )
    assert len(dataloader) == n_batches, "Length of dataloader is incorrect."
    
    # test example batch:
    example_batch = next(iter(dataloader))
    assert type(example_batch) == dict and len(example_batch) == 4, "Error with example batch."
    if image_augment_params is not None:
        assert tuple(example_batch["image"].size()) == (batch_size * image_augment_params["n_views"], 3, CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE), "Error with images shape."
        assert tuple(example_batch["image_label"].size()) == (batch_size * image_augment_params["n_views"], ), "Error with image labels shape."
    else:
        assert tuple(example_batch["image"].size()) == (batch_size, 3, CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE), "Error with images shape."
        assert tuple(example_batch["image_label"].size()) == (batch_size, ), "Error with image labels shape."
    if audio_augment_params is not None:
        assert tuple(example_batch["audio"].size()) == (batch_size * audio_augment_params["n_views"], audio_clip_length), "Error with audio clips shape."
        assert tuple(example_batch["audio_label"].size()) == (batch_size * audio_augment_params["n_views"], ), "Error with audio labels shape."
    else:
        assert tuple(example_batch["audio"].size()) == (batch_size, audio_clip_length), "Error with audio clips shape."
        assert tuple(example_batch["audio_label"].size()) == (batch_size, ), "Error with audio labels shape."


    # ----------
    # FULL MODEL
    # ----------

    if verbose:
        print("\nSetting up full model...")
    
    # create full model:
    hparams = {
        "loss_temperature": loss_temperature,
        "loss_weights": loss_weights,
        "optimizer": optimizer,
        "learn_rate": learn_rate
    }
    full_model = Image2Music(
        image_backbone=image_backbone,
        audio_backbone=audio_backbone,
        output_embed_dim=output_embed_dim,
        image_embed_dim=IMAGE_EMBED_DIM,
        audio_embed_dim=audio_embed_dim,
        hparams=hparams,

        multi_task = multi_task,
        base_proj_hidden_dim = base_proj_hidden_dim,
        base_proj_dropout = base_proj_dropout,
        base_proj_output_dim = base_proj_output_dim,
        task_proj_dropout = task_proj_dropout,

        normalize_image_embeds=normalize_image_embeds,
        normalize_audio_embeds=normalize_audio_embeds,
        freeze_image_backbone=True,
        freeze_audio_backbone=True,
        device=device
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
    images = example_batch["image"].to(device)
    audios = example_batch["audio"].to(device)
    if verbose:
        print("Input images size: {}".format(tuple(images.size())))
        print("Input audio clips size: {}".format(tuple(audios.size())))
    
    # test forward pass:
    if multi_task:
        image_intra_embeds, image_cross_embeds, audio_intra_embeds, audio_cross_embeds = full_model.forward(images, audios)
        if verbose:
            print()
            print("image_intra_embeds size: {}".format(tuple(image_intra_embeds.size())))
            print("image_cross_embds size: {}".format(tuple(image_cross_embeds.size())))
            print("audio_intra_embeds size: {}".format(tuple(audio_intra_embeds.size())))
            print("audio_cross_embds size: {}".format(tuple(audio_cross_embeds.size())))
        assert tuple(image_intra_embeds.size()) == (batch_size * image_augment_params["n_views"], output_embed_dim), "Error with shape of image_intra_embeds."
        assert tuple(image_cross_embeds.size()) == (batch_size * image_augment_params["n_views"], output_embed_dim), "Error with shape of image_cross_embeds."
        assert tuple(audio_intra_embeds.size()) == (batch_size * audio_augment_params["n_views"], output_embed_dim), "Error with shape of audio_intra_embeds."
        assert tuple(audio_cross_embeds.size()) == (batch_size * audio_augment_params["n_views"], output_embed_dim), "Error with shape of audio_cross_embeds."
    
    else:
        image_embeds, audio_embeds = full_model.forward(images, audios)
        if verbose:
            print()
            print("Image embeddings size: {}".format(tuple(image_embeds.size())))
            print("Audio embeddings size: {}".format(tuple(audio_embeds.size())))
        assert tuple(image_embeds.size()) == (batch_size * image_augment_params["n_views"], output_embed_dim), "Error with shape of image embeddings."
        assert tuple(audio_embeds.size()) == (batch_size * audio_augment_params["n_views"], output_embed_dim), "Error with shape of audio embeddings."

    # test training_step() method:
    if verbose:
        print("\nTesting training_step() method...")
    train_loss = full_model.training_step(example_batch, 0)
    assert type(train_loss) == Tensor, "Error with return type."
    assert len(tuple(train_loss.size())) == 0, "Error with return shape."
    print("Training loss: {}".format(train_loss))

    # test validation_step() method:
    if verbose:
        print("\nTesting validation_step() method...")
    val_loss = full_model.validation_step(example_batch, 0)
    assert type(val_loss) == Tensor, "Error with return type."
    assert len(tuple(val_loss.size())) == 0, "Error with return shape."
    print("Validation loss: {}".format(val_loss))

    # test configure_optimizers() method:
    if verbose:
        print("\nTesting configure_optimizers() method...")
    optimizer = full_model.configure_optimizers()


    # test compute_image_embeds() method:
    if verbose:
        print()
        print("\n\nTesting compute_image_embeds() method...")
    if multi_task:
        image_intra_embeds, image_cross_embeds = full_model.compute_image_embeds(images)
        if verbose:
            print()
            print("image_intra_embeds size: {}".format(tuple(image_intra_embeds.size())))
            print("image_cross_embds size: {}".format(tuple(image_cross_embeds.size())))
        assert tuple(image_intra_embeds.size()) == (batch_size * image_augment_params["n_views"], output_embed_dim), "Error with shape of image_intra_embeds."
        assert tuple(image_cross_embeds.size()) == (batch_size * image_augment_params["n_views"], output_embed_dim), "Error with shape of image_cross_embeds."
    else:
        image_embeds = full_model.compute_image_embeds(images)
        if verbose:
            print()
            print("Image embeddings size: {}".format(tuple(image_embeds.size())))
        assert tuple(image_embeds.size()) == (batch_size * image_augment_params["n_views"], output_embed_dim), "Error with shape of image embeddings."
    
    # test compute_audio_embeds() method:
    if verbose:
        print("\n\nTesting compute_audio_embeds() method...")
    if multi_task:
        audio_intra_embeds, audio_cross_embeds = full_model.compute_audio_embeds(audios)
        if verbose:
            print()
            print("audio_intra_embeds size: {}".format(tuple(audio_intra_embeds.size())))
            print("audio_cross_embds size: {}".format(tuple(audio_cross_embeds.size())))
        assert tuple(audio_intra_embeds.size()) == (batch_size * audio_augment_params["n_views"], output_embed_dim), "Error with shape of audio_intra_embeds."
        assert tuple(audio_cross_embeds.size()) == (batch_size * audio_augment_params["n_views"], output_embed_dim), "Error with shape of audio_cross_embeds."
    else:
        audio_embeds = full_model.compute_audio_embeds(audios)
        if verbose:
            print()
            print("Audio embeddings size: {}".format(tuple(audio_embeds.size())))
        assert tuple(audio_embeds.size()) == (batch_size * audio_augment_params["n_views"], output_embed_dim), "Error with shape of audio embeddings."


    print("\n")

