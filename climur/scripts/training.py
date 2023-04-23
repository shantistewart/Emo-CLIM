"""Training script for image-music supervised contrastive learning."""


import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.utils.data import DataLoader
import clip
import laion_clap

from climur.dataloaders.imac_images import IMACImages
from climur.dataloaders.audioset import AudioSetMood
from climur.dataloaders.multimodal import Multimodal
from climur.models.image_backbones import CLIPModel, CLIP_EMBED_DIM
from climur.models.audio_model_components import ShortChunkCNN_Res, HarmonicCNN
from climur.models.audio_backbones import (
    ShortChunkCNNEmbeddings,
    HarmonicCNNEmbeddings,
    CLAPEmbeddings,
    SHORTCHUNK_INPUT_LENGTH,
    HARMONIC_CNN_INPUT_LENGTH,
    CLAP_INPUT_LENGTH,
    SHORTCHUNK_EMBED_DIM,
    HARMONIC_CNN_EMBED_DIM,
    CLAP_EMBED_DIM
)
from climur.trainers.image2music import Image2Music
from climur.utils.misc import load_configs


# default config file:
CONFIG_FILE = "configs/config_train.yaml"
# script options:
verbose = True


if __name__ == "__main__":
    print("\n")

    # parse command-line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", nargs="?", const=CONFIG_FILE, default=CONFIG_FILE)
    args = parser.parse_args()

    # set PyTorch warnings:
    # torch.set_warn_always(False)


    # -------
    # CONFIGS
    # -------

    # load configs:
    configs = load_configs(args.config_file)
    # unpack configs:
    dataset_configs = configs["dataset"]
    audio_backbone_configs = configs["audio_backbone"]
    full_model_configs = configs["full_model"]
    training_configs = configs["training"]
    logging_configs = configs["logging"]

    # get device:
    gpu_id = training_configs["gpu"]
    device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
    # set random seed if selected:     # TODO: Double-check that this does not mess up randomness of dataloaders.
    if dataset_configs["random_seed"]:
        pl.seed_everything(dataset_configs["random_seed"], workers=True)


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
    # set output embedding dimension:
    image_embed_dim = CLIP_EMBED_DIM

    # set up audio backbone model:
    if audio_backbone_configs["model_name"] == "ShortChunk":
        # set audio input length and output embedding dimension:
        audio_clip_length = SHORTCHUNK_INPUT_LENGTH
        audio_embed_dim = SHORTCHUNK_EMBED_DIM
        # load pretrained full Short-Chunk CNN ResNet model:
        full_audio_backbone = ShortChunkCNN_Res()
        full_audio_backbone.load_state_dict(
            torch.load(
                audio_backbone_configs["pretrained_model_paths"][audio_backbone_configs["model_name"]], map_location=torch.device("cpu")
            )
        )
        # create wrapper model:
        sample_audio_input = torch.rand((2, audio_clip_length))
        audio_backbone = ShortChunkCNNEmbeddings(
            full_audio_backbone,
            sample_input=sample_audio_input,
            last_layer=audio_backbone_configs["last_layer_embed"],
            pool_type=audio_backbone_configs["pool_type"],
        )

    elif audio_backbone_configs["model_name"] == "HarmonicCNN":
        # set audio input length and output embedding dimension:
        audio_clip_length = HARMONIC_CNN_INPUT_LENGTH
        audio_embed_dim = HARMONIC_CNN_EMBED_DIM
        # load pretrained full Harmonic CNN model:
        full_audio_backbone = HarmonicCNN()
        full_audio_backbone.load_state_dict(
            torch.load(
                audio_backbone_configs["pretrained_model_paths"][audio_backbone_configs["model_name"]], map_location=torch.device("cpu")
            )
        )
        # create wrapper model:
        sample_audio_input = torch.rand((2, audio_clip_length))
        audio_backbone = HarmonicCNNEmbeddings(
            full_audio_backbone,
            sample_input=sample_audio_input,
            last_layer=audio_backbone_configs["last_layer_embed"],
            pool_type=audio_backbone_configs["pool_type"],
        )
    
    elif audio_backbone_configs["model_name"] == "CLAP":
        # set audio input length and output embedding dimension:
        audio_clip_length = CLAP_INPUT_LENGTH
        audio_embed_dim = CLAP_EMBED_DIM
        # load pretrained full CLAP model:
        full_audio_backbone = laion_clap.CLAP_Module(
            enable_fusion=False, amodel='HTSAT-base'
        )
        full_audio_backbone.load_ckpt(
           audio_backbone_configs["pretrained_model_paths"][audio_backbone_configs["model_name"]]
        )
        # create wrapper model:
        sample_audio_input = torch.rand((2, audio_clip_length))
        audio_backbone = CLAPEmbeddings(
            full_audio_backbone,
            sample_input=sample_audio_input,
            last_layer=audio_backbone_configs["last_layer_embed"],
            pool_type=audio_backbone_configs["pool_type"],
        )
    
    else:
        raise ValueError("{} model not supported".format(audio_backbone_configs["model_name"]))
    
    audio_backbone.to(device)


    # ------------
    # DATA LOADERS
    # ------------

    if verbose:
        print("\nSetting up datasets and data loaders...")

    # create image datasets:
    image_train_dataset = IMACImages(
        root=dataset_configs["image_dataset_dir"],
        metadata_file_name="metadata_train.csv",
        preprocess=image_preprocess_transform,
    )
    image_val_dataset = IMACImages(
        root=dataset_configs["image_dataset_dir"],
        metadata_file_name="metadata_val.csv",
        preprocess=image_preprocess_transform,
    )

    # create audio datasets:
    audio_train_dataset = AudioSetMood(
        root=dataset_configs["audio_dataset_dir"],
        metadata_file_name="new_split_metadata_files/metadata_train.csv",
        clip_length_samples=audio_clip_length,
        sample_rate=dataset_configs["sample_rate"],
        audio_model=audio_backbone_configs["model_name"]
    )
    audio_val_dataset = AudioSetMood(
        root=dataset_configs["audio_dataset_dir"],
        metadata_file_name="new_split_metadata_files/metadata_val.csv",
        clip_length_samples=audio_clip_length,
        sample_rate=dataset_configs["sample_rate"],
        audio_model=audio_backbone_configs["model_name"]
    )

    # create multimodal datasets:
    multimodal_train_dataset = Multimodal(
        image_dataset=image_train_dataset,
        audio_dataset=audio_train_dataset,
        length=dataset_configs["train_n_batches"] * training_configs["batch_size"],
    )
    multimodal_val_dataset = Multimodal(
        image_dataset=image_val_dataset,
        audio_dataset=audio_val_dataset,
        length=dataset_configs["val_n_batches"] * training_configs["batch_size"],
    )
    if verbose:
        print(
            "For training: using {} batches per epoch, sampling from {} images and {} audio clips.".format(
                dataset_configs["train_n_batches"],
                multimodal_train_dataset.image_dataset_len,
                multimodal_train_dataset.audio_dataset_len,
            )
        )
        print(
            "For validation: using {} batches, sampling from {} images and {} audio clips.".format(
                dataset_configs["val_n_batches"],
                multimodal_val_dataset.image_dataset_len,
                multimodal_val_dataset.audio_dataset_len,
            )
        )

    # create dataloaders:
    train_loader = DataLoader(
        multimodal_train_dataset,
        batch_size=training_configs["batch_size"],
        shuffle=True,
        num_workers=training_configs["n_workers"],
        drop_last=True,
    )
    assert (
        len(train_loader) == dataset_configs["train_n_batches"]
    ), "Length of train_loader is incorrect."
    val_loader = DataLoader(
        multimodal_val_dataset,
        batch_size=training_configs["batch_size"],
        shuffle=False,
        num_workers=training_configs["n_workers"],
        drop_last=True,
    )
    assert len(val_loader) == dataset_configs["val_n_batches"], "Length of val_loader is incorrect."


    # -------------------
    # FULL MODEL & LOGGER
    # -------------------

    if verbose:
        print("\nSetting up full model and logger...")
    
    full_model = Image2Music(
        image_backbone=image_backbone,
        audio_backbone=audio_backbone,
        output_embed_dim=full_model_configs["output_embed_dim"],
        image_embed_dim=image_embed_dim,
        audio_embed_dim=audio_embed_dim,
        hparams=training_configs,  # TODO: Maybe change to include all configs (need to modify Image2Music class).

        multi_task = full_model_configs["multi_task"],
        base_proj_hidden_dim = full_model_configs["base_proj_hidden_dim"],
        base_proj_dropout = full_model_configs["base_proj_dropout"],
        base_proj_output_dim = full_model_configs["base_proj_output_dim"],
        task_proj_dropout = full_model_configs["task_proj_dropout"],

        normalize_image_embeds=full_model_configs["normalize_image_embeds"],
        normalize_audio_embeds=full_model_configs["normalize_audio_embeds"],
        freeze_image_backbone=full_model_configs["freeze_image_backbone"],
        freeze_audio_backbone=full_model_configs["freeze_audio_backbone"],
        device=device,
    )

    # automatically set up experiment name if not provided: experiment_name = "task_mode/audio_backbone_model_name/audio_backbone_mode/loss_weights_mode"
    if logging_configs["experiment_name"] is None:
        if full_model_configs["multi_task"]:
            task_mode = "multi_task"
        else:
            task_mode = "single_task"
        
        if full_model_configs["freeze_audio_backbone"]:
            audio_backbone_mode = "frozen"
        else:
            audio_backbone_mode = "unfrozen"
        
        if training_configs["loss_weights"]["image2image"] > 0.0 and training_configs["loss_weights"]["audio2audio"] > 0.0:
            if training_configs["loss_weights"]["image2audio"] > 0.0 and training_configs["loss_weights"]["audio2image"] > 0.0:
                loss_weights_mode = "all_losses"
            else:
                loss_weights_mode = "intra_losses_only"
        elif training_configs["loss_weights"]["image2audio"] > 0.0 and training_configs["loss_weights"]["audio2image"] > 0.0:
            loss_weights_mode = "cross_losses_only"
        else:
            raise ValueError("All loss weights can't be 0.")
        
        experiment_name = os.path.join(task_mode, audio_backbone_configs["model_name"], audio_backbone_mode, loss_weights_mode)
    
    # else use provided experiment name:
    else:
        experiment_name = logging_configs["experiment_name"]
    
    # create logger (logs are saved to /save_dir/name/version/):
    logger = TensorBoardLogger(
        save_dir=logging_configs["log_dir"],
        name=experiment_name,
        version=logging_configs["experiment_version"],
    )


    # --------
    # TRAINING
    # --------

    # create trainer:
    model_ckpt_callback = ModelCheckpoint(monitor="validation/total_loss", mode="min", save_top_k=1)
    trainer = Trainer(
        logger=logger,
        max_epochs=training_configs["max_epochs"],
        callbacks=[model_ckpt_callback],
        val_check_interval=training_configs["val_check_interval"],
        log_every_n_steps=logging_configs["log_every_n_steps"],
        # sync_batchnorm=True,     # TODO: Check if this is only required for multi-GPU training.
        accelerator="gpu",
        devices=[training_configs["gpu"]]
        # deterministic="warn",     # set when running training sessions for reproducibility
    )

    # train model:
    trainer.fit(full_model, train_dataloaders=train_loader, val_dataloaders=val_loader)


    print("\n\n")

