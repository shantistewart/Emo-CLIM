"""Training script for downstream supervised learning."""

import torch, clip
import argparse, pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from climur.dataloaders.mtat import MTAT
from climur.models.image_backbones import CLIPModel
from climur.models.audio_model_components import ShortChunkCNN_Res, HarmonicCNN
from climur.models.audio_backbones import (
    ShortChunkCNNEmbeddings,
    HarmonicCNNEmbeddings,
    CLAPEmbeddings,
    SHORTCHUNK_INPUT_LENGTH,
    HARMONIC_CNN_INPUT_LENGTH,
    CLAP_INPUT_LENGTH,
)
from climur.trainers.image2music import Image2Music
from climur.trainers.music_tagging import MTAT_Training
from climur.utils.misc import load_configs


# default config file:
CONFIG_FILE = "configs/config_mtat.yaml"
# script options:
verbose = True


if __name__ == "__main__":
    print("\n")

    # parse command-line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", nargs="?", const=CONFIG_FILE, default=CONFIG_FILE
    )
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
    image_backbone_configs = configs["image_backbone"]
    audio_backbone_configs = configs["audio_backbone"]
    full_model_configs = configs["full_model"]
    training_configs = configs["training"]
    logging_configs = configs["logging"]

    # get device:
    gpu_id = training_configs["gpu"]
    device = (
        torch.device(f"cuda:{gpu_id}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    # set random seed if selected:
    if dataset_configs["random_seed"]:
        pl.seed_everything(dataset_configs["random_seed"], workers=True)

    # -------
    # DATASET
    # -------

    train_dataset = MTAT(
        root=dataset_configs["dataset_dir"],
        download=False,
        subset=dataset_configs["subset"],
        sr=dataset_configs["sample_rate"],
        duration=HARMONIC_CNN_INPUT_LENGTH,
    )
    valid_dataset = MTAT(
        root=dataset_configs["dataset_dir"],
        download=False,
        subset="valid",
        sr=dataset_configs["sample_rate"],
        duration=HARMONIC_CNN_INPUT_LENGTH,
    )
    test_dataset = MTAT(
        root=dataset_configs["dataset_dir"],
        download=False,
        subset="test",
        sr=dataset_configs["sample_rate"],
        duration=HARMONIC_CNN_INPUT_LENGTH,
    )

    # ---------------
    # BACKBONE MODELS
    # ---------------

    if verbose:
        print("\nSetting up backbone model...")

    # load CLIP model:
    orig_clip_model, image_preprocess_transform = clip.load("ViT-B/32", device=device)
    # create CLIP wrapper model:
    image_backbone = CLIPModel(orig_clip_model)
    image_backbone.to(device)

    # set up audio backbone model:
    if audio_backbone_configs["model_name"] == "ShortChunk":
        # set audio input length:
        audio_clip_length = SHORTCHUNK_INPUT_LENGTH
        # load pretrained full Short-Chunk CNN ResNet model:
        full_audio_backbone = ShortChunkCNN_Res()
        # create wrapper model:
        sample_audio_input = torch.rand((2, audio_clip_length))
        audio_backbone = ShortChunkCNNEmbeddings(
            full_audio_backbone,
            sample_input=sample_audio_input,
            last_layer=audio_backbone_configs["last_layer_embed"],
            pool_type=audio_backbone_configs["pool_type"],
        )
    elif audio_backbone_configs["model_name"] == "HarmonicCNN":
        # set audio input length:
        audio_clip_length = HARMONIC_CNN_INPUT_LENGTH
        # load pretrained full Harmonic CNN model:
        full_audio_backbone = HarmonicCNN()
        # create wrapper model:
        sample_audio_input = torch.rand((2, audio_clip_length))
        audio_backbone = HarmonicCNNEmbeddings(
            full_audio_backbone,
            sample_input=sample_audio_input,
            last_layer=audio_backbone_configs["last_layer_embed"],
            pool_type=audio_backbone_configs["pool_type"],
        )
    elif audio_backbone_configs["model_name"] == "CLAP":
        # set audio input length:
        audio_clip_length = CLAP_INPUT_LENGTH
        # load pretrained full CLAP model:
        full_audio_backbone = laion_clap.CLAP_Module(
            enable_fusion=False, amodel="HTSAT-base"
        )

        full_audio_backbone.load_ckpt(audio_backbone_configs["pretrained_model_path"])

        # create wrapper model:
        sample_audio_input = torch.rand((2, audio_clip_length))
        audio_backbone = CLAPEmbeddings(
            full_audio_backbone,
            sample_input=sample_audio_input,
            last_layer=audio_backbone_configs["last_layer_embed"],
            pool_type=audio_backbone_configs["pool_type"],
        )
    else:
        raise ValueError(
            "{} model not supported".format(audio_backbone_configs["model_name"])
        )

    audio_backbone.to(device)

    # ------------
    # DATA LOADERS
    # ------------

    if verbose:
        print("\nSetting up datasets and data loaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_configs["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=training_configs["batch_size"],
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_configs["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    # -------------------
    # FULL MODEL & LOGGER
    # -------------------

    if verbose:
        print("\nSetting up full model and logger...")

    full_model = Image2Music.load_from_checkpoint(
        full_model_configs["checkpoint_path"],
        image_backbone=image_backbone,
        audio_backbone=audio_backbone,
        output_embed_dim=full_model_configs["output_embed_dim"],
        image_embed_dim=image_backbone_configs["embed_dim"],
        audio_embed_dim=audio_backbone_configs["embed_dim"],
        multi_task=full_model_configs["multi_task"],
        base_proj_hidden_dim=full_model_configs["base_proj_hidden_dim"],
        base_proj_dropout=full_model_configs["base_proj_dropout"],
        base_proj_output_dim=full_model_configs["base_proj_output_dim"],
        task_proj_dropout=full_model_configs["task_proj_dropout"],
        normalize_image_embeds=full_model_configs["normalize_image_embeds"],
        normalize_audio_embeds=full_model_configs["normalize_audio_embeds"],
        freeze_image_backbone=full_model_configs["freeze_image_backbone"],
        freeze_audio_backbone=full_model_configs["freeze_audio_backbone"],
        device=device,
    )
    full_model.to(device)
    full_model.eval()

    # create MTAT trainer:
    mtat_model = MTAT_Training(
        backbone=full_model,
        embed_dim=full_model_configs["output_embed_dim"],
        hparams=training_configs,
        num_classes=full_model_configs["n_classes"],
        device=device,
    )

    # create logger (logs are saved to /save_dir/name/version/):
    logger = TensorBoardLogger(
        save_dir=logging_configs["log_dir"],
        name=logging_configs["experiment_name"],
        version=logging_configs["experiment_version"],
    )

    # --------
    # TRAINING
    # --------

    # create trainer:
    model_ckpt_callback = ModelCheckpoint(
        monitor="validation/pr_auc", mode="max", save_top_k=1
    )
    early_stop_callback = EarlyStopping(
        monitor="validation/loss", mode="min", patience=10
    )
    trainer = Trainer(
        logger=logger,
        max_epochs=training_configs["max_epochs"],
        callbacks=[model_ckpt_callback, early_stop_callback],
        val_check_interval=training_configs["val_check_interval"],
        log_every_n_steps=logging_configs["log_every_n_steps"],
        # sync_batchnorm=True,     # TODO: Check if this is only required for multi-GPU training.
        accelerator="gpu",
        devices=[training_configs["gpu"]]
        # deterministic="warn",     # set when running training sessions for reproducibility
    )
    # train model:
    trainer.fit(
        mtat_model, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )

    # ----------
    # EVALUATION
    # ----------

    if verbose:
        print("\nEvaluating model... still TODO")

    print("\n\n")
