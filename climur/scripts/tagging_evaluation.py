"""Cross-modal retrieval evaluation script."""


import os
import argparse
import torch
import json
import clip
from torch.utils.data import DataLoader
from climur.utils.eval import evaluate

try:
    import laion_clap
except:
    pass

from climur.utils.eval import get_image_embeddings, get_audio_embeddings
from climur.utils.retrieval import compute_retrieval_metrics
from climur.dataloaders.mtat import MTAT
from climur.models.image_backbones import CLIPModel
from climur.models.audio_model_components import ShortChunkCNN_Res, HarmonicCNN
from climur.models.audio_backbones import (
    ShortChunkCNNEmbeddings,
    HarmonicCNNEmbeddings,
    SampleCNNEmbeddings,
    CLAPEmbeddings,
)
from climur.trainers.image2music import Image2Music
from climur.trainers.music_tagging import MTAT_Training
from climur.utils.misc import load_configs
from climur.utils.constants import (
    IMAGE2AUDIO_TAG_MAP,
    SHORTCHUNK_INPUT_LENGTH,
    HARMONIC_CNN_INPUT_LENGTH,
    SAMPLE_CNN_INPUT_LENGTH,
    CLAP_INPUT_LENGTH,
    CLIP_EMBED_DIM,
    SHORTCHUNK_EMBED_DIM,
    HARMONIC_CNN_EMBED_DIM,
    SAMPLE_CNN_EMBED_DIM,
    CLAP_EMBED_DIM,
    SAMPLE_CNN_DEFAULT_PARAMS,
)


# default config file:
CONFIG_FILE = "configs/config_mtat_eval.yaml"
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
    training_configs = configs["training"]
    audio_backbone_configs = configs["audio_backbone"]
    full_model_configs = configs["full_model"]
    eval_configs = configs["eval"]

    # get device:
    gpu_id = eval_configs["gpu"]
    device = (
        torch.device(f"cuda:{gpu_id}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

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
        # create full Short-Chunk CNN ResNet model:
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
        # set audio input length and output embedding dimension:
        audio_clip_length = HARMONIC_CNN_INPUT_LENGTH
        audio_embed_dim = HARMONIC_CNN_EMBED_DIM
        # create full Harmonic CNN model:
        full_audio_backbone = HarmonicCNN()
        # create wrapper model:
        sample_audio_input = torch.rand((2, audio_clip_length))
        audio_backbone = HarmonicCNNEmbeddings(
            full_audio_backbone,
            sample_input=sample_audio_input,
            last_layer=audio_backbone_configs["last_layer_embed"],
            pool_type=audio_backbone_configs["pool_type"],
        )

    elif audio_backbone_configs["model_name"] == "SampleCNN":
        # set audio input length and output embedding dimension:
        audio_clip_length = SAMPLE_CNN_INPUT_LENGTH
        audio_embed_dim = SAMPLE_CNN_EMBED_DIM
        # create (empty) SampleCNNEmbeddings model:
        audio_backbone = SampleCNNEmbeddings(params=SAMPLE_CNN_DEFAULT_PARAMS)

    elif audio_backbone_configs["model_name"] == "CLAP":
        # set audio input length and output embedding dimension:
        audio_clip_length = CLAP_INPUT_LENGTH
        audio_embed_dim = CLAP_EMBED_DIM
        # load pretrained full CLAP model:
        full_audio_backbone = laion_clap.CLAP_Module(
            enable_fusion=False, amodel="HTSAT-base"
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
        raise ValueError(
            "{} model not supported".format(audio_backbone_configs["model_name"])
        )

    audio_backbone.to(device)

    # --------
    # DATASETS
    # --------

    if verbose:
        print("\nSetting up datasets...")

    test_dataset = MTAT(
        root=dataset_configs["dataset_dir"],
        download=False,
        subset="test",
        sr=dataset_configs["sample_rate"],
        duration=audio_clip_length,
    )

    # ----------
    # FULL MODEL
    # ----------

    if verbose:
        print("\nLoading pretrained full model...")

    full_model = Image2Music(
        image_backbone=image_backbone,
        audio_backbone=audio_backbone,
        output_embed_dim=full_model_configs["output_embed_dim"],
        image_embed_dim=image_embed_dim,
        audio_embed_dim=audio_embed_dim,
        multi_task=full_model_configs["multi_task"],
        base_proj_hidden_dim=full_model_configs["base_proj_hidden_dim"],
        base_proj_dropout=full_model_configs[
            "base_proj_dropout"
        ],  # TODO: Probably can remove this.
        base_proj_output_dim=full_model_configs["base_proj_output_dim"],
        task_proj_dropout=full_model_configs[
            "task_proj_dropout"
        ],  # TODO: Probably can remove this.
        normalize_image_embeds=full_model_configs["normalize_image_embeds"],
        normalize_audio_embeds=full_model_configs["normalize_audio_embeds"],
        freeze_image_backbone=full_model_configs[
            "freeze_image_backbone"
        ],  # TODO: Probably can just set to True.
        freeze_audio_backbone=full_model_configs[
            "freeze_audio_backbone"
        ],  # TODO: Probably can just set to True.
        hparams=training_configs,
        device=device,
    )
    full_model.to(device)
    full_model.eval()

    # retrieve MTAT trained model:
    mtat_model = MTAT_Training.load_from_checkpoint(
        full_model_configs["checkpoint_path"],
        backbone=full_model,
        embed_dim=full_model_configs["output_embed_dim"],
        # hparams=training_configs,
        num_classes=full_model_configs["n_classes"],
        device=device,
    )

    # ----------
    # EVALUATION
    # ----------

    audio_metrics = evaluate(
        mtat_model,
        dataset=test_dataset,
        dataset_name=dataset_configs["dataset_name"],
        audio_length=audio_clip_length,
        device=device,
    )
