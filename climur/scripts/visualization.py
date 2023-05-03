"""Script to visualize embeddings with t-SNE."""


import os
import argparse
import torch
import numpy as np
import clip
try:
    import laion_clap
except:
    pass

from climur.utils.eval import get_image_embeddings, get_audio_embeddings
from climur.utils.visualize import visualize_embeds
from climur.dataloaders.imac_images import IMACImages
from climur.dataloaders.audioset import AudioSetMood
from climur.models.image_backbones import CLIPModel
from climur.models.audio_model_components import ShortChunkCNN_Res, HarmonicCNN
from climur.models.audio_backbones import ShortChunkCNNEmbeddings, HarmonicCNNEmbeddings, SampleCNNEmbeddings, CLAPEmbeddings
from climur.trainers.image2music import Image2Music
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
    SAMPLE_CNN_DEFAULT_PARAMS
)


# default config file:
CONFIG_FILE = "configs/config_visualize.yaml"
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
    eval_configs = configs["eval"]

    # get device:
    gpu_id = eval_configs["gpu"]
    device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")


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
        audio_backbone = SampleCNNEmbeddings(
            params=SAMPLE_CNN_DEFAULT_PARAMS
        )
    
    elif audio_backbone_configs["model_name"] == "CLAP":
        # set audio input length and output embedding dimension:
        audio_clip_length = CLAP_INPUT_LENGTH
        audio_embed_dim = CLAP_EMBED_DIM
        # load pretrained full CLAP model:
        full_audio_backbone = laion_clap.CLAP_Module(
            enable_fusion=False, amodel='HTSAT-base'
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


    # --------
    # DATASETS
    # --------

    if verbose:
        print("\nSetting up datasets...")
    
    # set up metadata file names:
    if dataset_configs["subset"] == "train":
        image_dataset_metadata_file = "metadata_train.csv"
        audio_dataset_metadata_file = "new_split_metadata_files/metadata_train.csv"
    elif dataset_configs["subset"] == "val":
        image_dataset_metadata_file = "metadata_val.csv"
        audio_dataset_metadata_file = "new_split_metadata_files/metadata_val.csv"
    elif dataset_configs["subset"] == "test":
        image_dataset_metadata_file = "metadata_test.csv"
        audio_dataset_metadata_file = "new_split_metadata_files/metadata_test.csv"
    else:
        raise ValueError("Invalid subset.")
    
    # create image dataset:
    image_dataset = IMACImages(
        root=dataset_configs["image_dataset_dir"],
        metadata_file_name=image_dataset_metadata_file,
        augment_params=None,
        eval=True,
        preprocess=image_preprocess_transform
    )
    # create audio dataset:
    audio_dataset = AudioSetMood(
        root=dataset_configs["audio_dataset_dir"],
        metadata_file_name=audio_dataset_metadata_file,
        clip_length_samples=audio_clip_length,
        sample_rate=dataset_configs["sample_rate"],
        augment_params=None,
        eval=True,
        overlap_ratio=eval_configs["overlap_ratio"],
        audio_model=audio_backbone_configs["model_name"]
    )


    # ----------
    # FULL MODEL
    # ----------

    if verbose:
        print("\nLoading pretrained full model...")
    
    full_model = Image2Music.load_from_checkpoint(
        full_model_configs["checkpoint_path"],

        image_backbone=image_backbone,
        audio_backbone=audio_backbone,
        output_embed_dim=full_model_configs["output_embed_dim"],
        image_embed_dim=image_embed_dim,
        audio_embed_dim=audio_embed_dim,

        multi_task = full_model_configs["multi_task"],
        base_proj_hidden_dim = full_model_configs["base_proj_hidden_dim"],
        base_proj_dropout = full_model_configs["base_proj_dropout"],     # TODO: Probably can remove this.
        base_proj_output_dim = full_model_configs["base_proj_output_dim"],
        task_proj_dropout = full_model_configs["task_proj_dropout"],     # TODO: Probably can remove this.

        normalize_image_embeds=full_model_configs["normalize_image_embeds"],
        normalize_audio_embeds=full_model_configs["normalize_audio_embeds"],
        freeze_image_backbone=full_model_configs["freeze_image_backbone"],     # TODO: Probably can just set to True.
        freeze_audio_backbone=full_model_configs["freeze_audio_backbone"],     # TODO: Probably can just set to True.
        device=device
    )
    full_model.to(device)
    # set to eval mode:
    full_model.eval()


    # ----------
    # EMBEDDINGS
    # ----------

    # set image and audio dataset emotion tags:
    image_dataset_tags = list(IMAGE2AUDIO_TAG_MAP.keys())
    audio_dataset_tags = list(IMAGE2AUDIO_TAG_MAP.values())

    # extract image embeddings and emotion tags:
    if verbose:
        print("\n\n")
    if full_model_configs["multi_task"]:
        image_intra_embeds, image_cross_embeds, image_tags = get_image_embeddings(
            model=full_model,
            image_dataset=image_dataset,
            image_dataset_tags=image_dataset_tags,
            device=device
        )
    else:
        image_cross_embeds, image_tags = get_image_embeddings(
            model=full_model,
            image_dataset=image_dataset,
            image_dataset_tags=image_dataset_tags,
            device=device
        )
    if verbose:
        print("Extracted embeddings of {} images.".format(len(image_tags)))
    
    # extract audio embeddings and emotion tags:
    if verbose:
        print("\n")
    if full_model_configs["multi_task"]:
        audio_intra_embeds, audio_cross_embeds, audio_tags = get_audio_embeddings(
            model=full_model,
            audio_dataset=audio_dataset,
            audio_dataset_tags=audio_dataset_tags,
            device=device
        )
    else:
        audio_cross_embeds, audio_tags = get_audio_embeddings(
            model=full_model,
            audio_dataset=audio_dataset,
            audio_dataset_tags=audio_dataset_tags,
            device=device
        )
    if verbose:
        print("Extracted embeddings of {} audio clips.".format(len(audio_tags)))


    # -------------
    # VISUALIZATION
    # -------------

    if verbose:
        print("\n\n\nCreating visualizations...\n")
    
    # automatically set plots directory if not provided: plots_dir = "plots/subset/task_mode/audio_backbone_model_name/audio_backbone_mode/loss_weights_mode"
    if eval_configs["plots_dir"] is None:
        if full_model_configs["multi_task"]:
            task_mode = "multi_task"
        else:
            task_mode = "single_task"
        
        if full_model_configs["freeze_audio_backbone"]:
            audio_backbone_mode = "frozen"
        else:
            audio_backbone_mode = "unfrozen"
        
        plots_dir = os.path.join("plots", dataset_configs["subset"], task_mode, audio_backbone_configs["model_name"], audio_backbone_mode, eval_configs["loss_weights_mode"])
    
    # else use provided plots directory:
    else:
        plots_dir = eval_configs["plots_dir"]
    
    # create directories for saving plots:
    os.makedirs(plots_dir, exist_ok=True)

    # map image dataset emotion tags to audio dataset emotion tags:
    image_labels = [IMAGE2AUDIO_TAG_MAP[tag] for tag in image_tags]
    audio_labels = audio_tags
    assert set(image_labels) == set(audio_labels), "Error mapping image dataset emotion tags to audio dataset emotion tags."

    # convert image lists to numpy arrays:
    if full_model_configs["multi_task"]:
        image_intra_embeds = torch.stack(image_intra_embeds, dim=0).cpu().numpy()
        assert image_intra_embeds.shape == (len(image_labels), full_model.output_embed_dim), "Error converting list to numpy array."
    image_cross_embeds = torch.stack(image_cross_embeds, dim=0).cpu().numpy()
    assert image_cross_embeds.shape == (len(image_labels), full_model.output_embed_dim), "Error converting list to numpy array."
    image_labels = np.asarray(image_labels, dtype=object)
    assert image_labels.shape == (image_cross_embeds.shape[0], ), "Error converting list to numpy array."

    # convert audio lists to numpy arrays:
    if full_model_configs["multi_task"]:
        audio_intra_embeds = torch.stack(audio_intra_embeds, dim=0).cpu().numpy()
        assert audio_intra_embeds.shape == (len(audio_labels), full_model.output_embed_dim), "Error converting list to numpy array."
    audio_cross_embeds = torch.stack(audio_cross_embeds, dim=0).cpu().numpy()
    assert audio_cross_embeds.shape == (len(audio_labels), full_model.output_embed_dim), "Error converting list to numpy array."
    audio_labels = np.asarray(audio_labels, dtype=object)
    assert audio_labels.shape == (audio_cross_embeds.shape[0], ), "Error converting list to numpy array."

    # create dictionary mapping emotion labels to colors:
    emotion_labels = audio_dataset_tags
    colors = ["red", "purple", "blue", "green", "orange", "black"]     # arbitrary colors
    label2color = {}
    for i in range(len(emotion_labels)):
        label2color[emotion_labels[i]] = colors[i]
    
    # visualize image and audio embeddings together:
    if verbose:
        print("\nVisualizing image and audio embeddings together...")
    visualize_embeds(
        image_embeds=image_cross_embeds,
        image_labels=image_labels,
        audio_embeds=audio_cross_embeds,
        audio_labels=audio_labels,
        label2color=label2color,
        save_path=os.path.join(plots_dir, "image_audio_tsne.png")
    )


    print("\n\n")

