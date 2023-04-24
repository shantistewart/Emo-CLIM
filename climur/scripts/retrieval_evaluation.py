"""Cross-modal retrieval evaluation script."""


import os
import argparse
import torch
import json
import clip
import laion_clap

from climur.utils.eval import get_image_embeddings, get_audio_embeddings
from climur.utils.retrieval import compute_retrieval_metrics
from climur.dataloaders.imac_images import IMACImages
from climur.dataloaders.audioset import AudioSetMood
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


# default manual mapping from image dataset emotion tags to audio dataset emotion tags:
IMAGE2AUDIO_TAG_MAP = {
    "excitement": "exciting",
    "contentment": "happy",
    "amusement": "funny",
    "anger": "angry",
    "fear": "scary",
    "sadness": "sad"
}

# default config file:
CONFIG_FILE = "configs/config_retrieval_eval.yaml"
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
    if dataset_configs["subset"] == "val":
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
        preprocess=image_preprocess_transform
    )
    # create audio dataset:
    audio_dataset = AudioSetMood(
        root=dataset_configs["audio_dataset_dir"],
        metadata_file_name=audio_dataset_metadata_file,
        clip_length_samples=audio_clip_length,
        sample_rate=dataset_configs["sample_rate"],
        eval=True,
        overlap_ratio=eval_configs["overlap_ratio"]
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


    # ---------
    # RETRIEVAL
    # ---------

    if verbose:
        print("\n\n\nRunning retrieval evaluations...\n")
    
    # automatically set results directory if not provided: results_dir = "results_subset/task_mode/audio_backbone_model_name/audio_backbone_mode/loss_weights_mode"
    if eval_configs["results_dir"] is None:
        if full_model_configs["multi_task"]:
            task_mode = "multi_task"
        else:
            task_mode = "single_task"
        
        if full_model_configs["freeze_audio_backbone"]:
            audio_backbone_mode = "frozen"
        else:
            audio_backbone_mode = "unfrozen"
        
        results_dir = os.path.join("results_{}".format(dataset_configs["subset"]), task_mode, audio_backbone_configs["model_name"], audio_backbone_mode, eval_configs["loss_weights_mode"])
    
    # else use provided results directory:
    else:
        results_dir = eval_configs["results_dir"]
    
    # create directories for saving results:
    os.makedirs(results_dir, exist_ok=False)
    os.makedirs(os.path.join(results_dir, "macro"), exist_ok=False)
    os.makedirs(os.path.join(results_dir, "per_class"), exist_ok=False)

    # map image dataset emotion tags to audio dataset emotion tags:
    image_labels = [IMAGE2AUDIO_TAG_MAP[tag] for tag in image_tags]
    audio_labels = audio_tags
    assert set(image_labels) == set(audio_labels), "Error mapping image dataset emotion tags to audio dataset emotion tags."


    # run image-to-music retrieval:
    if verbose:
        print("\nRunning image-to-music retrieval...")
    macro_metrics, metrics_per_class = compute_retrieval_metrics(
        query_embeds=image_cross_embeds,
        query_labels=image_labels,
        item_embeds=audio_cross_embeds,
        item_labels=audio_labels,
        metric_names=eval_configs["retrieval_metrics"],
        k_vals=eval_configs["k_vals"],
        device=device,
        mode="cross-modal"
    )
    # save to json files:
    with open(os.path.join(results_dir, "macro", "image2music_retrieval.json"), "w") as json_file:
        json.dump(macro_metrics, json_file, indent=3)
    with open(os.path.join(results_dir, "per_class", "image2music_retrieval.json"), "w") as json_file:
        json.dump(metrics_per_class, json_file, indent=3)
    
    # run music-to-image retrieval:
    if verbose:
        print("\nRunning music-to-image retrieval...")
    macro_metrics, metrics_per_class = compute_retrieval_metrics(
        query_embeds=audio_cross_embeds,
        query_labels=audio_labels,
        item_embeds=image_cross_embeds,
        item_labels=image_labels,
        metric_names=eval_configs["retrieval_metrics"],
        k_vals=eval_configs["k_vals"],
        device=device,
        mode="cross-modal"
    )
    # save to json files:
    with open(os.path.join(results_dir, "macro", "music2image_retrieval.json"), "w") as json_file:
        json.dump(macro_metrics, json_file, indent=3)
    with open(os.path.join(results_dir, "per_class", "music2image_retrieval.json"), "w") as json_file:
        json.dump(metrics_per_class, json_file, indent=3)


    # run image-to-image retrieval:
    if verbose:
        print("\nRunning image-to-image retrieval...")
    if full_model_configs["multi_task"]:
        macro_metrics, metrics_per_class = compute_retrieval_metrics(
            query_embeds=image_intra_embeds,
            query_labels=image_labels,
            item_embeds=image_intra_embeds,
            item_labels=image_labels,
            metric_names=eval_configs["retrieval_metrics"],
            k_vals=eval_configs["k_vals"],
            device=device,
            mode="intra-modal"
        )
    else:
        macro_metrics, metrics_per_class = compute_retrieval_metrics(
            query_embeds=image_cross_embeds,
            query_labels=image_labels,
            item_embeds=image_cross_embeds,
            item_labels=image_labels,
            metric_names=eval_configs["retrieval_metrics"],
            k_vals=eval_configs["k_vals"],
            device=device,
            mode="intra-modal"
        )
    # save to json files:
    with open(os.path.join(results_dir, "macro", "image2image_retrieval.json"), "w") as json_file:
        json.dump(macro_metrics, json_file, indent=3)
    with open(os.path.join(results_dir, "per_class", "image2image_retrieval.json"), "w") as json_file:
        json.dump(metrics_per_class, json_file, indent=3)
    
    # run music-to-music retrieval:
    if verbose:
        print("\nRunning music-to-music retrieval...")
    if full_model_configs["multi_task"]:
        macro_metrics, metrics_per_class = compute_retrieval_metrics(
            query_embeds=audio_intra_embeds,
            query_labels=audio_labels,
            item_embeds=audio_intra_embeds,
            item_labels=audio_labels,
            metric_names=eval_configs["retrieval_metrics"],
            k_vals=eval_configs["k_vals"],
            device=device,
            mode="intra-modal"
        )
    else:
        macro_metrics, metrics_per_class = compute_retrieval_metrics(
            query_embeds=audio_cross_embeds,
            query_labels=audio_labels,
            item_embeds=audio_cross_embeds,
            item_labels=audio_labels,
            metric_names=eval_configs["retrieval_metrics"],
            k_vals=eval_configs["k_vals"],
            device=device,
            mode="intra-modal"
        )
    # save to json files:
    with open(os.path.join(results_dir, "macro", "music2music_retrieval.json"), "w") as json_file:
        json.dump(macro_metrics, json_file, indent=3)
    with open(os.path.join(results_dir, "per_class", "music2music_retrieval.json"), "w") as json_file:
        json.dump(metrics_per_class, json_file, indent=3)


    print("\n\n")

