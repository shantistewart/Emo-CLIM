"""Script for testing evaluation functions."""


import argparse
import torch
import numpy as np
import clip

from climur.utils.eval import get_image_embeddings, get_audio_embeddings
from climur.utils.retrieval import compute_retrieval_metrics
from climur.dataloaders.imac_images import IMACImages
from climur.dataloaders.audioset import AudioSetMood
from climur.models.image_backbones import CLIPModel
from climur.models.audio_model_components import ShortChunkCNN_Res, HarmonicCNN
from climur.models.audio_backbones import ShortChunkCNNEmbeddings, HarmonicCNNEmbeddings, SHORTCHUNK_INPUT_LENGTH, HARMONIC_CNN_INPUT_LENGTH
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
    image_backbone_configs = configs["image_backbone"]
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

    # set up audio backbone model:
    if audio_backbone_configs["model_name"] == "ShortChunk":
        # set audio input length:
        audio_clip_length = SHORTCHUNK_INPUT_LENGTH
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
        # set audio input length:
        audio_clip_length = HARMONIC_CNN_INPUT_LENGTH
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
        image_embed_dim=image_backbone_configs["embed_dim"],
        audio_embed_dim=audio_backbone_configs["embed_dim"],

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


    # -----
    # TESTS
    # -----

    # set image and audio dataset emotion tags:
    image_dataset_tags = list(IMAGE2AUDIO_TAG_MAP.keys())
    audio_dataset_tags = list(IMAGE2AUDIO_TAG_MAP.values())
    
    # test get_image_embeddings() function:
    if verbose:
        print()
        print("\n\nTesting get_image_embeddings() function...")
    
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
    
    # tests:
    if full_model_configs["multi_task"]:
        assert len(image_intra_embeds) == len(image_cross_embeds) and len(image_cross_embeds) == len(image_tags), "List lengths don't match."
        assert tuple(image_intra_embeds[0].size()) == (full_model.output_embed_dim, ), "Error with shape of image_intra_embeds tensors."
    else:
        assert len(image_cross_embeds) == len(image_tags), "List lengths don't match."
    assert tuple(image_cross_embeds[0].size()) == (full_model.output_embed_dim, ), "Error with shape of image_cross_embeds tensors."
    if verbose:
        print("Extracted embeddings of {} images.".format(len(image_tags)))
    

    # test get_audio_embeddings() function:
    if verbose:
        print("\n\nTesting get_audio_embeddings() function...")
    
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
    
    # tests:
    if full_model_configs["multi_task"]:
        assert len(audio_intra_embeds) == len(audio_cross_embeds) and len(audio_cross_embeds) == len(audio_tags), "List lengths don't match."
        assert tuple(audio_intra_embeds[0].size()) == (full_model.output_embed_dim, ), "Error with shape of audio_intra_embeds tensors."
    else:
        assert len(audio_cross_embeds) == len(audio_tags), "List lengths don't match."
    assert tuple(audio_cross_embeds[0].size()) == (full_model.output_embed_dim, ), "Error with shape of audio_cross_embeds tensors."
    if verbose:
        print("Extracted embeddings of {} audio clips.".format(len(audio_tags)))
    

    # test compute_retrieval_metrics() function:
    if verbose:
        print()
        print("\n\nTesting compute_retrieval_metrics() function...\n")
    
    # map image dataset emotion tags to audio dataset emotion tags:
    image_labels = [IMAGE2AUDIO_TAG_MAP[tag] for tag in image_tags]
    audio_labels = audio_tags
    assert set(image_labels) == set(audio_labels), "Error mapping image dataset emotion tags to audio dataset emotion tags."

    # test image-to-music retrieval:
    print()
    macro_metrics, metrics_per_class = compute_retrieval_metrics(
        query_embeds=image_cross_embeds,
        query_labels=image_labels,
        item_embeds=audio_cross_embeds,
        item_labels=audio_labels,
        metric_names=eval_configs["retrieval_metrics"],
        k_vals=eval_configs["k_vals"],
        device=device
    )
    print("\nImage-to-music retrieval metrics:")
    for metric_name, metrics in macro_metrics.items():
        print(f"macro {metric_name} @ k:")
        for k_str, value in metrics.items():
            print("\t{}: {:.2f}%".format(k_str, 100 * np.around(value, decimals=4)))
    
    # test music-to-image retrieval:
    print()
    macro_metrics, metrics_per_class = compute_retrieval_metrics(
        query_embeds=audio_cross_embeds,
        query_labels=audio_labels,
        item_embeds=image_cross_embeds,
        item_labels=image_labels,
        metric_names=eval_configs["retrieval_metrics"],
        k_vals=eval_configs["k_vals"],
        device=device
    )
    print("\nMusic-to-image retrieval metrics:")
    for metric_name, metrics in macro_metrics.items():
        print(f"macro {metric_name} @ k:")
        for k_str, value in metrics.items():
            print("\t{}: {:.2f}%".format(k_str, 100 * np.around(value, decimals=4)))


    print("\n\n")

