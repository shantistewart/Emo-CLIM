"""Utility functions for evaluation."""


import torch, tqdm
from torch.utils.data import Dataset
from typing import Union, List, Tuple, Any


def get_image_embeddings(
    model: Any, image_dataset: Dataset, image_dataset_tags: List, device: Any
) -> Union[Tuple[List, List, List], Tuple[List, List]]:
    """Extracts image embeddings and their emotion labels.

    Args:
        model (LightningModule): PyTorch Lightning LightningModule.
        image_dataset (PyTorch Dataset): Image dataset.
        image_dataset_tags (list): List of image dataset emotion tags to use.
        device (PyTorch device): PyTorch device.

    Returns:
        if model.multi_task == True:
            image_intra_embeds (list): Image embeddings in intra-modal image embedding space.
            image_cross_embeds (list): Image embeddings in (cross-modal) joint embedding space.
            image_tags (list): Image emotion tags.
        else:
            image_embeds (list): Image embeddings (in joint embedding space).
            image_tags (list): Image emotion tags.
    """

    # set model to eval mode if not already:
    model.eval()

    # extract image embeddings and emotion tags:     # TODO: Change to using dataloader instead of dataset class.
    if model.multi_task:
        image_intra_embeds = []
        image_cross_embeds = []
    else:
        image_embeds = []
    image_tags = []

    for idx in tqdm.tqdm(
        range(len(image_dataset)),
        total=len(image_dataset),
        desc="Extracting image embeddings and labels",
    ):
        # get image and emotion tag:
        image, tag = image_dataset[idx]
        # insert batch dimension:
        image = image.unsqueeze(dim=0)
        image = image.to(device)
        assert (
            len(tuple(image.size())) == 4 and image.size(dim=1) == 3
        ), "Error with image shape."

        # compute image embedding if emotion tag class was used during training:
        if tag in image_dataset_tags:
            if model.multi_task:
                with torch.no_grad():
                    intra_embed, cross_embed = model.compute_image_embeds(image)
                # remove batch dimension:
                intra_embed = intra_embed.squeeze(dim=0)
                cross_embed = cross_embed.squeeze(dim=0)

                assert tuple(intra_embed.size()) == (
                    model.output_embed_dim,
                ), "Error with shape of intra_embed."
                assert tuple(cross_embed.size()) == (
                    model.output_embed_dim,
                ), "Error with shape of cross_embed."
                image_intra_embeds.append(intra_embed)
                image_cross_embeds.append(cross_embed)

            else:
                with torch.no_grad():
                    embed = model.compute_image_embeds(image)
                # remove batch dimension:
                embed = embed.squeeze(dim=0)

                assert tuple(embed.size()) == (
                    model.output_embed_dim,
                ), "Error with shape of embed."
                image_embeds.append(embed)

            image_tags.append(tag)

    if model.multi_task:
        return image_intra_embeds, image_cross_embeds, image_tags
    else:
        return image_embeds, image_tags


def get_audio_embeddings(
    model: Any, audio_dataset: Dataset, audio_dataset_tags: List, device: Any
) -> Union[Tuple[List, List, List], Tuple[List, List]]:
    """Extracts audio embeddings and their emotion labels.

    Args:
        model (LightningModule): PyTorch Lightning LightningModule.
        audio_dataset (PyTorch Dataset): Audio dataset.
        audio_dataset_tags (list): List of audio dataset emotion tags to use.
        device (PyTorch device): PyTorch device.

    Returns:
        if model.multi_task == True:
            audio_intra_embeds (list): Audio embeddings in intra-modal audio embedding space.
            audio_cross_embeds (list): Audio embeddings in (cross-modal) joint embedding space.
            audio_tags (list): Audio emotion tags.
        else:
            audio_embeds (list): Audio embeddings (in joint embedding space).
            audio_tags (list): Audio emotion tags.
    """

    # verify that audio dataset is in evaluation model:
    assert audio_dataset.eval, "Audio dataset not in evaluation mode."

    # set model to eval mode if not already:
    model.eval()

    # extract audio embeddings and emotion tags:     # TODO: Change to using dataloader instead of dataset class.
    if model.multi_task:
        audio_intra_embeds = []
        audio_cross_embeds = []
    else:
        audio_embeds = []
    audio_tags = []

    for idx in tqdm.tqdm(
        range(len(audio_dataset)),
        total=len(audio_dataset),
        desc="Extracting audio embeddings and labels",
    ):
        # get chunked audio clip and emotion tag:
        audio_chunks, tag = audio_dataset[idx]
        audio_chunks = audio_chunks.to(device)
        assert (
            len(tuple(audio_chunks.size())) == 2
            and audio_chunks.size(dim=-1) == audio_dataset.clip_length
        ), "Error with audio shape."

        # compute audio embedding if emotion tag class was used during training:
        if tag in audio_dataset_tags:
            if model.multi_task:
                with torch.no_grad():
                    chunk_intra_embeds, chunk_cross_embeds = model.compute_audio_embeds(
                        audio_chunks
                    )
                # compute mean over chunks:
                intra_embed = chunk_intra_embeds.mean(dim=0)
                cross_embed = chunk_cross_embeds.mean(dim=0)

                assert tuple(intra_embed.size()) == (
                    model.output_embed_dim,
                ), "Error with shape of intra_embed."
                assert tuple(cross_embed.size()) == (
                    model.output_embed_dim,
                ), "Error with shape of cross_embed."
                audio_intra_embeds.append(intra_embed)
                audio_cross_embeds.append(cross_embed)

            else:
                with torch.no_grad():
                    chunk_embeds = model.compute_audio_embeds(audio_chunks)
                # compute mean over chunks:
                embed = chunk_embeds.mean(dim=0)

                assert tuple(embed.size()) == (
                    model.output_embed_dim,
                ), "Error with shape of embed."
                audio_embeds.append(embed)

            audio_tags.append(tag)

    if model.multi_task:
        return audio_intra_embeds, audio_cross_embeds, audio_tags
    else:
        return audio_embeds, audio_tags


def get_embedding_ds(model, audio_chunks):
    """Extracts audio embeddings from input audio dataset.

    Args:
        model (LightningModule): PyTorch Lightning LightningModule.
        audio_chunks (list): Batch of audio chunks.

    Returns:
        if model.multi_task == True:
            audio_intra_embeds (list): Audio embeddings in intra-modal audio embedding space.
            audio_cross_embeds (list): Audio embeddings in (cross-modal) joint embedding space.
        else:
            audio_embeds (list): Audio embeddings (in joint embedding space).
    """
    model.eval()

    if model.multi_task:
        with torch.no_grad():
            chunk_intra_embeds, chunk_cross_embeds = model.compute_audio_embeds(
                audio_chunks
            )
        return chunk_intra_embeds, chunk_cross_embeds

    else:
        with torch.no_grad():
            chunk_embeds = model.compute_audio_embeds(audio_chunks)
        return chunk_embeds
