"""Utility functions for evaluation."""

import os, torch, numpy as np, json
import tqdm, matplotlib.pyplot as plt
from torch.utils.data import Dataset
from typing import Union, List, Tuple, Any
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn import metrics
import pandas as pd


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


def visualize(features_au, labels_au, features_im=None, labels_im=None, name="temp"):
    """
    Visualizes the given features and labels using t-SNE.
    """
    # concatenate image and audio features:
    if features_im is not None:
        features = np.concatenate((features_im, features_au), axis=0)
        labels = np.concatenate((labels_im, labels_au), axis=0)
    else:
        features = features_au
        labels = labels_au

    tsne = TSNE(
        n_components=2,
        perplexity=5,
        learning_rate=130,
        metric="cosine",
        square_distances=True,
    ).fit_transform(features)

    # normalize t-SNE output:
    tx = MinMaxScaler().fit_transform(tsne[:, 0].reshape(-1, 1))[:, 0]
    ty = MinMaxScaler().fit_transform(tsne[:, 1].reshape(-1, 1))[:, 0]

    fig = plt.figure()
    plt.rcParams["font.size"] = 10
    ax = fig.add_subplot(111)

    # plot the t-SNE for a single class occurence
    labels = ["class" if l[0] else "no_class" for l in labels]
    list_labels = list(set(labels))
    colors = ["red", "purple", "blue", "green", "orange", "black"]
    colors = {emotion: color for emotion, color in zip(list_labels, colors)}

    for label in list_labels:
        # find the samples of this class
        indices = [i for (i, l) in enumerate(labels) if l == label]
        # we assume features = [audio_features, image_features]
        if features_im is not None:
            ln = int(len(indices) / 2)
            # audio points
            curr_tx, curr_ty = np.take(tx, indices[:ln]), np.take(ty, indices[:ln])
            ax.scatter(
                curr_tx,
                curr_ty,
                c=colors[label],
                marker=".",
                label=label,
            )
            # image points
            curr_tx, curr_ty = np.take(tx, indices[ln:]), np.take(ty, indices[ln:])
            ax.scatter(
                curr_tx,
                curr_ty,
                c=colors[label],
                marker="+",
                label=label,
            )
        else:
            curr_tx, curr_ty = np.take(tx, indices), np.take(ty, indices)
            ax.scatter(
                curr_tx,
                curr_ty,
                c=colors[label],
                marker=".",
                label=label,
            )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best")

    os.makedirs("tsne/", exist_ok=True)
    fig.savefig("tsne/" + name)
    plt.close()


def evaluate(
    model: Any,
    dataset: Any,
    dataset_name: str,
    audio_length: int,
    device: torch.device = None,
    output_dir: str = "results/",
    tsne_name: str = "tsne_mtat_binary.png",
):
    """Performs evaluation of supervised models on music tagging"""
    os.makedirs(output_dir, exist_ok=True)

    # run inference:
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        # create wrapper dataset for splitting songs:
        test_dataset = dataset
        features, y_pred, y_true = [], [], []

        # run inference:
        for idx in tqdm.tqdm(range(len(test_dataset))):
            # get audio of song (split into segments of length audio_length) and label:
            audio_song, label = test_dataset[idx]
            assert (
                audio_song.size(dim=-1) == audio_length
            ), "Error with shape of song audio."
            audio_song = audio_song.to(device)

            # pass song through model to get features (embeddings) and outputs (logits) of each segment:
            feat = get_embedding_ds(model.backbone, audio_song)
            output = model(audio_song)
            output = torch.sigmoid(output)

            # average segment features across song = song-level feature:
            feat_song = feat.mean(dim=0)
            assert feat_song.dim() == 1 and feat_song.size(dim=0) == feat.size(
                dim=-1
            ), "Error with shape of song-level feature."

            # save true label and song-level feature:
            y_true.append(label)
            features.append(feat_song)
            pred_song = torch.mean(output, dim=0)

            # sanity check shape:
            assert pred_song.dim() == 1 and pred_song.size(dim=0) == output.size(
                dim=-1
            ), "Error with shape of song-level output."

            # save predicted label (song-level output):
            y_pred.append(pred_song)

    # convert lists to numpy arrays:
    y_true = torch.stack(y_true, dim=0).cpu().numpy()
    y_pred = torch.stack(y_pred, dim=0).cpu().numpy()
    features = torch.stack(features, dim=0).cpu().numpy()

    # save true labels and song-level features:
    np.save(os.path.join(f"{output_dir}/labels.npy"), y_true)
    np.save(os.path.join(f"{output_dir}/features.npy"), features)

    visualize(features, y_true, name=tsne_name)

    # compute performance metrics:
    if dataset_name in ["magnatagatune", "mtg-jamendo-dataset"]:
        # convert tag names to tag indices:
        tag_indices = [dataset.label2idx[tag] for tag in dataset.label_list]
        try:
            global_roc = metrics.roc_auc_score(
                y_true[:, tag_indices],
                y_pred[:, tag_indices],
                average="macro",
            )
            global_precision = metrics.average_precision_score(
                y_true[:, tag_indices],
                y_pred[:, tag_indices],
                average="macro",
            )
        except:
            print("Warning: at least 1 global metric was not able to be computed.")
            global_roc = np.nan
            global_precision = np.nan

        # save to json file (rounded to 4 decimal places):
        global_metrics_dict = {
            "ROC-AUC": np.around(global_roc, decimals=4),
            "PR-AUC": np.around(global_precision, decimals=4),
        }
        with open(f"{output_dir}/global_metrics.json", "w") as json_file:
            json.dump(global_metrics_dict, json_file, indent=3)

        # compute tag-wise metrics:
        try:
            tag_roc = metrics.roc_auc_score(
                y_true[:, tag_indices],
                y_pred[:, tag_indices],
                average=None,
            )
            tag_precision = metrics.average_precision_score(
                y_true[:, tag_indices],
                y_pred[:, tag_indices],
                average=None,
            )
            # save to csv file:
            tag_metrics_dict = {
                name: {"ROC-AUC": roc, "PR-AUC": precision}
                for name, roc, precision in zip(
                    dataset.label_list, tag_roc, tag_precision
                )
            }
            tag_metrics_df = pd.DataFrame.from_dict(tag_metrics_dict, orient="index")
            tag_metrics_df.to_csv(f"{output_dir}/tag_metrics.csv", index_label="tag")
        except:
            print("Warning: at least 1 tag-wise metric was not able to be computed.")
            pass

    return global_metrics_dict
