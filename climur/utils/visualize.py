"""Function for visualizing embeddings with t-SNE."""


import numpy as np
from numpy import ndarray
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from typing import Dict


def visualize_embeds(image_embeds: ndarray, image_labels: ndarray, audio_embeds: ndarray, audio_labels: ndarray, label2color: Dict, save_path: str) -> None:
    """Visualizes image and audio embeddings together with t-SNE.

    Args:
        image_embeds (ndarray): Image embeddings.
            shape: (n_images, embed_dim)
        image_labels (ndarray): Image labels.
            shape: (n_images, )
        audio_embeds (ndarray): Audio embeddings.
            shape: (n_audio_clips, embed_dim)
        audio_labels (ndarray): Audio labels.
            shape: (n_audio_clips, )
        label2color (dict): Dictionary mapping emotion labels to colors.
        save_path (str): Path to save plot.
    
    Returns: None
    """

    # extract set of emotion labels:
    emotion_labels = list(label2color.keys())

    # concatenate image and audio embeddings:
    all_embeds = np.concatenate((image_embeds, audio_embeds), axis=0)
    # labels = np.concatenate((image_labels, audio_labels), axis=0)
    # create t-SNE transform:
    tsne = TSNE(
        n_components=2,
        perplexity=5,
        learning_rate=130,
        metric="cosine",
        random_state=42
    )
    # map embeddings to 2 dimensions:
    transform_embeds = tsne.fit_transform(all_embeds)

    # normalize t-SNE embeddings and separate into 2 features:
    x_all_feats = MinMaxScaler().fit_transform(transform_embeds[:, 0].reshape(-1, 1))[:, 0]
    y_all_feats = MinMaxScaler().fit_transform(transform_embeds[:, 1].reshape(-1, 1))[:, 0]

    # split back into image and audio features:
    x_image_feats = x_all_feats[0:image_embeds.shape[0]]
    x_audio_feats = x_all_feats[image_embeds.shape[0]:]
    y_image_feats = y_all_feats[0:image_embeds.shape[0]]
    y_audio_feats = y_all_feats[image_embeds.shape[0]:]
    # sanity checks:
    assert x_image_feats.shape == (image_embeds.shape[0], ) and y_image_feats.shape == (image_embeds.shape[0], ), "Error with shape of image t-SNE features."
    assert x_audio_feats.shape == (audio_embeds.shape[0], ) and y_audio_feats.shape == (audio_embeds.shape[0], ), "Error with shape of audio t-SNE features."


    # create plot:
    fig = plt.figure()
    plt.rcParams["font.size"] = 10
    ax = fig.add_subplot(111)     # what is this?

    for label in emotion_labels:
        # get images t-SNE features for current emotion label:
        x_image_feats_curr = x_image_feats[image_labels == label]
        y_image_feats_curr = y_image_feats[image_labels == label]
        assert x_image_feats_curr.shape == y_image_feats_curr.shape, "Error with shape of per-class image t-SNE features."

        # add image points to scatter plot:
        ax.scatter(x_image_feats_curr, y_image_feats_curr, c=label2color[label], marker=".", label=f"image - {label}")

        # get audio t-SNE features for current emotion label:
        x_audio_feats_curr = x_audio_feats[audio_labels == label]
        y_audio_feats_curr = y_audio_feats[audio_labels == label]
        assert x_audio_feats_curr.shape == y_audio_feats_curr.shape, "Error with shape of per-class audio t-SNE features."

        # add audio points to scatter plot:
        ax.scatter(x_audio_feats_curr, y_audio_feats_curr, c=label2color[label], marker="x", label=f"audio - {label}")
    
    # format plot:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best")

    # save plot:
    fig.savefig(save_path)
    plt.close()

