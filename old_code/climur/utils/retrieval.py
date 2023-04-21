"""Functions for cross-modal retrieval evaluation."""


import torch
import torch.nn.functional as F
from torchmetrics.functional import retrieval_precision, retrieval_reciprocal_rank
import numpy as np
import tqdm
from typing import Dict, List, Any


SUPPORTED_RETRIEVAL_METRICS = ["precision", "MRR"]


def compute_retrieval_metrics(
    query_embeds: List,
    query_labels: List,
    item_embeds: List,
    item_labels: List,
    metric_names: List,
    k_vals: List,
    device: Any,
) -> Dict:
    """Computes various retrieval metrics.

    Args:
        query_embeds (list): Query embeddings.
        query_labels (list): Query labels.
        item_embeds (list): Item (thing that is retrieved) embeddings.
        item_labels (list): Item labels.
        metric_names (list): List of retrieval metric names to compute.
        k_vals (list): List of k values for retrieval metrics.
        device (PyTorch device): PyTorch device.

    Returns:
        mean_metrics (dict): Dictionary of retrieval metrics.
    """

    # sanity checks:
    assert len(query_embeds) == len(
        query_labels
    ), "Query embedding and label list lengths don't match."
    assert len(item_embeds) == len(
        item_labels
    ), "Item embedding and label list lengths don't match."
    assert tuple(query_embeds[0].size()) == tuple(
        item_embeds[0].size()
    ), "Query and item embedding dimensions don't match."
    # validate metric names:
    assert set(metric_names) <= set(
        SUPPORTED_RETRIEVAL_METRICS
    ), "At least one metric in metric_names isn't supported."

    # convert item embeddings to tensor:
    item_embeds = torch.stack(item_embeds, dim=0)
    assert tuple(item_embeds.size()) == (
        len(item_labels),
        query_embeds[0].size(dim=0),
    ), "Error converting list to tensor."

    # initialize dictionary for storing all per-query retrieval metrics:
    all_metrics = {}
    for name in metric_names:
        all_metrics[name] = {}
        if name == "precision":
            for k in k_vals:
                all_metrics[name][f"k={k}"] = []
        elif name == "MRR":
            all_metrics[name][f"k=N/A"] = []

    # compute retrieval metrics:
    for idx in tqdm.tqdm(
        range(len(query_labels)), total=len(query_labels), desc="Computing retrieval metrics"
    ):
        # compute cosine similarity scores between query embedding and all item embeddings:
        query_embed = query_embeds[idx].unsqueeze(dim=0)
        cos_sim_scores = F.cosine_similarity(
            query_embed, item_embeds, dim=-1
        )  # (1, embed_dim) * (n_items, embed_dim) -> (n_items, )

        # map cosine similarity values from [-1.0, 1.0] -> [0.0, 1.0]:
        sim_scores = 0.5 * (cos_sim_scores + 1.0)
        assert tuple(sim_scores.size()) == (
            len(item_labels),
        ), "Similarity scores has an unexpected shape."
        assert (
            torch.min(sim_scores).item() >= 0.0 and torch.max(sim_scores).item() <= 1.0
        ), "Similarity values are not in range [0.0, 1.0]"

        # create retrieval ground-truth:
        retrieval_gt = np.equal(query_labels[idx], np.asarray(item_labels, dtype=object))
        retrieval_gt = torch.from_numpy(retrieval_gt)
        retrieval_gt = retrieval_gt.to(device)
        assert tuple(retrieval_gt.size()) == tuple(
            sim_scores.size()
        ), "Error creating retrieval ground-truth."

        # compute retrieval metrics:
        for name in metric_names:
            if name == "precision":
                for k in k_vals:
                    precision = retrieval_precision(sim_scores, retrieval_gt, k=k).item()
                    all_metrics[name][f"k={k}"].append(precision)
            elif name == "MRR":
                reciprocal_rank = retrieval_reciprocal_rank(sim_scores, retrieval_gt).item()
                all_metrics[name][f"k=N/A"].append(reciprocal_rank)

    # compute mean of per-query retrieval metrics over all queries:
    mean_metrics = {}
    for name, metrics in all_metrics.items():
        mean_metrics[name] = {}
        for k_str in metrics.keys():
            mean_metrics[name][k_str] = np.mean(np.asarray(metrics[k_str]))

    return mean_metrics
