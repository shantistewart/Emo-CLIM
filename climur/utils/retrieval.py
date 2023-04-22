"""Functions for cross-modal retrieval evaluation."""


import torch
import torch.nn.functional as F
from torchmetrics.functional import retrieval_precision, retrieval_reciprocal_rank, retrieval_average_precision
import numpy as np
import tqdm
from typing import Dict, List, Tuple, Any


SUPPORTED_RETRIEVAL_METRICS = ["precision", "MRR", "MAP"]


def compute_retrieval_metrics(query_embeds: List, query_labels: List, item_embeds: List, item_labels: List, metric_names: List, k_vals: List, device: Any, mode: str = "cross-modal") -> Tuple[Dict, Dict]:
    """Computes various retrieval metrics.

    Args:
        query_embeds (list): Query embeddings.
        query_labels (list): Query labels.
        item_embeds (list): Item (thing that is retrieved) embeddings.
        item_labels (list): Item labels.
        metric_names (list): List of retrieval metric names to compute.
        k_vals (list): List of k values for retrieval metrics.
        device (PyTorch device): PyTorch device.
        mode (str): Retrieval mode ("cross-modal" or "intra-modal")
    
    Returns:
        macro_metrics (dict): Dictionary of macro-averaged retrieval metrics.
        metrics_per_class (dict): Dictionary of per-class retrieval metrics.
    """

    # sanity checks:
    assert len(query_embeds) == len(query_labels), "Query embedding and label list lengths don't match."
    assert len(item_embeds) == len(item_labels), "Item embedding and label list lengths don't match."
    assert tuple(query_embeds[0].size()) == tuple(item_embeds[0].size()), "Query and item embedding dimensions don't match."
    # validate parameters:
    if not(set(metric_names) <= set(SUPPORTED_RETRIEVAL_METRICS)):
        raise ValueError("At least one metric in metric_names isn't supported.")
    if mode not in ["cross-modal", "intra-modal"]:
        raise ValueError("Invalid retrieval mode.")
    if mode == "intra-modal":
        assert len(query_labels) == len(item_labels), "Query label and item label list lengths don't match."
    
    # convert item embeddings to tensor:
    item_embeds = torch.stack(item_embeds, dim=0)
    assert tuple(item_embeds.size()) == (len(item_labels), query_embeds[0].size(dim=0),), "Error converting list to tensor."
    # convert item labels to numpy array:
    item_labels = np.asarray(item_labels, dtype=object)
    assert item_labels.shape == (item_embeds.size(dim=0), ), "Error converting list to numpy array."

    # get set of unique query label:
    query_label_set = set(query_labels)
    # initialize dictionary for storing all per-query retrieval metrics:
    all_metrics = {}
    for name in metric_names:
        all_metrics[name] = {}
        if name == "precision":
            for k in k_vals:
                all_metrics[name][f"k={k}"] = {}
                for label in query_label_set:
                    all_metrics[name][f"k={k}"][label] = []
        else:
            all_metrics[name][f"k=N/A"] = {}
            for label in query_label_set:
                all_metrics[name][f"k=N/A"][label] = []
    
    # compute retrieval metrics:
    for idx in tqdm.tqdm(range(len(query_labels)), total=len(query_labels), desc="Computing retrieval metrics"):
        # remove query embedding from item embeddings if running intra-modal retrieval:
        if mode == "intra-modal":
            if idx == 0:
                item_embeds_new = item_embeds[idx+1:]
                item_labels_new = item_labels[idx+1:]
            elif idx == len(query_labels) - 1:
                item_embeds_new = item_embeds[:idx]
                item_labels_new = item_labels[:idx]
            else:
                item_embeds_new = torch.cat((item_embeds[:idx], item_embeds[idx+1:]), dim=0)
                item_labels_new = np.concatenate((item_labels[:idx], item_labels[idx+1:]), axis=0)
            assert tuple(item_embeds_new.size()) == (item_embeds.size(dim=0)-1, item_embeds.size(dim=1)), "Error removing query embedding from item embeddings."
            assert item_labels_new.shape == (len(item_labels)-1, ), "Error removing query label from item labels."
        else:
            item_embeds_new = item_embeds
            item_labels_new = item_labels
        
        # compute cosine similarity scores between query embedding and all item embeddings:
        query_embed = query_embeds[idx].unsqueeze(dim=0)
        cos_sim_scores = F.cosine_similarity(query_embed, item_embeds_new, dim=-1)  # (1, embed_dim) * (n_items, embed_dim) -> (n_items, )

        # map cosine similarity values from [-1.0, 1.0] -> [0.0, 1.0]:
        sim_scores = 0.5 * (cos_sim_scores + 1.0)
        assert tuple(sim_scores.size()) == (len(item_labels_new),), "Similarity scores has an unexpected shape."
        assert (torch.min(sim_scores).item() >= 0.0 and torch.max(sim_scores).item() <= 1.01), "Similarity values are not in range [0.0, 1.0]"

        # create retrieval ground-truth:
        query_label = query_labels[idx]
        retrieval_gt = np.equal(query_label, item_labels_new)
        retrieval_gt = torch.from_numpy(retrieval_gt)
        retrieval_gt = retrieval_gt.to(device)
        assert tuple(retrieval_gt.size()) == tuple(sim_scores.size()), "Error creating retrieval ground-truth."

        # compute retrieval metrics:
        for name in metric_names:
            if name == "precision":
                for k in k_vals:
                    precision = retrieval_precision(sim_scores, retrieval_gt, k=k).item()
                    all_metrics[name][f"k={k}"][query_label].append(precision)
            elif name == "MRR":
                reciprocal_rank = retrieval_reciprocal_rank(sim_scores, retrieval_gt).item()
                all_metrics[name][f"k=N/A"][query_label].append(reciprocal_rank)
            elif name == "MAP":
                avg_precision = retrieval_average_precision(sim_scores, retrieval_gt).item()
                all_metrics[name][f"k=N/A"][query_label].append(avg_precision)
    
    # compute mean of per-query retrieval metrics for each emotion class label:
    metrics_per_class = {}
    for name in all_metrics.keys():
        metrics_per_class[name] = {}
        for k_str in all_metrics[name].keys():
            metrics_per_class[name][k_str] = {}
            for label in query_label_set:
                metrics_per_class[name][k_str][label] = np.mean(np.asarray(all_metrics[name][k_str][label]))
    
    # compute mean of per-class retrieval metrics over classes (macro-averaging):
    macro_metrics = {}
    for name in metrics_per_class.keys():
        macro_metrics[name] = {}
        for k_str in metrics_per_class[name].keys():
            per_class_values = list(metrics_per_class[name][k_str].values())
            macro_metrics[name][k_str] = np.around(np.mean(np.asarray(per_class_values)), decimals=4)
    
    # round per-class retrieval metrics:
    metrics_per_class = {}
    for name in all_metrics.keys():
        metrics_per_class[name] = {}
        for k_str in all_metrics[name].keys():
            metrics_per_class[name][k_str] = {}
            for label in query_label_set:
                metrics_per_class[name][k_str][label] = np.around(np.mean(np.asarray(all_metrics[name][k_str][label])), decimals=4)
    
    return macro_metrics, metrics_per_class

