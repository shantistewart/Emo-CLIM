"""Intra-modal SupCon loss. Adapted from from https://github.com/HobbitLong/SupContrast/blob/master/losses.py
Reference: "Supervised Contrastive Learning", Khosla et al., 2020."""


import torch
import torch.nn as nn
from torch import Tensor


# TEMP:
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")


class IntramodalSupCon(nn.Module):
    """Intra-modal SupCon loss.

    Attributes:
        temperature (float): Temperature hyperparameter.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        """Initialization.

        Args:
            temperature (float): Temperature hyperparameter.
        
        Returns: None
        """

        super(IntramodalSupCon, self).__init__()

        # save parameters:
        self.temperature = temperature
    
    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        """Forward pass.

        Args:
            features (Tensor): Features (embeddings).
                shape: (N, n_views, embed_dim)
            labels (Tensor): Labels.
                shape: (N, )
        
        Returns:
            loss (Tensor): SupCon loss.
        """

        # validate shapes:
        assert len(tuple(features.size())) == 3, "Features has unexpected shape."
        assert features.size(dim=0) == labels.size(dim=0), "Batch sizes of features and labels don't match."
        # get batch size:
        batch_size = features.size(dim=0)

        # reshape labels: (N, ) -> (N, 1)
        labels = labels.unsqueeze(dim=-1)
        assert labels.shape[0] == batch_size, "Error reshaping labels."

        # create mask:
        mask = torch.eq(labels, labels.T).float()     # shape: (N, N)
        mask = mask.to(device)
        assert tuple(mask.size()) == (batch_size, batch_size), "Mask has incorrect shape."

        # save n_views:
        contrast_count = features.shape[1]
        # remove views dimension from features: (N, n_views, embed_dim) -> (n_views * N, embed_dim)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # set things:
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute anchor logits: (n_views * N, embed_dim) * (embed_dim, n_views * N) -> (n_views * N, n_views * N)
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # normalize for numerical stability:
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()     # shape: (n_views * N, n_views * N)

        # tile mask to account for n_views: (N, N) -> (n_views * N, n_views * N)
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast pairs:
        logits_mask = 1 - torch.eye(mask.size(dim=0))     # shape: (n_views * N, n_views * N)
        logits_mask = logits_mask.to(device)
        mask = logits_mask * mask     # shape: (n_views * N, n_views * N)

        # compute denominator inside log: exp(z_i * z_k / tau), for all i, for all k != i
        #    note: only self-contrast pairs are masked out here
        exp_logits = logits_mask * torch.exp(logits)     # shape: (n_views * N, n_views * N)
        # sum over z_k dimension: (n_views * N, n_views * N) -> (n_views * N, 1)
        exp_logits_sum = exp_logits.sum(dim=1, keepdim=True)

        # compute log probabilities (divide anchor logits by denominator and take log):
        log_prob = logits - torch.log(exp_logits_sum)     # shape: (n_views * N, n_views * N)

        # compute mean of log probabilities over positive pairs (masking out negative pairs):
        mean_log_prob_pos = (mask * log_prob).sum(dim=1)     # shape: (n_views * N, )
        # divide by size of positive pair sets:
        mean_log_prob_pos = mean_log_prob_pos / mask.sum(dim=1)     # shape: (n_views * N, )

        # flip sign:
        loss = -1 * mean_log_prob_pos     # shape: (n_views * N, )
        # sum over anchors (z_i dimension):
        loss = loss.mean()

        return loss

