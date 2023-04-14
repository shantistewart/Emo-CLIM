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
    
    def forward(self, embeds: Tensor, labels: Tensor) -> Tensor:
        """Forward pass.

        Args:
            embeds (Tensor): Embeddings.
                shape: (N, n_views, embed_dim)
            labels (Tensor): Labels.
                shape: (N, )
        
        Returns:
            loss (Tensor): SupCon loss.
        """

        # validate shapes:
        assert len(tuple(embeds.size())) == 3, "Embeddings has unexpected shape."
        assert embeds.size(dim=0) == labels.size(dim=0), "Batch sizes of embeddings and labels don't match."
        # get input dimensions:
        batch_size, n_views, _ = tuple(embeds.size())


        # ------------
        # CREATE MASKS
        # ------------

        # reshape labels: (N, ) -> (N, 1)
        labels = labels.unsqueeze(dim=-1)

        # create supervised mask for positive pairs (masks out negative pairs):
        sup_mask = torch.eq(labels, labels.T).float()     # shape: (N, N)
        sup_mask = sup_mask.to(device)     # TODO: Try removing .to(device), since Lightning should take care of this.
        # tile (expand) mask to account for n_views: (N, N) -> (N_new, N_new)
        sup_mask = sup_mask.repeat(n_views, n_views)

        # create mask to mask out same-sample pairs:
        same_mask = 1 - torch.eye(sup_mask.size(dim=0))     # shape: (N_new, N_new)
        same_mask = same_mask.to(device)     # TODO: Try removing .to(device), since Lightning should take care of this.

        # combine supervised and same-sample masks:
        mask = same_mask * sup_mask     # shape: (N_new, N_new)
        assert tuple(mask.size()) == (n_views * batch_size, n_views * batch_size), "mask has incorrect shape."


        # ------------------
        # COMPUTE ALL LOGITS
        # ------------------

        # reshape embeddings: (N, n_views, embed_dim) -> (N_new, embed_dim), where N_new = n_views * N
        embeds = torch.cat(torch.unbind(embeds, dim=1), dim=0)

        # compute all logits = dot products between all pairs of embeddings (divided by temperature):
        all_logits = torch.div(torch.matmul(embeds, embeds.T), self.temperature)     # shape: (N_new, N_new)
        # normalize for numerical stability:
        all_logits_max, _ = torch.max(all_logits, dim=1, keepdim=True)
        all_logits = all_logits - all_logits_max.detach()     # shape: (N_new, N_new)
        assert tuple(all_logits.size()) == (n_views * batch_size, n_views * batch_size), "all_logits has incorrect shape."


        # -------------------------
        # COMPUTE LOG PROBABILITIES
        # -------------------------

        # compute exponents of all logits and mask out same-sample pairs:
        exp_all_logits = torch.exp(all_logits)
        exp_denom_logits = same_mask * exp_all_logits     # shape: (N_new, N_new)
        # sum over non-anchor (z_k) dimension: (N_new, N_new) -> (N_new, 1)
        exp_denom_logits_sum = exp_denom_logits.sum(dim=1, keepdim=True)

        # compute log probabilities (everything inside inner sum) and mask out negative and same-sample pairs:
        log_prob_all = (all_logits - torch.log(exp_denom_logits_sum))     # shape: (N_new, N_new)
        log_prob = mask * log_prob_all     # shape: (N_new, N_new)
        assert tuple(log_prob.size()) == (n_views * batch_size, n_views * batch_size), "log_prob has incorrect shape."


        # -------------------------
        # AVERAGE LOG PROBABILITIES
        # -------------------------

        # sum over non-anchor (z_p) dimension: (N_new, N_new) -> (N_new, )
        mean_log_prob = log_prob.sum(dim=1)
        # divide by size of positive pair sets:
        mean_log_prob = mean_log_prob / mask.sum(dim=1)     # shape: (N_new, )
        assert tuple(mean_log_prob.size()) == (n_views * batch_size, ), "mean_log_prob has incorrect shape."

        # flip sign:
        mean_log_prob = -1 * mean_log_prob
        # compute mean over anchor (z_i) dimension:
        loss = mean_log_prob.mean()

        return loss

