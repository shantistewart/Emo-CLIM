"""Cross-modal SupCon loss (original SupCon loss adapted for multiple data modalities). Adapted from from https://github.com/HobbitLong/SupContrast/blob/master/losses.py
Reference: "Supervised Contrastive Learning", Khosla et al., 2020."""


import torch
import torch.nn as nn
from torch import Tensor
import warnings
from typing import Any


class CrossModalSupCon(nn.Module):
    """Cross-modal SupCon loss.

    Attributes:
        temperature (float): Temperature hyperparameter.
        torch_device (PyTorch device): PyTorch device.
    """

    def __init__(self, temperature: float = 0.07, device: Any = None) -> None:
        """Initialization.

        Args:
            temperature (float): Temperature hyperparameter.
            device (PyTorch device): PyTorch device.
        
        Returns: None
        """

        super(CrossModalSupCon, self).__init__()

        # save parameters:
        self.temperature = temperature
        self.torch_device = device
    
    def forward(self, embeds_M1: Tensor, labels_M1: Tensor, embeds_M2: Tensor, labels_M2: Tensor) -> Tensor:
        """Forward pass.

        Args:
            embeds_M1 (Tensor): Embeddings of modality M1 (used as anchor samples).
                shape: (N, n_views, embed_dim)     # TODO: Maybe allow n_views to be diffent for modalities.
            labels_M1 (Tensor): Labels of modality M1.
                shape: (N, )
            embeds_M2 (Tensor): Embeddings of modality M2 (used as samples to contrast against).
                shape: (N, n_views, embed_dim)
            labels_M2 (Tensor): Labels of modality M2.
                shape: (N, )
        
        Returns:
            loss (Tensor): SupCon loss.
        """

        # validate shapes:
        assert len(tuple(embeds_M1.size())) == 3 and len(tuple(embeds_M2.size())) == 3, "embeds_M1 or embeds_M2 has an unexpected shape."
        assert embeds_M1.size(dim=0) == labels_M1.size(dim=0) and embeds_M2.size(dim=0) == labels_M2.size(dim=0), "Batch sizes of embeddings and labels don't match."
        assert tuple(embeds_M1.size()) == tuple(embeds_M2.size()), "Shape of embeds_M1 and embeds_M2 don't match."
        # get input dimensions:
        batch_size, n_views, _ = tuple(embeds_M1.size())


        # ------------
        # CREATE MASKS
        # ------------

        # reshape labels: (N, ) -> (N, 1)
        labels_M1 = labels_M1.unsqueeze(dim=-1)
        labels_M2 = labels_M2.unsqueeze(dim=1)

        # create supervised mask for positive pairs (masks out negative pairs):
        sup_mask = torch.eq(labels_M1, labels_M2.T).float()     # shape: (N, N)
        if self.torch_device is not None:
            sup_mask = sup_mask.to(self.torch_device)
        # tile (expand) mask to account for n_views: (N, N) -> (N_new, N_new)
        mask = sup_mask.repeat(n_views, n_views)
        assert tuple(mask.size()) == (n_views * batch_size, n_views * batch_size), "mask has incorrect shape."


        # ------------------
        # COMPUTE ALL LOGITS
        # ------------------

        # reshape embeddings: (N, n_views, embed_dim) -> (N_new, embed_dim), where N_new = n_views * N
        embeds_M1 = torch.cat(torch.unbind(embeds_M1, dim=1), dim=0)
        embeds_M2 = torch.cat(torch.unbind(embeds_M2, dim=1), dim=0)

        # compute all logits = dot products between all pairs of embeddings (divided by temperature):
        all_logits = torch.div(torch.matmul(embeds_M1, embeds_M2.T), self.temperature)     # shape: (N_new, N_new)
        # normalize for numerical stability:
        all_logits_max, _ = torch.max(all_logits, dim=1, keepdim=True)
        all_logits = all_logits - all_logits_max.detach()     # shape: (N_new, N_new)
        assert tuple(all_logits.size()) == (n_views * batch_size, n_views * batch_size), "all_logits has incorrect shape."


        # -------------------------
        # COMPUTE LOG PROBABILITIES
        # -------------------------

        # compute exponents of all logits:
        exp_all_logits = torch.exp(all_logits)     # shape: (N_new, N_new)
        # sum over M2 (z_k) dimension: (N_new, N_new) -> (N_new, 1)
        exp_denom_logits_sum = exp_all_logits.sum(dim=1, keepdim=True)

        # compute log probabilities (everything inside inner sum) and mask out negative pairs:
        log_prob_all = all_logits - torch.log(exp_denom_logits_sum)     # shape: (N_new, N_new)
        log_prob = mask * log_prob_all     # shape: (N_new, N_new)
        # replace possible NaN and +/- infinity values with zeros:
        if not torch.all(torch.isfinite(log_prob)):     # TODO: Maybe clip very large values using a threshold.
            print()
            warnings.warn("log_prob contains NaN or +/- infinity values, replacing with zeros.", RuntimeWarning)
            print()
            log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=0.0)
        assert tuple(log_prob.size()) == (n_views * batch_size, n_views * batch_size), "log_prob has incorrect shape."


        # -------------------------
        # AVERAGE LOG PROBABILITIES
        # -------------------------

        # sum over M2 (z_p) dimension: (N_new, N_new) -> (N_new, )
        mean_log_prob = log_prob.sum(dim=1)
        # divide by size of positive pair sets:
        if torch.any(mask.sum(dim=1) == 0.0).item():
            print()
            warnings.warn("mask.sum(dim=1) contains zeros.", RuntimeWarning)
        mean_log_prob = mean_log_prob / mask.sum(dim=1)     # shape: (N_new, )
        # replace possible NaN and +/- infinity values with zeros:
        if not torch.all(torch.isfinite(mean_log_prob)):
            warnings.warn("mean_log_prob contains NaN or +/- infinity values, replacing with zeros.", RuntimeWarning)
            print()
            mean_log_prob = torch.nan_to_num(mean_log_prob, nan=0.0, posinf=0.0, neginf=0.0)
        assert tuple(mean_log_prob.size()) == (n_views * batch_size, ), "mean_log_prob has incorrect shape."

        # flip sign:
        mean_log_prob = -1 * mean_log_prob
        # compute mean over M1 (z_i) dimension:
        loss = mean_log_prob.mean()
        if torch.isnan(loss).item():
            raise RuntimeError("loss is NaN.")

        return loss

