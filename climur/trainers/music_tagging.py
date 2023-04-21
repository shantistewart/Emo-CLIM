"""PyTorch Lightning LightningModule class for image/music supervised fine-tuning."""


from pytorch_lightning import LightningModule
import torch, torch.nn as nn
import torch.nn.functional as F
from torchmetrics import AveragePrecision, functional
from typing import Dict, Any


class MTAT_Training(LightningModule):
    """LightningModule class for MagnaTagATune downstream task.

    Attributes:
        audio_backbone (nn.Module): Audio backbone model.
        output_embed_dim (int): Dimension of output embedding space(s).
        normalize_audio_embeds (bool): Selects whether to normalize audio embeds.
        torch_device (PyTorch device): PyTorch device.
    """

    def __init__(
        self,
        audio_backbone: nn.Module,
        output_embed_dim: int,
        audio_embed_dim: int,
        hparams: Dict[str, Any],
        normalize_audio_embeds: bool = True,
        freeze_audio_backbone: bool = False,
        device: Any = None,
    ) -> None:
        """Initialization.

        Args:
            audio_backbone (nn.Module): Audio backbone model.
            output_embed_dim (int): Dimension of output embedding space(s).
            audio_embed_dim (int): Dimension of audio embeddings (outputs of audio backbone model).
            hparams (dict): Dictionary of hyperparameters.
            normalize_audio_embeds (bool): Selects whether to normalize audio embeddings.
            freeze_audio_backbone (bool): Selects whether to freeze audio backbone model.
            device (PyTorch device): PyTorch device.

        Returns: None
        """

        super().__init__()
        # saves to self.hparams:
        self.save_hyperparameters(hparams)

        # save parameters:
        self.output_embed_dim = output_embed_dim
        self.normalize_audio_embeds = normalize_audio_embeds
        self.torch_device = device

        # save audio backbone model and freeze if selected:
        self.audio_backbone = audio_backbone
        if freeze_audio_backbone:
            self.audio_backbone.requires_grad_(False)

        # create base audio projector:
        self.projector = nn.Sequential(
            nn.Linear(in_features=audio_embed_dim, out_features=audio_embed_dim, bias=True),
            nn.BatchNorm1d(audio_embed_dim),
            nn.ReLU(),
            nn.Linear(in_features=audio_embed_dim, out_features=50, bias=True),
        )

        # create loss functions:
        self.criterion = nn.BCEWithLogitsLoss()
        self.pr_auc = AveragePrecision(task="multilabel")

    def forward(self, audios: torch.Tensor):
        """Forward pass.

        Args:
            audios (torch.Tensor): Raw audio clips.
                shape: (batch_size, audio_clip_length)

        Returns:
            predictions (torch.Tensor): Predictions for each audio clip.
                shape: (batch_size, num_labels)
        """

        audios_enc = self.audio_backbone(audios).float()
        # L2-normalize embeddings if selected:
        if self.normalize_audio_embeds:
            audios_enc = F.normalize(audios_enc, p=2, dim=-1)

        # pass through projector:
        return self.projector(audios_enc)

    def training_step(self, batch: Dict, _: int) -> torch.Tensor:
        """Training step.

        Args:
            batch (dict): Batch dictionary with keys/values:
                "audio": (torch.Tensor) Audio clips.
                    shape: (batch_size, audio_clip_length)
                "audio_label" (torch.Tensor): Audio emotion label indices.
                    shape: (batch_size, )

        Returns:
            loss (torch.Tensor): BCE loss.
        """

        # unpack batch:
        audios = batch["audio"]
        audio_labels = batch["audio_label"]
        # send to device if required:
        if self.torch_device is not None:
            audios = audios.to(self.torch_device)
            audio_labels = audio_labels.to(self.torch_device)
        # forward pass:
        predictions = self.forward(audios)
        # compute loss:
        loss = self.criterion(predictions, audio_labels)
        # log loss:
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: Dict, _: int) -> torch.Tensor:
        """Validation step.

        Args:
            batch (dict): Batch dictionary with keys ("audio", "audio_label").

        Returns:
            loss (torch.Tensor): BCE loss.
        """

        # unpack batch:
        audios = batch["audio"]
        audio_labels = batch["audio_label"]
        # send to device if required:
        if self.torch_device is not None:
            audios = audios.to(self.torch_device)
            audio_labels = audio_labels.to(self.torch_device)
        # forward pass:
        predictions = self.forward(audios)
        # evaluate:
        loss = self.criterion(predictions, audio_labels)
        pr_auc = self.average_precision(predictions, audio_labels)
        roc_auc = functional.auroc(predictions, audio_labels)
        # log results:
        self.log("validation/loss", loss)
        self.log("validation/pr_auc", pr_auc)
        self.log("validation/roc_auc", roc_auc)
        return loss

    def configure_optimizers(self) -> Any:
        """Configures optimizer.

        Args: None

        Returns:
            optimizer (subclass of torch.optim.Optimizer): PyTorch optimizer.
        """

        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learn_rate)
        elif self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learn_rate)
        else:
            raise ValueError("Invalid optimizer.")
        return optimizer
