"""PyTorch Lightning LightningModule class for image/music supervised fine-tuning."""


from pytorch_lightning import LightningModule
import torch, torch.nn as nn
from torchmetrics import AveragePrecision
from typing import Dict, Any
from climur.utils.eval import get_embedding_ds
from sklearn.metrics import average_precision_score, roc_auc_score


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
        backbone: LightningModule,
        embed_dim: int,
        hparams: Dict[str, Any],
        num_classes: int = 50,
        device: Any = None,
    ) -> None:
        """Initialization.

        Args:
            audio_embed_dim (int): Dimension of audio embeddings (outputs of audio backbone model).
            hparams (dict): Dictionary of hyperparameters.
            num_classes (int): Number of classes.
            device (PyTorch device): PyTorch device.

        Returns: None
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        self.out_dim = num_classes
        self.torch_device = device
        self.backbone = backbone
        self.backbone.eval()

        # create base audio projector:
        self.projector = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=True),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(in_features=embed_dim, out_features=self.out_dim, bias=True),
        )

        # create loss functions:
        self.criterion = nn.BCEWithLogitsLoss()
        #self.pr_auc = AveragePrecision(task="multilabel", pos_label=1)

    def forward(self, audios: torch.Tensor):
        embeds = get_embedding_ds(self.backbone, audios)
        return self.projector(embeds[0])

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
        audios, labels = batch
        # send to device if required:
        if self.torch_device is not None:
            audios = audios.to(self.torch_device)
            labels = labels.to(self.torch_device)

        # forward pass:
        predictions = self.forward(audios)
        # compute loss:
        loss = self.criterion(predictions, labels)
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
        audios, labels = batch
        # send to device if required:
        if self.torch_device is not None:
            audios = audios.to(self.torch_device)
            labels = labels.to(self.torch_device)

        # forward pass:
        predictions = self.forward(audios)
        # evaluate:
        loss = self.criterion(predictions, labels)
        #pr_auc = self.pr_auc(predictions, labels)
        # log results:
        self.log("validation/loss", loss)
        #self.log("validation/pr_auc", pr_auc)
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
        return {"optimizer": optimizer}
