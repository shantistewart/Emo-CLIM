"""PyTorch Lightning LightningModule class for image-music supervised contrastive learning."""


from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple, Any

from climur.losses.original_supcon import SupConLoss


# TEMP:
TESTING = True
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")


class Image2Music(LightningModule):
    """LightningModule class for image-music supervised contrastive learning.

    Attributes:
        image_backbone (nn.Module): Image backbone model.
        audio_backbone (nn.Module): Audio backbone model.
        image_projector (nn.Module): Image projector model.
        audio_projector (nn.Module): Audio projector model.
        criterion (PyTorch loss): Loss function.
        joint_embed_dim (int): Dimension of joint embedding space.
        normalize_image_embeds (bool): Selects whether to normalize image embeddings.
        normalize_audio_embeds (bool): Selects whether to normalize audio embeddings.
    
    """

    def __init__(
            self,
            image_backbone: nn.Module,
            audio_backbone: nn.Module, 
            joint_embed_dim: int,
            hparams: Dict,
            image_embed_dim: int,
            audio_embed_dim: int,
            normalize_image_embeds: bool = True,
            normalize_audio_embeds: bool = True,
            freeze_image_backbone: bool = False,
            freeze_audio_backbone: bool = False
        ) -> None:
        """Initialization.

        Args:
            image_backbone (nn.Module): Image backbone model.
            audio_backbone (nn.Module): Audio backbone model.
            joint_embed_dim (int): Dimension of joint embedding space.
            hparams (dict): Dictionary of hyperparameters.
            image_embed_dim (int): Dimension of image embeddings (outputs of image backbone model).
            audio_embed_dim (int): Dimension of audio embeddings (outputs of audio backbone model).
            normalize_image_embeds (bool): Selects whether to normalize image embeddings.
            normalize_audio_embeds (bool): Selects whether to normalize audio embeddings.
            freeze_image_backbone (bool): Selects whether to freeze image backbone model.
            freeze_audio_backbone (bool): Selects whether to freeze audio backbone model.
        
        Returns: None
        """

        super().__init__()
        # save hyperparameters (saves to self.hparams):
        self.save_hyperparameters(hparams)

        # save parameters:
        self.normalize_image_embeds = normalize_image_embeds
        self.normalize_audio_embeds = normalize_audio_embeds
        self.joint_embed_dim = joint_embed_dim

        # save image backbone model and freeze if selected:
        self.image_backbone = image_backbone
        if freeze_image_backbone:
            self.image_backbone.requires_grad_(False)
        # save audio backbone model and freeze if selected:
        self.audio_backbone = audio_backbone
        if freeze_audio_backbone:
            self.audio_backbone.requires_grad_(False)
        
        # create projectors:
        projector_hidden_dim = max(image_embed_dim, audio_embed_dim)
        self.image_projector = nn.Sequential(
            nn.Linear(in_features=image_embed_dim, out_features=projector_hidden_dim, bias=True),     # TODO: Maybe also try bias=False.
            nn.ReLU(),
            nn.Linear(in_features=projector_hidden_dim, out_features=joint_embed_dim, bias=True)
        )
        self.audio_projector = nn.Sequential(
            nn.Linear(in_features=audio_embed_dim, out_features=projector_hidden_dim, bias=True),     # TODO: Maybe also try bias=False.
            nn.ReLU(),
            nn.Linear(in_features=projector_hidden_dim, out_features=joint_embed_dim, bias=True)
        )

        # create loss function:
        self.criterion = SupConLoss(temperature=self.hparams.loss_temperature)
    
    def forward(self, images: Tensor, audios: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            images (Tensor): Preprocessed images.
                shape: (batch_size, image_channels, image_height, image_width)
            audios (Tensor): Raw audio clips.
                shape: (batch_size, audio_clip_length)
        
        Returns:
            image_embeds (Tensor): Image embeddings (in joint embedding space).
                shape: (batch_size, joint_embed_dim)
            audio_embeds (Tensor): Audio embeddings (in joint embedding space).
                shape: (batch_size, joint_embed_dim)
        """

        # encode each modality with backbone models:
        images_enc = self.image_backbone(images)
        audios_enc = self.audio_backbone(audios)
        # convert to float32 datatype (if not already):
        images_enc = images_enc.float()     # TODO: Double-check that this is ok for CLIP outputs.
        audios_enc = audios_enc.float()
        # L2-normalize embeddings if selected:
        if self.normalize_image_embeds:
            images_enc = F.normalize(images_enc, p=2, dim=-1)
        if self.normalize_audio_embeds:
            audios_enc = F.normalize(audios_enc, p=2, dim=-1)
        
        # project to joint multimodal embedding space:
        image_embeds = self.image_projector(images_enc)
        audio_embeds = self.audio_projector(audios_enc)
        # L2-normalize embeddings:
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        audio_embeds = F.normalize(audio_embeds, p=2, dim=-1)

        return image_embeds, audio_embeds
    
    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        """Training step.

        Args:
            batch (dict): Batch dictionary with keys/values:
                "image": (Tensor) Images.
                    shape: (batch_size, image_channels, image_height, image_width)
                "image_label" (Tensor): Image emotion label indices.
                    shape: (batch_size, )
                "audio": (Tensor) Audio clips.
                    shape: (batch_size, audio_clip_length)
                "audio_label" (Tensor): Audio emotion label indices.
                    shape: (batch_size, )
            batch_idx (int): Batch index (unused).
        
        Returns:
            loss (torch.Tensor): SupCon loss.
        """

        # unpack batch:
        images = batch["image"]
        image_labels = batch["image_label"]
        audios = batch["audio"]
        audio_labels = batch["audio_label"]

        # TEMP:
        if TESTING:
            images = images.to(device)
            image_labels = image_labels.to(device)
            audios = audios.to(device)
            audio_labels = audio_labels.to(device)
        
        # forward pass:
        image_embeds, audio_embeds = self.forward(images, audios)

        # TEMP——compute SupCon loss using original code:
        all_embeds = torch.cat((image_embeds, audio_embeds), dim=0)
        all_labels = torch.cat((image_labels, audio_labels), dim=0)
        # insert views dimension (1 view per sample for now):
        all_embeds = all_embeds.unsqueeze(dim=1)
        # compute loss:
        loss = self.criterion(all_embeds, labels=all_labels)

        # TODO: Adapt original SupCon code to multimodal case.

        # log training losses:
        self.log("train_loss", loss)

        return loss
    
    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        """Validation step.

        Args:
            batch (dict): Batch dictionary with keys ("image", "image_label", "audio", "audio_label").
            batch_idx (int): Batch index (unused).
        
        Returns:
            loss (torch.Tensor): Loss for batch (singleton Tensor).
        """

        # unpack batch:
        images = batch["image"]
        image_labels = batch["image_label"]
        audios = batch["audio"]
        audio_labels = batch["audio_label"]

        # TEMP:
        if TESTING:
            images = images.to(device)
            image_labels = image_labels.to(device)
            audios = audios.to(device)
            audio_labels = audio_labels.to(device)
        
        # forward pass:
        image_embeds, audio_embeds = self.forward(images, audios)

        # TEMP——compute SupCon loss using original code:
        all_embeds = torch.cat((image_embeds, audio_embeds), dim=0)
        all_labels = torch.cat((image_labels, audio_labels), dim=0)
        # insert views dimension (1 view per sample for now):
        all_embeds = all_embeds.unsqueeze(dim=1)
        # compute loss:
        loss = self.criterion(all_embeds, labels=all_labels)

        # TODO: Adapt original SupCon code to multimodal case.

        # log validation losses:
        self.log("val_loss", loss)

        return loss
    
    def configure_optimizers(self) -> Any:     # TODO: Maybe add a learning rate scheduler.
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

