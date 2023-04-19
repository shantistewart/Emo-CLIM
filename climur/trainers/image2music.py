"""PyTorch Lightning LightningModule class for image-music supervised contrastive learning."""


from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple, Any

from climur.losses.intramodal_supcon import IntraModalSupCon
from climur.losses.crossmodal_supcon import CrossModalSupCon


class Image2Music(LightningModule):
    """LightningModule class for image-music supervised contrastive learning.

    Attributes:
        image_backbone (nn.Module): Image backbone model.
        audio_backbone (nn.Module): Audio backbone model.
        image_projector (nn.Module): Image projector model.
        audio_projector (nn.Module): Audio projector model.
        joint_embed_dim (int): Dimension of joint embedding space.
        normalize_image_embeds (bool): Selects whether to normalize image embeddings.
        normalize_audio_embeds (bool): Selects whether to normalize audio embeddings.
        image2image_supcon (nn.Module): Image-to-image SupCon loss criterion.
        audio2audio_supcon (nn.Module): Audio-to-audio SupCon loss criterion.
        image2audio_supcon (nn.Module): Image-to-audio SupCon loss criterion.
        audio2image_supcon (nn.Module): Audio-to-image SupCon loss criterion.
        torch_device (PyTorch device): PyTorch device.
    """

    def __init__(
            self,
            image_backbone: nn.Module,
            audio_backbone: nn.Module, 
            joint_embed_dim: int,
            image_embed_dim: int,
            audio_embed_dim: int,
            hparams: Dict,
            projector_dropout_prob: float = 0.5,
            normalize_image_embeds: bool = True,
            normalize_audio_embeds: bool = True,
            freeze_image_backbone: bool = False,
            freeze_audio_backbone: bool = False,
            device: Any = None
        ) -> None:
        """Initialization.

        Args:
            image_backbone (nn.Module): Image backbone model.
            audio_backbone (nn.Module): Audio backbone model.
            joint_embed_dim (int): Dimension of joint embedding space.
            image_embed_dim (int): Dimension of image embeddings (outputs of image backbone model).
            audio_embed_dim (int): Dimension of audio embeddings (outputs of audio backbone model).
            hparams (dict): Dictionary of hyperparameters.
            projector_dropout_prob (float): Dropout probability for projectors.
            normalize_image_embeds (bool): Selects whether to normalize image embeddings.
            normalize_audio_embeds (bool): Selects whether to normalize audio embeddings.
            freeze_image_backbone (bool): Selects whether to freeze image backbone model.
            freeze_audio_backbone (bool): Selects whether to freeze audio backbone model.
            device (PyTorch device): PyTorch device.
        
        Returns: None
        """

        super().__init__()
        # save hyperparameters (saves to self.hparams):
        self.save_hyperparameters(hparams)

        # save parameters:
        self.normalize_image_embeds = normalize_image_embeds
        self.normalize_audio_embeds = normalize_audio_embeds
        self.joint_embed_dim = joint_embed_dim
        self.torch_device = device

        # save image backbone model and freeze if selected:
        self.image_backbone = image_backbone
        if freeze_image_backbone:
            self.image_backbone.requires_grad_(False)
        # save audio backbone model and freeze if selected:
        self.audio_backbone = audio_backbone
        if freeze_audio_backbone:
            self.audio_backbone.requires_grad_(False)

        # create projectors:
        projector_hidden_dim = min(image_embed_dim, audio_embed_dim)
        self.image_projector = nn.Sequential(
            nn.Linear(in_features=image_embed_dim, out_features=projector_hidden_dim, bias=True),     # TODO: Maybe also try bias=False.
            nn.BatchNorm1d(projector_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=projector_dropout_prob),
            nn.Linear(in_features=projector_hidden_dim, out_features=joint_embed_dim, bias=True)
        )
        self.audio_projector = nn.Sequential(
            nn.Linear(in_features=audio_embed_dim, out_features=projector_hidden_dim, bias=True),     # TODO: Maybe also try bias=False.
            nn.BatchNorm1d(projector_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=projector_dropout_prob),
            nn.Linear(in_features=projector_hidden_dim, out_features=joint_embed_dim, bias=True)
        )

        # create loss functions:
        self.image2image_supcon = IntraModalSupCon(temperature=hparams["loss_temperature"], device=device)
        self.audio2audio_supcon = IntraModalSupCon(temperature=hparams["loss_temperature"], device=device)
        self.image2audio_supcon = CrossModalSupCon(temperature=hparams["loss_temperature"], device=device)
        self.audio2image_supcon = CrossModalSupCon(temperature=hparams["loss_temperature"], device=device)
    
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
        # send to device if required:
        if self.torch_device is not None:
            images = images.to(self.torch_device)
            image_labels = image_labels.to(self.torch_device)
            audios = audios.to(self.torch_device)
            audio_labels = audio_labels.to(self.torch_device)
        
        # forward pass:
        image_embeds, audio_embeds = self.forward(images, audios)

        # insert views dimension (1 view per sample for now) for compatibility with SupCon losses:     # TODO: Change this once dataloader returns tensors with n_views dimension.
        image_embeds = image_embeds.unsqueeze(dim=1)
        audio_embeds = audio_embeds.unsqueeze(dim=1)

        # compute intra-modal SupCon losses:
        image2image_loss = self.image2image_supcon(image_embeds, image_labels)
        audio2audio_loss = self.audio2audio_supcon(audio_embeds, audio_labels)
        # compute cross-modal SupCon losses:
        image2audio_loss = self.image2audio_supcon(image_embeds, image_labels, audio_embeds, audio_labels)
        audio2image_loss = self.audio2image_supcon(audio_embeds, audio_labels, image_embeds, image_labels)
        
        # compute weighted average of individual losses:
        loss_weights = self.hparams.loss_weights
        total_loss = (loss_weights["image2image"] * image2image_loss) + (loss_weights["audio2audio"] * audio2audio_loss) + (loss_weights["image2audio"] * image2audio_loss) + (loss_weights["audio2image"] * audio2image_loss)

        # log training losses:
        self.log("train/image2image_loss", image2image_loss)
        self.log("train/audio2audio_loss", audio2audio_loss)
        self.log("train/image2audio_loss", image2audio_loss)
        self.log("train/audio2image_loss", audio2image_loss)
        self.log("train/total_loss", total_loss)

        return total_loss
    
    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        """Validation step.

        Args:
            batch (dict): Batch dictionary with keys ("image", "image_label", "audio", "audio_label").
            batch_idx (int): Batch index (unused).
        
        Returns:
            loss (torch.Tensor): SupCon loss.
        """

        # unpack batch:
        images = batch["image"]
        image_labels = batch["image_label"]
        audios = batch["audio"]
        audio_labels = batch["audio_label"]
        # send to device if required:
        if self.torch_device is not None:
            images = images.to(self.torch_device)
            image_labels = image_labels.to(self.torch_device)
            audios = audios.to(self.torch_device)
            audio_labels = audio_labels.to(self.torch_device)
        
        # forward pass:
        image_embeds, audio_embeds = self.forward(images, audios)

        # insert views dimension (1 view per sample for now) for compatibility with SupCon losses:     # TODO: Change this once dataloader returns tensors with n_views dimension.
        image_embeds = image_embeds.unsqueeze(dim=1)
        audio_embeds = audio_embeds.unsqueeze(dim=1)

        # compute intra-modal SupCon losses:
        image2image_loss = self.image2image_supcon(image_embeds, image_labels)
        audio2audio_loss = self.audio2audio_supcon(audio_embeds, audio_labels)
        # compute cross-modal SupCon losses:
        image2audio_loss = self.image2audio_supcon(image_embeds, image_labels, audio_embeds, audio_labels)
        audio2image_loss = self.audio2image_supcon(audio_embeds, audio_labels, image_embeds, image_labels)

        # compute weighted average of individual losses:
        loss_weights = self.hparams.loss_weights
        total_loss = (loss_weights["image2image"] * image2image_loss) + (loss_weights["audio2audio"] * audio2audio_loss) + (loss_weights["image2audio"] * image2audio_loss) + (loss_weights["audio2image"] * audio2image_loss)

        # log validation losses:
        self.log("validation/image2image_loss", image2image_loss)
        self.log("validation/audio2audio_loss", audio2audio_loss)
        self.log("validation/image2audio_loss", image2audio_loss)
        self.log("validation/audio2image_loss", audio2image_loss)
        self.log("validation/total_loss", total_loss)

        return total_loss
    
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

