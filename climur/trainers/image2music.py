"""PyTorch Lightning LightningModule class for image-music supervised contrastive learning."""


from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Tuple, Any


class Image2Music(LightningModule):
    """LightningModule class for image-music supervised contrastive learning.

    Attributes:
        image_backbone (nn.Module): Image backbone model.
        audio_backbone (nn.Module): Audio backbone model.
        image_projector (nn.Module): Image projector model.
        audio_projector (nn.Module): Audio projector model.
        joint_embed_dim (int): Dimension of joint embedding space.

    """

    def __init__(     # TODO: Add a normalization param.
            self,
            image_backbone: nn.Module,
            audio_backbone: nn.Module, 
            joint_embed_dim: int,
            hparams: Dict,
            image_embed_dim: int = 512,
            audio_embed_dim: int = 256,
            freeze_image_backbone: bool = False,
            freeze_audio_backbone: bool = False
        ) -> None:
        """Initialization.

        Args:
            image_backbone (nn.Module): Image backbone model.
            audio_backbone (nn.Module): Audio backbone model.
            joint_embed_dim (int): Dimension of joint embedding space.
            hparams (dict): Dictionary of hyperparameters.
            image_embed_dim (int): Dimension of image embeddings (output of image backbone model).
            audio_embed_dim (int): Dimension of audio embeddings (output of audio backbone model).
            freeze_image_backbone (bool): Selects whether to freeze image backbone model.
            freeze_audio_backbone (bool): Selects whether to freeze audio backbone model.
        
        Returns: None
        """

        super().__init__()
        # save hyperparameters (saves to self.hparams):
        self.save_hyperparameters(hparams)

        # save image backbone model and freeze if selected:
        self.image_backbone = image_backbone
        if freeze_image_backbone:
            self.image_backbone.requires_grad_(False)
        # save audio backbone model and freeze if selected:
        self.audio_backbone = audio_backbone
        if freeze_audio_backbone:
            self.audio_backbone.requires_grad_(False)
        
        # save other params:
        self.joint_embed_dim = joint_embed_dim

        # create projectors:
        projector_hidden_dim = max(image_embed_dim, audio_embed_dim)
        self.image_projector = nn.Sequential(
            nn.Linear(in_features=image_embed_dim, out_features=projector_hidden_dim, bias=False),     # TODO: Maybe change to bias=True?
            nn.ReLU(),
            nn.Linear(in_features=projector_hidden_dim, out_features=joint_embed_dim, bias=False)
        )
        self.audio_projector = nn.Sequential(
            nn.Linear(in_features=audio_embed_dim, out_features=projector_hidden_dim, bias=False),     # TODO: Maybe change to bias=True?
            nn.ReLU(),
            nn.Linear(in_features=projector_hidden_dim, out_features=joint_embed_dim, bias=False)
        )
    
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

        # project to joint multimodal embedding space:
        image_embeds = self.image_projector(images_enc)
        audio_embeds = self.audio_projector(audios_enc)

        return image_embeds, audio_embeds
    
    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        """Training step.

        Args:

        Returns:
        """

        pass

    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        """Validation step.

        Args:

        Returns:
        """

        pass

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

