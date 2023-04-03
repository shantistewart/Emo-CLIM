"""PyTorch model classes for image backbone models."""


import torch.nn as nn
from torch import Tensor


class CLIPModel(nn.Module):
    """Wrapper class for CLIP model.

    Attributes:
        model (nn.Module): CLIP model.
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialization.

        Args:
            model (nn.Module): CLIP model.
        
        Returns: None
        """

        super(CLIPModel, self).__init__()

        # save model:
        self.model = model
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Preprocessed images.
                shape: (batch_size, image_channels, image_height, image_width)
        
        Returns:
            image_feats (Tensor): Image features (embeddings).
                shape: (batch_size, embed_dim)
        """

        # encode images:
        image_feats = self.model.encode_image(x)

        return image_feats

