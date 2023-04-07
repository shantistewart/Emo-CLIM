"""PyTorch model classes for audio backbone models."""


import torch.nn as nn
from torch import Tensor


class HarmonicCNNEmbeddings(nn.Module):
    """Wrapper class to extract embeddings from Harmonic CNN. Reference: "Data-Driven Harmonic Filters for Audio Representation Learning", Won et al., 2020.
    
    Attributes:
        model (nn.Module): Model to extract embeddings from Harmonic CNN.
        global_pool (nn.Module): Global pooling layer.
        shrink_freq (bool): Selects whether to shrink and squeeze frequency dimension.
        shrink_time (bool): Selects whether to shrink and squeeze time dimension.
    """

    def __init__(self, full_model: nn.Module, sample_input: Tensor, last_layer: str = "layer7", shrink_freq: bool = True, shrink_time: bool = True, pool_type: str = "max") -> None:
        """Initialization.

        Args:
            full_model (nn.Module): Full Harmonic CNN model (pretrained).
            sample_input (Tensor): Sample raw audio input, for determining shape of pooling layer.
                shape: (batch_size, audio_length)
            last_layer (str): Name of last layer (child) to include (stop at) for extracting embeddings.
            shrink_freq (bool): Selects whether to shrink and squeeze frequency dimension.
            shrink_time (bool): Selects whether to shrink and squeeze time dimension.
            pool_type (str): Type of global pooling to use ("average" or "max").
        
        Returns: None
        """

        super(HarmonicCNNEmbeddings, self).__init__()

        # validate parameters:
        if len(tuple(sample_input.size())) != 2:
            raise ValueError("Invalid sample input shape.")
        children_names = [name for name, _ in full_model.named_children()]
        if last_layer not in children_names:
            raise ValueError("Invalid last_layer value.")
        if pool_type not in ["average", "max"]:
            raise ValueError("Invalid pooling type.")
        
        # save parameters:
        self.shrink_freq = shrink_freq
        self.shrink_time = shrink_time
        
        # extract part of full model for use in extracting embeddings:
        children_list = []
        for name, child in full_model.named_children():
            # save child:
            children_list.append(child)
            # stop at last desired child:
            if name == last_layer:
                break
        
        # convert list to nn.Sequential() object:
        self.model = nn.Sequential(*children_list)

        # shrink frequency and/or time dimensions, using global pooling:
        sample_output = self.model(sample_input)
        if self.shrink_freq and self.shrink_time:
            kernel_size = (sample_output.size(dim=-2), sample_output.size(dim=-1))
        elif self.shrink_freq and not self.shrink_time:
            kernel_size = (sample_output.size(dim=-2), 1)
        elif not self.shrink_freq and self.shrink_time:
            kernel_size = (1, sample_output.size(dim=-1))
        else:
            kernel_size = None     # no global pooling
        
        # use average or max global pooling, if necessary:
        if kernel_size is None or (sample_output.size(dim=-2) == 1 and sample_output.size(dim=-1) == 1):     # no shrinking is requested or (freq_dim, time_dim) are both already size 1
            self.global_pool = None     # no global pooling
        else:
            if pool_type == "average":
                self.global_pool = nn.AvgPool2d(kernel_size=kernel_size)
            elif pool_type == "max":
                self.global_pool = nn.MaxPool2d(kernel_size=kernel_size)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Raw audio input.
                shape: (batch_size, audio_length)
        
        Returns:
            x (Tensor): Model output.
                shape:
                    if shrink_freq == True and shrink_time == True: (batch_size, embed_dim)
                    else if shrink_freq == True and shrink_time == False: (batch_size, embed_dim, time_dim)
                    else if shrink_freq == False and shrink_time == True: (batch_size, embed_dim, freq_dim)
                    else shrink_freq == False and shrink_time == False: (batch_size, embed_dim, freq_dim, time_dim)
        """

        # forward pass through main layers:
        x = self.model(x)

        # shrink frequency and/or time dimensions (using global pooling), if necessary:
        if self.global_pool is not None:
            x = self.global_pool(x)
        
        # squeeze frequency dimension:
        if self.shrink_freq:
            assert x.size(dim=-2) == 1, "Unexpected tensor shape in forward pass."
            x = x.squeeze(dim=-2)
        # squeeze time dimension:
        if self.shrink_time:
            assert x.size(dim=-1) == 1, "Unexpected tensor shape in forward pass."
            x = x.squeeze(dim=-1)
        
        return x

