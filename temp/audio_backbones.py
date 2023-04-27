"""PyTorch model classes for audio backbone models."""


import torch.nn as nn
from torch import Tensor
import numpy as np
from typing import List, Dict


# audio input lengths (in samples) each model:
SHORTCHUNK_INPUT_LENGTH     = 59049         # ~3.69 seconds
HARMONIC_CNN_INPUT_LENGTH   = 80000         # 5.0 seconds
SAMPLE_CNN_INPUT_LENGTH     = 98415         # ~6.15 seconds
CLAP_INPUT_LENGTH           = 480000        # 10.0 seconds
# output embedding dimensions for each model:
SHORTCHUNK_EMBED_DIM = 512     # assumes last_layer = "layer7" or later
HARMONIC_CNN_EMBED_DIM = 256     # assumes last_layer = "layer5" or later
SAMPLE_CNN_EMBED_DIM = 512
CLAP_EMBED_DIM = 512


class ShortChunkCNNEmbeddings(nn.Module):
    """Wrapper class to extract embeddings from Short-Chunk CNN ResNet model. Reference: "Evaluation of CNN-based Automatic Music Tagging Models", Won et al., 2020.

    Attributes:
        spec_layers (nn.Module): Spectrogram layers.
        spec_batchnorm (nn.Module): Spectrogram BatchNorm layer.
        cnn_layers (nn.Module): CNN layers.
        global_pool (nn.Module): Global pooling layer.
    """

    def __init__(self, full_model: nn.Module, sample_input: Tensor, last_layer: str = "layer7", pool_type: str = "max") -> None:
        """Initialization.

        Args:
            full_model (nn.Module): Full Short-Chunk CNN ResNet model (pretrained).
            sample_input (Tensor): Sample raw audio input, for determining shape of pooling layer.
                shape: (batch_size, audio_length)
            last_layer (str): Name of last layer to include (stop at) for extracting embeddings.
            pool_type (str): Type of global pooling to use ("average" or "max").
        
        Returns: None
        """

        super(ShortChunkCNNEmbeddings, self).__init__()

        # names of all CNN layers in full model:
        all_cnn_layer_names = ["layer1", "layer2", "layer3", "layer4", "layer5", "layer6", "layer7"]

        # validate parameters:
        if len(tuple(sample_input.size())) != 2:
            raise ValueError("Invalid sample input shape.")
        if last_layer not in all_cnn_layer_names:
            raise ValueError("Invalid last_layer value.")
        if pool_type not in ["average", "max"]:
            raise ValueError("Invalid pooling type.")
        
        # extract spectrogram layers:
        spec_layers_list = [full_model.spec, full_model.to_db]
        self.spec_layers = nn.Sequential(*spec_layers_list)
        # extract spectrogram BatchNorm layer:
        self.spec_batchnorm = full_model.spec_bn

        # extract desired CNN layers:
        cnn_layers_list = []
        for name, child in full_model.named_children():
            if name in all_cnn_layer_names:
                cnn_layers_list.append(child)
            # stop at last desired layer:
            if name == last_layer:
                break
        self.cnn_layers = nn.Sequential(*cnn_layers_list)

        # run sample input through network to determine output shape (before global pooling):
        sample_x = self.spec_layers(sample_input)
        sample_x = sample_x.unsqueeze(dim=1)
        sample_x = self.spec_batchnorm(sample_x)
        sample_output = self.cnn_layers(sample_x)
        
        # shrink frequency and time dimensions, using global pooling:
        if sample_output.size(dim=-2) == 1 and sample_output.size(dim=-1) == 1:     # global pooling not required, since (freq_dim, time_dim) are both already size 1
            self.global_pool = None     # no global pooling
        else:
            kernel_size = (sample_output.size(dim=-2), sample_output.size(dim=-1))
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
            x (Tensor): Embeddings.
                shape: (batch_size, embed_dim)
        """

        # spectrogram layers:
        x = self.spec_layers(x)
        x = x.unsqueeze(dim=1)
        x = self.spec_batchnorm(x)

        # CNN layers:
        x = self.cnn_layers(x)

        # shrink frequency and time dimensions (using global pooling), if necessary:
        if self.global_pool is not None:
            x = self.global_pool(x)
        # squeeze frequency and time dimensions:
        assert x.size(dim=-2) == 1, "Frequency dimension not size 1 in forward pass."
        x = x.squeeze(dim=-2)
        assert x.size(dim=-1) == 1, "Time dimension not size 1 in forward pass."
        x = x.squeeze(dim=-1)

        return x


class HarmonicCNNEmbeddings(nn.Module):
    """Wrapper class to extract embeddings from Harmonic CNN. Reference: "Data-Driven Harmonic Filters for Audio Representation Learning", Won et al., 2020.

    Attributes:
        model (nn.Module): Model to extract embeddings from Harmonic CNN.
        global_pool (nn.Module): Global pooling layer.
    """

    def __init__(self, full_model: nn.Module, sample_input: Tensor, last_layer: str = "layer7", pool_type: str = "max") -> None:
        """Initialization.

        Args:
            full_model (nn.Module): Full Harmonic CNN model (pretrained).
            sample_input (Tensor): Sample raw audio input, for determining shape of pooling layer.
                shape: (batch_size, audio_length)
            last_layer (str): Name of last layer to include (stop at) for extracting embeddings.
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

        # run sample input through network to determine output shape (before global pooling):
        sample_output = self.model(sample_input)
        # shrink frequency and time dimensions, using global pooling:
        if sample_output.size(dim=-2) == 1 and sample_output.size(dim=-1) == 1:     # global pooling not required, since (freq_dim, time_dim) are both already size 1
            self.global_pool = None     # no global pooling
        else:
            kernel_size = (sample_output.size(dim=-2), sample_output.size(dim=-1))
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
            x (Tensor): Embeddings.
                shape: (batch_size, embed_dim)
        """

        # forward pass through main layers:
        x = self.model(x)

        # shrink frequency and time dimensions (using global pooling), if necessary:
        if self.global_pool is not None:
            x = self.global_pool(x)
        # squeeze frequency and time dimensions:
        assert x.size(dim=-2) == 1, "Frequency dimension not size 1 in forward pass."
        x = x.squeeze(dim=-2)
        assert x.size(dim=-1) == 1, "Time dimension not size 1 in forward pass."
        x = x.squeeze(dim=-1)

        return x


class SampleCNNEmbeddings(nn.Module):
    """Configurable PyTorch class for SampleCNN.

    Attributes:
        n_blocks (int): Number of middle convolutional blocks (equal to n_total_blocks - 2).
        output_size (int): Number of output channels for last block.
        pool_size (int): Size of pooling kernel and pooling stride for middle blocks.
        input_size (int): Size (length) of input.
        all_blocks (nn.Sequential: nn.Sequential object for all blocks.
    """

    def __init__(self, n_blocks: int, n_channels: List, output_size: int, conv_kernel_size: int, pool_size: int, activation: str = "relu", first_block_params: Dict = None, input_size: int = None) -> None:
        """Initialization.

        Args:
            n_blocks (int): Number of middle convolutional blocks (equal to n_total_blocks - 2).
            n_channels (list): List of number of (output) channels for middle blocks.
                length: n_blocks
            output_size (int): Number of output channels for last block.
            conv_kernel_size (int): Size of convolutional kernel for middle blocks.
                Convolution stride is equal to 1 for middle blocks.
            pool_size (int): Size of pooling kernel and pooling stride for middle blocks.
                Kernel size is equal to stride to ensure even division of input size.
            activation (str): Type of activation to use for all blocks ("relu" or "leaky_relu").
            first_block_params (dict): Dictionary describing first block, with keys/values:
                out_channels (int): Number of output channels.
                conv_size (int): Size of convolutional kernel and convolution stride (kernel size is equal to stride to ensure even division of input size).
            input_size (int): Size (length) of input.
        
        Returns: None
        """

        super(SampleCNNEmbeddings, self).__init__()

        # validate parameters:
        assert len(n_channels) == n_blocks, "Length of n_channels doesn't match n_blocks."
        assert activation == "relu" or activation == "leaky_relu", "Invalid activation type."

        # save attributes:
        self.n_blocks = n_blocks
        self.output_size = output_size
        self.pool_size = pool_size

        # if items of first_block are unspecified, set to default values:
        if first_block_params is None:
            first_block_params = {}
        if first_block_params.get("out_channels") is None:
            first_block_params["out_channels"] = n_channels[0]
        if first_block_params.get("conv_size") is None:
            first_block_params["conv_size"] = conv_kernel_size
        
        # if input_size is not None, validate input_size:
        if input_size is not None:
            assert input_size == first_block_params["conv_size"] * np.power(pool_size, n_blocks), "Input size is incompatible with network architecture."
        # else infer from network architecture:
        else:
            input_size == first_block_params["conv_size"] * np.power(pool_size, n_blocks)
        self.input_size = input_size


        # create first block:
        first_block = nn.Sequential(
            nn.Conv1d(1, first_block_params["out_channels"], kernel_size=first_block_params["conv_size"], stride=first_block_params["conv_size"], padding=0),
            nn.BatchNorm1d(first_block_params["out_channels"])
        )
        if activation == "relu":
            first_block.append(nn.ReLU())
        elif activation == "leaky_relu":
            first_block.append(nn.LeakyReLU())
        
        # create middle blocks:
        middle_blocks = []
        for i in range(n_blocks):
            # get number of input and output channels for convolutional layer:
            if i == 0:
                in_channels = first_block_params["out_channels"]
            else:
                in_channels = n_channels[i-1]
            out_channels = n_channels[i]
            
            # create block:
            block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=conv_kernel_size, stride=1, padding="same"),
                nn.BatchNorm1d(out_channels)
            )
            if activation == "relu":
                block.append(nn.ReLU())
            elif activation == "leaky_relu":
                block.append(nn.LeakyReLU())
            block.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_size, padding=0))
            # append block to list:
            middle_blocks.append(block)
        
        # create last block:
        last_block = nn.Sequential(
            nn.Conv1d(n_channels[n_blocks-1], output_size, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(output_size)
        )
        if activation == "relu":
            last_block.append(nn.ReLU())
        elif activation == "leaky_relu":
            last_block.append(nn.LeakyReLU())
        
        # concatenate all blocks and convert list to nn.Sequential() object:
        all_blocks_list = [first_block] + middle_blocks + [last_block]
        self.all_blocks = nn.Sequential(*all_blocks_list)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Raw audio input.
                shape: (batch_size, audio_length)
        
        Returns:
            x (Tensor): Embeddings.
                shape: (batch_size, embed_dim)
        """

        # forward pass through all blocks:
        output = self.all_blocks(x)
        # remove temporal dimension (since it is size 1):
        output = output.squeeze(dim=-1)

        return output


class CLAPEmbeddings(nn.Module):
    """Wrapper class to extract embeddings from CLAP. Reference: "Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation", Wu et al., 2023.

    Attributes:
        model (nn.Module): Model to extract embeddings from CLAP Pretrained model.
        global_pool (nn.Module): Global pooling layer.
    """

    def __init__(self, full_model: nn.Module, sample_input: Tensor, last_layer: str = "layer7", pool_type: str = "max") -> None:
        """Initialization.

        Args:
            full_model (nn.Module): Full Harmonic CNN model (pretrained).
            sample_input (Tensor): Sample raw audio input, for determining shape of pooling layer.
                shape: (batch_size, audio_length)
            last_layer (str): Name of last layer to include (stop at) for extracting embeddings.
            pool_type (str): Type of global pooling to use ("average" or "max").
        
        Returns: None
        """

        super(CLAPEmbeddings, self).__init__()

        # validate parameters:
        if len(tuple(sample_input.size())) != 2:
            raise ValueError("Invalid sample input shape.")
        if pool_type not in ["average", "max"]:
            raise ValueError("Invalid pooling type.")
        
        # convert list to nn.Sequential() object:
        self.model = full_model

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Raw audio input.
                shape: (batch_size, audio_length)
        
        Returns:
            x (Tensor): Embeddings.
                shape: (batch_size, embed_dim)
        """
        # forward pass through main layers:
        x = self.model.get_audio_embedding_from_data(
            x=x.float(), use_tensor=True
        )
        return x

