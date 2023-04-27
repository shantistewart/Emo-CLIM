"""Contains PyTorch Lightning LightningModule class for VCMR.
Only used for loading pretrained SampleCNN models.
Adapted from https://github.com/klean2050/VCMR/blob/master/vcmr/trainers/multimodal_learning.py."""


from torch import nn
from pytorch_lightning import LightningModule
from typing import Dict


class VCMR(LightningModule):
    """VCMR LightningModule class."""

    def __init__(self, args, encoder: nn.Module, video_params: Dict) -> None:
        """Initialization.

        Args:
            encoder (nn.Module): Audio encoder model.
            video_params (dict): Dictionary of video model parameters.
        
        Returns: None
        """

        super().__init__()
        self.save_hyperparameters(args)

        # dimensionality of representation:
        self.n_features = encoder.output_size

        # audio encoder:
        self.encoder = encoder

        # audio projector:
        self.audio_projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, self.hparams.projection_dim, bias=False)
        )
        # full audio model (encoder + projector):
        self.audio_model = nn.Sequential(self.encoder, self.audio_projector)

        # video temporal model:
        self.video_temporal = nn.LSTM(
            input_size=video_params["video_n_features"],
            hidden_size=video_params["video_n_features"],
            num_layers=video_params["video_lstm_n_layers"],
            batch_first=True,
            dropout=0.1
        )
        # video encoder:
        self.video_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(video_params["video_crop_length_sec"] * video_params["video_n_features"], self.n_features),
            nn.ReLU()
        )

        # video projector:
        self.video_projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, self.hparams.projection_dim, bias=False)
        )
        # video model (encoder + projector):
        self.video_model = nn.Sequential(self.video_encoder, self.video_projector)

