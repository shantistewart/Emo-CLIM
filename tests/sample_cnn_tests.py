"""Script for testing SampleCNN model."""


import os
import torch
import torchinfo
from copy import deepcopy
import warnings
from climur.models.audio_backbones import SampleCNNEmbeddings
from climur.models.vcmr_trainer import VCMR
from climur.utils.constants import SAMPLE_CNN_INPUT_LENGTH, SAMPLE_CNN_DEFAULT_PARAMS, VCMR_DEFAULT_PARAMS


# constants:
AUDIO_LENGTH = SAMPLE_CNN_INPUT_LENGTH     # ~6.15 seconds

# script options:
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
batch_size = 16
# model options:
pretrained_full_model_path = "/proj/systewar/pretrained_models/VCMR/multimodal/multimodal_model_1.ckpt"
# for model summaries:
model_summaries = True
if model_summaries:
    summaries_dir = "tests/model_summaries"
    model_summary_info = ["input_size", "output_size", "num_params"]


if __name__ == "__main__":
    print("\n\n")
    # suppress warnings:
    warnings.filterwarnings("ignore")

    # test SampleCNNEmbeddings model:
    print("Testing SampleCNNEmbeddings model:")

    # create (empty) SampleCNNEmbeddings model:
    sample_cnn = SampleCNNEmbeddings(
        params=SAMPLE_CNN_DEFAULT_PARAMS
    )

    # load pretrained VCMR model:
    full_model = VCMR.load_from_checkpoint(
        pretrained_full_model_path,
        encoder=sample_cnn,
        video_params=VCMR_DEFAULT_PARAMS
    )
    # extract SampleCNNEmbeddings component:
    embed_model = deepcopy(full_model.encoder)
    embed_model.requires_grad_(True)     # make sure all weights are unfrozen
    embed_model.to(device)


    # test forward pass:
    print("\nTesting forward pass...")
    x = torch.rand((batch_size, AUDIO_LENGTH))
    x = x.to(device)
    print("Input size: {}".format(tuple(x.size())))
    output = embed_model(x)
    print("Output size: {}".format(tuple(output.size())))

    # tests:
    assert len(tuple(output.size())) == 2, "Error with shape of forward pass output."
    assert output.size(dim=0) == batch_size, "Error with shape of forward pass output."

    # create model summary, if selected:
    if model_summaries:
        print("\nCreating model summary...")
        os.makedirs(summaries_dir, exist_ok=True)
        embed_model_summary = str(torchinfo.summary(
            embed_model,
            input_size=(batch_size, AUDIO_LENGTH),
            col_names=model_summary_info,
            depth=2,
            verbose=0
        ))
        embed_model_summary_file = os.path.join(summaries_dir, "sample_cnn_embed_model_summary.txt")
        with open(embed_model_summary_file, "w") as text_file:
            text_file.write(embed_model_summary)
    
    print("\n")

