"""Script for testing Short-Chunk CNN ResNet model."""


import os
import torch
import torchinfo
import warnings
from climur.models.audio_backbones import ShortChunkCNNEmbeddings
from climur.models.audio_model_components import ShortChunkCNN_Res
from climur.utils.constants import SHORTCHUNK_INPUT_LENGTH


# constants:
AUDIO_LENGTH = SHORTCHUNK_INPUT_LENGTH     # ~3.69 seconds
N_CLASSES = 50

# script options:
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
batch_size = 16
# model options:
pretrained_full_model_path = "/proj/systewar/pretrained_models/music_tagging/msd/short_chunk_resnet/best_model.pth"
last_layer_embed = "layer7"
pool_type = "max"
# for model summaries:
model_summaries = True
if model_summaries:
    summaries_dir = "tests/model_summaries"
    model_summary_info = ["input_size", "output_size", "num_params"]


if __name__ == "__main__":
    print("\n\n")
    # suppress warnings:
    warnings.filterwarnings("ignore")


    # test full Short-Chunk CNN ResNet model:
    print("Testing full Short-Chunk CNN ResNet model:")

    # load pretrained full Short-Chunk CNN ResNet model:
    full_model = ShortChunkCNN_Res()
    full_model.load_state_dict(torch.load(pretrained_full_model_path, map_location=device))
    full_model.to(device)
    
    # test forward pass:
    print("Testing forward pass...")
    x = torch.rand((batch_size, AUDIO_LENGTH))
    x = x.to(device)
    output = full_model(x)
    assert tuple(output.size()) == (batch_size, N_CLASSES), "Error with shape of forward pass output."

    # create model summary, if selected:
    if model_summaries:
        print("Creating model summary...")
        os.makedirs(summaries_dir, exist_ok=True)
        full_model_summary = str(torchinfo.summary(
            full_model,
            input_size=(batch_size, AUDIO_LENGTH),
            col_names=model_summary_info,
            depth=1,
            verbose=0
        ))
        full_model_summary_file = os.path.join(summaries_dir, "shortchunk_full_model_summary.txt")
        with open(full_model_summary_file, "w") as text_file:
            text_file.write(full_model_summary)
    

    # test ShortChunkCNNEmbeddings model:
    print("\n\nTesting ShortChunkCNNEmbeddings model:")
    
    # create ShortChunkCNNEmbeddings model:
    x = torch.rand((batch_size, AUDIO_LENGTH))
    x = x.to(device)
    embed_model = ShortChunkCNNEmbeddings(
        full_model,
        sample_input=x,
        last_layer=last_layer_embed,
        pool_type=pool_type
    )
    embed_model.to(device)

    print("\nTesting forward pass...")
    # test forward pass:
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
        embed_model_summary_file = os.path.join(summaries_dir, "shortchunk_embed_model_summary.txt")
        with open(embed_model_summary_file, "w") as text_file:
            text_file.write(embed_model_summary)
    
    print("\n")

