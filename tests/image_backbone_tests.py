"""Script for testing image backbone models."""


import os
import torch
import torchinfo
import clip
from climur.models.image_backbones import CLIPModel
from climur.utils.constants import CLIP_EMBED_DIM


# constants:
IMAGE_CHANNELS = 3
IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224

# script options:
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
batch_size = 16
# for model summaries:
model_summaries = True
if model_summaries:
    summaries_dir = "tests/model_summaries"
    model_summary_info = ["input_size", "output_size", "num_params"]


if __name__ == "__main__":
    print("\n\n")

    # test CLIP model:
    print("Testing CLIP model:")

    # load CLIP model:
    orig_model, preprocess = clip.load("ViT-B/32", device=device)
    # create CLIP wrapper model:
    wrap_model = CLIPModel(orig_model)
    wrap_model.to(device)

    # test forward pass:
    print("\nTesting forward pass...")
    wrap_model.eval()
    x = torch.rand((batch_size, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH))
    x = x.to(device)
    print("Input size: {}".format(tuple(x.size())))
    output = wrap_model(x)
    print("Output size: {}".format(tuple(output.size())))
    assert tuple(output.size()) == (batch_size, CLIP_EMBED_DIM), "Error with shape of forward pass output."

    # create model summary, if selected:
    if model_summaries:
        print("\nCreating model summary...")
        os.makedirs(summaries_dir, exist_ok=True)
        clip_model_summary = str(torchinfo.summary(
            wrap_model,
            input_size=(batch_size, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH),
            col_names=model_summary_info,
            depth=3,
            verbose=0
        ))
        clip_model_summary_file = os.path.join(summaries_dir, "clip_model_summary.txt")
        with open(clip_model_summary_file, "w") as text_file:
            text_file.write(clip_model_summary)
    
    print("\n")

