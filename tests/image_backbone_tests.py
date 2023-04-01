"""Script for testing image backbone models."""


import torch
import clip
import sys
sys.path.append('../climur')

from climur.models.image_backbones import CLIPModel
from climur.dataloaders.imac_images import IMACImages


# constants:
CLIP_EMBED_SIZE = 512

# dataset paths:
IMAC_IMAGES_DATA_ROOT = "/proj/systewar/datasets/IMAC/image_dataset"
IMAC_IMAGES_METADATA_FILE = "metadata_train.csv"

# script options:
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
example_idx = 9


if __name__ == "__main__":
    print("\n\n")

    # test CLIP model:
    print("Testing CLIP model:")

    # load CLIP model:
    model, preprocess = clip.load("ViT-B/32", device=device)

    # create image dataset:
    imac_dataset = IMACImages(
        root=IMAC_IMAGES_DATA_ROOT,
        metadata_file_name=IMAC_IMAGES_METADATA_FILE,
        preprocess=preprocess
    )
    # get example image:
    image, tag = imac_dataset[example_idx]
    image = image.unsqueeze(dim=0)
    image = image.to(device)

    # create CLIP wrapper model:
    img_model = CLIPModel(model)

    # test forward pass:
    print("\nTesting forward pass...")
    print("Input size: {}".format(tuple(image.size())))
    output = img_model(image)
    print("Output size: {}".format(tuple(output.size())))
    assert tuple(output.size()) == (1, CLIP_EMBED_SIZE), "Error with shape of forward pass output."

    print("\n")

