"""Script for testing PyTorch loss classes."""


import torch
from torch import Tensor
from climur.losses.intramodal_supcon import IntramodalSupCon


# script options:
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
# for multimodal SupCon loss:
temperature = 0.07
base_temperature = 0.07
n_classes = 3
# input dimensions:
batch_size = 16
n_views = 2
embed_dim = 128


if __name__ == "__main__":
    print("\n\n")


    # test IntramodalSupCon class:
    print("Testing IntramodalSupCon class:")

    # create loss:
    criterion = IntramodalSupCon(
        temperature=temperature,
        base_temperature=base_temperature
    )
    criterion.to(device)

    # test forward pass:
    print("\nTesting forward pass...")
    embeds = torch.rand((batch_size, n_views, embed_dim)).to(device)
    labels = torch.randint(low=0, high=n_classes, size=(batch_size, )).to(device)
    print("Embeddings size: {}".format(tuple(embeds.size())))
    loss = criterion(embeds, labels)
    assert type(loss) == Tensor, "Loss is of incorrect data type."
    assert len(tuple(loss.size())) == 0, "Loss has incorrect shape."


    print("\n")

