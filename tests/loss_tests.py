"""Script for testing PyTorch loss classes."""


import torch
from torch import Tensor
from climur.losses.intramodal_supcon import IntraModalSupCon
from climur.losses.crossmodal_supcon import CrossModalSupCon


# script options:
device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")
# for SupCon losses:
temperature = 0.07
n_classes = 3
# input dimensions:
batch_size = 128
n_views = 2
embed_dim = 128


if __name__ == "__main__":
    print("\n\n")


    # test IntraModalSupCon class:
    print("Testing IntraModalSupCon class:")

    # create loss:
    intra_supcon = IntraModalSupCon(temperature=temperature, device=device)
    intra_supcon.to(device)

    # test forward pass:
    print("\nTesting forward pass...")
    embeds = torch.rand((batch_size, n_views, embed_dim)).to(device)
    labels = torch.randint(low=0, high=n_classes, size=(batch_size, )).to(device)
    print("Embeddings size: {}".format(tuple(embeds.size())))
    loss = intra_supcon(embeds, labels)
    assert type(loss) == Tensor, "Loss is of incorrect data type."
    assert len(tuple(loss.size())) == 0, "Loss has incorrect shape."


    # test CrossModalSupCon class:
    print("\n\nTesting CrossModalSupCon class:")

    # create loss:
    cross_supcon = CrossModalSupCon(temperature=temperature)
    cross_supcon.to(device)

    # test forward pass:
    print("\nTesting forward pass...")
    embeds_M1 = torch.rand((batch_size, n_views, embed_dim)).to(device)
    labels_M1 = torch.randint(low=0, high=n_classes, size=(batch_size, )).to(device)
    embeds_M2 = torch.rand((batch_size, n_views, embed_dim)).to(device)
    labels_M2 = torch.randint(low=0, high=n_classes, size=(batch_size, )).to(device)
    print("Embeddings size: {}".format(tuple(embeds_M1.size())))
    loss = cross_supcon(embeds_M1, labels_M1, embeds_M2, labels_M2)
    assert type(loss) == Tensor, "Loss is of incorrect data type."
    assert len(tuple(loss.size())) == 0, "Loss has incorrect shape."


    print("\n")

