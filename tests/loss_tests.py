"""Script for testing PyTorch loss classes."""


import torch
from torch import Tensor
from climur.losses.intramodal_supcon import IntraModalSupCon
from climur.losses.crossmodal_supcon import CrossModalSupCon


# script options:
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
# for SupCon losses:
temperature = 0.07
n_classes = 3
# input dimensions:
batch_size = 64
n_views = 1
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
    assert not torch.isnan(loss).item(), "Loss is NaN."

    # test forward pass with an empty positive index set:
    print("\n\nTesting forward pass with an empty positive index set...")
    embeds = torch.rand((batch_size, n_views, embed_dim)).to(device)
    labels = torch.zeros(batch_size).to(device)
    labels[0] = 1.0
    print("Embeddings size: {}".format(tuple(embeds.size())))
    loss = intra_supcon(embeds, labels)
    assert type(loss) == Tensor, "Loss is of incorrect data type."
    assert len(tuple(loss.size())) == 0, "Loss has incorrect shape."
    assert not torch.isnan(loss).item(), "Loss is NaN."


    # test CrossModalSupCon class:
    print("\n\n\nTesting CrossModalSupCon class:")

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
    assert not torch.isnan(loss).item(), "Loss is NaN."

    # test forward pass with an empty positive index set:
    print("\n\nTesting forward pass with an empty positive index set...")
    embeds_M1 = torch.rand((batch_size, n_views, embed_dim)).to(device)
    labels_M1 = torch.zeros(batch_size).to(device)
    labels_M1[0] = 1.0
    embeds_M2 = torch.rand((batch_size, n_views, embed_dim)).to(device)
    labels_M2 = torch.zeros(batch_size).to(device)
    labels_M2[0] = 2.0
    print("Embeddings size: {}".format(tuple(embeds_M1.size())))
    loss = cross_supcon(embeds_M1, labels_M1, embeds_M2, labels_M2)
    assert type(loss) == Tensor, "Loss is of incorrect data type."
    assert len(tuple(loss.size())) == 0, "Loss has incorrect shape."
    assert not torch.isnan(loss).item(), "Loss is NaN."
    assert loss.item() != 0.0, "Loss is 0."

    """
    # test forward pass with all empty positive index sets:
    print("\n\nTesting forward pass with all empty positive index sets...")
    embeds_M1 = torch.rand((batch_size, n_views, embed_dim)).to(device)
    labels_M1 = torch.zeros(batch_size).to(device)
    embeds_M2 = torch.rand((batch_size, n_views, embed_dim)).to(device)
    labels_M2 = torch.ones(batch_size).to(device)
    print("Embeddings size: {}".format(tuple(embeds_M1.size())))
    loss = cross_supcon(embeds_M1, labels_M1, embeds_M2, labels_M2)
    assert type(loss) == Tensor, "Loss is of incorrect data type."
    assert len(tuple(loss.size())) == 0, "Loss has incorrect shape."
    assert not torch.isnan(loss).item(), "Loss is NaN."
    assert loss.item() == 0.0, "Loss is not 0."
    """


    print("\n")

