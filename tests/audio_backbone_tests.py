"""Script for testing audio backbone models."""


import os
import torch
import torchinfo
from climur.models.audio_backbones import HarmonicCNNEmbeddings
from climur.models.audio_model_components import HarmonicCNN


# constants:
SAMPLE_RATE = 16000
AUDIO_LENGTH = 5 * SAMPLE_RATE     # 5.0 seconds
HCNN_N_CLASSES = 50

# script options:
device = torch.device("cuda")
pretrained_harmonic_cnn_path = "/proj/systewar/pretrained_models/music_tagging/msd/harmonic_cnn/best_model.pth"
# for HarmonicCNNEmbeddings model:
last_layer_embed = "layer7"
shrink_freq = True
shrink_time = True
pool_type = "max"
# for model summaries:
batch_size = 16
model_summaries = True
if model_summaries:
    summaries_dir = "tests/model_summaries"
    model_summary_info = ["input_size", "output_size", "num_params"]


if __name__ == "__main__":
    print("\n\n")

    # test full HarmonicCNN model:
    print("Testing full Harmonic CNN model:")

    # load pretrained full Harmonic CNN model:
    full_model = HarmonicCNN()
    full_model.load_state_dict(torch.load(pretrained_harmonic_cnn_path, map_location=device))
    full_model.to(device)

    print("Testing forward pass...")
    # test forward pass:
    full_model.eval()
    x = torch.rand((batch_size, AUDIO_LENGTH))
    x = x.to(device)
    output = full_model(x)
    assert tuple(output.size()) == (batch_size, HCNN_N_CLASSES), "Error with shape of forward pass output."
    
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
        full_model_summary_file = os.path.join(summaries_dir, "harmonic_cnn_full_model_summary.txt")
        with open(full_model_summary_file, "w") as text_file:
            text_file.write(full_model_summary)
    

    # test HarmonicCNNEmbeddings model:
    print("\n\nTesting HarmonicCNNEmbeddings model:")

    # create HarmonicCNNEmbeddings model:
    x = torch.rand((batch_size, AUDIO_LENGTH))
    x = x.to(device)
    embed_model = HarmonicCNNEmbeddings(
        full_model,
        sample_input=x,
        last_layer=last_layer_embed,
        shrink_freq=shrink_freq,
        shrink_time=shrink_time,
        pool_type=pool_type
    )
    embed_model.to(device)

    print("\nTesting forward pass...")
    # test forward pass:
    embed_model.eval()
    print("Input size: {}".format(tuple(x.size())))
    output = embed_model(x)
    print("Output size: {}".format(tuple(output.size())))

    # tests:
    if shrink_freq == True and shrink_time == True:
        assert len(tuple(output.size())) == 2, "Error with shape of forward pass output."
    elif (shrink_freq == True and shrink_time == False) or (shrink_freq == False and shrink_time == True):
        assert len(tuple(output.size())) == 3, "Error with shape of forward pass output."
    else:
        assert len(tuple(output.size())) == 4, "Error with shape of forward pass output."
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
        embed_model_summary_file = os.path.join(summaries_dir, "harmonic_cnn_embed_model_summary.txt")
        with open(embed_model_summary_file, "w") as text_file:
            text_file.write(embed_model_summary)
    
    print("\n")
