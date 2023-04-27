"""Various constants."""


# DATASET CONSTANTS:

# for AudioSet:
SAMPLE_RATE = 16000
# dictionary mapping original emotion tag names to shorter names:
AUDIOSET_EMOTION_TAGS_MAP = {
    "Happy music": "happy",
    "Funny music": "funny",
    "Sad music": "sad",
    "Tender music": "tender",
    "Exciting music": "exciting",
    "Angry music": "angry",
    "Scary music": "scary"
}

# default manual mapping from image dataset emotion tags to audio dataset emotion tags:
IMAGE2AUDIO_TAG_MAP = {
    "excitement": "exciting",
    "contentment": "happy",
    "amusement": "funny",     # TODO: not totally sure about this one (amusement may be closer to "entertaining" than "funny")
    "anger": "angry",
    "fear": "scary",
    "sadness": "sad"
}


# MODEL CONSTANTS:

# audio input lengths (in samples) for each audio backbone model:
SHORTCHUNK_INPUT_LENGTH     = 59049         # ~3.69 seconds
HARMONIC_CNN_INPUT_LENGTH   = 80000         # 5.0 seconds
SAMPLE_CNN_INPUT_LENGTH     = 98415         # ~6.15 seconds
CLAP_INPUT_LENGTH           = 480000        # 10.0 seconds

# output embedding dimensions for each image/audio backbone model:
CLIP_EMBED_DIM = 512
SHORTCHUNK_EMBED_DIM = 512     # assumes last_layer = "layer7" or later
HARMONIC_CNN_EMBED_DIM = 256     # assumes last_layer = "layer5" or later
SAMPLE_CNN_EMBED_DIM = 512
CLAP_EMBED_DIM = 512


# SAMPLECNN AND VCMR CONSTANTS:

# SampleCNN default paramters:
SAMPLE_CNN_DEFAULT_PARAMS = {
    "n_blocks": 9,
    "n_channels": [128, 128, 256, 256, 256, 256, 256, 512, 512],
    "output_size": 512,
    "conv_kernel_size": 3,
    "pool_size": 3,
    "activation": "relu",
    "first_block_params": {
        "out_channels": 128,
        "conv_size": 5
    },
    "input_size": SAMPLE_CNN_INPUT_LENGTH
}

# VCMR default parameters (only used to load VCMR model from checkpoint):
VCMR_DEFAULT_PARAMS = {
    "video_crop_length_sec": 6,
    "video_n_features": 512,
    "video_lstm_n_layers": 2
}

