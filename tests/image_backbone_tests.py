import os
import torch
import os 
import sys
sys.path.append('../climur')
from models.image_backbones import clip_model 
from dataloaders.imac_images import IMAC_CLIP_Images
import clip
from PIL import Image

# dataset paths:
IMAC_IMAGES_DATA_ROOT = "/proj/systewar/datasets/IMAC/image_dataset"
IMAC_IMAGES_METADATA_FILE = "metadata_train.csv"
AUDIOSET_DATA_ROOT = "/proj/systewar/datasets/audioset_music_mood"
AUDIOSET_METADATA_FILE = "metadata_unbalanced_train.csv"

#instantiate clip here
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# create dataset:
imac_dataset = IMAC_CLIP_Images(
        root=IMAC_IMAGES_DATA_ROOT,
        preprocess=preprocess,
        metadata_file_name=IMAC_IMAGES_METADATA_FILE
    )

example_idx = 9
image, tag = imac_dataset[example_idx]

#passing through device
image=image.unsqueeze(0).to(device)

#instaiate clip model
img_model=clip_model(model,device,True)

#passing through clip model
feats=img_model(image)

print(feats.shape)

