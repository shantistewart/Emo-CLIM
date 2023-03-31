import torch
import clip
import torch.nn as nn

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
class clip_model(nn.Module):

    def __init__(self,model,device,freeze_img=True):
        super(clip_model, self).__init__()

        #model option, device 
        self.model= model
        self.device=device

        if freeze_img:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            image_features = self.model.encode_images(x)
        return image_features
