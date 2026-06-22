import torch
import open_clip
from mobileclip.modules.common.mobileone import reparameterize_model
from transformers import AutoImageProcessor, AutoModel

class TextEncoder():
    def __init__(self, model_id="meta/ViT-S16", device='cpu'):
        self.model_id = model_id
        self.device = device

        if not (self.model_id == "MobileCLIP2-S3" or self.model_id == "MobileCLIP2-S4" or self.model_id.endswith("L-14")):
            model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}
        self.model, _, preprocess = open_clip.create_model_and_transforms(model_id, pretrained='./pretrained/mobileclip2_s0.pt',device=self.device, **model_kwargs)
        self.tokenizer = open_clip.get_tokenizer(self.model_id)

        self.model.eval()

        self.model = reparameterize_model(self.model)

        for p in self.model.parameters():
            p.requires_grad = False

    def __call__(self, texts):
        text = self.tokenizer(texts).to(self.device)
        with torch.no_grad(), torch.amp.autocast(device_type=self.device):
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features.float()

class ImageEncoder():
    def __init__(self, model_id="", layers=None, device="cuda"):
        self.model_id = model_id
        self.layers = layers
        self.device = device

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

    def __call__(self, images):
        inputs = {"pixel_values": images.to(self.device)}

        with torch.no_grad():
            hidden_states = self.model(**inputs, output_hidden_states=True).hidden_states

        selected = [hidden_states[layer] for layer in self.layers]
        cls = torch.stack([h[:, 0, :] for h in selected], dim=1)
        patches = torch.stack([h[:, 5:, :] for h in selected], dim=1)

        return cls, patches
