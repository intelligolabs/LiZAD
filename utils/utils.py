import json

import torch

from backbones.encoders import TextEncoder
from tqdm import tqdm
import os
import numpy as np

def prompt_generator(dataset):
    class_names = dataset.get_class_names()
    prompts = dataset.get_prompts()

    generated_prompts = {'normal': {}, 'abnormal': {}}

    for class_name in class_names:
        generated_prompts['normal'][class_name] = []
        generated_prompts['abnormal'][class_name] = []

    for template in prompts["prompt_templates"]:
        for class_name in class_names:
            for normal_template in prompts["prompt_normal"]:
                text = normal_template.format(class_name)
                generated_prompts['normal'][class_name].append(
                    template.format(text)
                )

            for abnormal_template in prompts["prompt_abnormal"]:
                text = abnormal_template.format(class_name)
                generated_prompts['abnormal'][class_name].append(
                    template.format(text)
                )

    return generated_prompts

def generate_text_embeddings(args, dataset):
    prompts = prompt_generator(dataset)
    text_encoder = TextEncoder(args.model_id, args.device)

    embeddings = {}
    
    total = sum(1 for state in prompts.keys() for _ in prompts[state].keys())
    tqdm_bar = tqdm(total=total, desc="Encoding prompts with text encoder")
    for state in prompts.keys():
        embeddings[state] = {}
        for class_name in prompts[state].keys():
            embeddings[state][class_name] = text_encoder(prompts[state][class_name]).mean(dim=0)
            tqdm_bar.update(1)
            tqdm_bar.set_postfix({"State": state, "Class": class_name})
    return embeddings

def save_model(model, optimizer, path, epoch):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, os.path.join(path, f'model_epoch_{epoch}.pth'))