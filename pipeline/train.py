import os
import torch
import json
from datasets.constants import DatasetConstants
from utils.loss import BinaryDiceLoss, FocalLoss
from utils.utils import generate_text_embeddings, save_model
from tqdm import tqdm
from backbones.encoders import ImageEncoder
from torch.utils.data import DataLoader
from datasets import get_data
from utils.transformations import get_transforms
from model.model import ZSADModel
from torch.nn import functional as F

def training(args):
    dataset_constants = DatasetConstants(args.base_dir, args.dataset_name)

    text_embeddings = generate_text_embeddings(args, dataset_constants)

    transform_img, transform_mask = get_transforms(args.img_size)

    model = ZSADModel(args).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)

    if args.start_epochs > 0 and os.path.exists(os.path.join(args.output_dir, f'model_epoch_{args.start_epochs}.pth')):
        model_path = os.path.join(args.output_dir, f'model_epoch_{args.start_epochs}.pth')
        print(f"Loading checkpoint from: {model_path}")
        checkpoint = torch.load(model_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Successfully loaded model and optimizer states.")            
    else:
        print("No pre-trained model found. Starting training from scratch.")
        args.start_epochs = 0

    image_encoder = ImageEncoder(args.vision_model_id, args.vision_layers, device=args.device)

    dataset = get_data(args.dataset_name, transform_img, transform_mask, training=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    loss_dice = BinaryDiceLoss()
    pixel_focal = FocalLoss(alpha=0.75, gamma=2.0)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    json.dump(vars(args), open(os.path.join(args.output_dir, f'args.json'), 'w'), indent=4)

    for epoch in range(args.start_epochs, args.end_epochs):
        losses = {
            'localization_loss': 0.0,
            'global_context_regularizer': 0.0,
            'total_loss': 0.0
        }
        for data in tqdm(dataloader):
            normal_list = []
            abnormal_list = []

            for cls in data['cls_name']:
                normal_list.append(text_embeddings['normal'][cls])
                abnormal_list.append(text_embeddings['abnormal'][cls])

            normal_batch = torch.stack(normal_list, dim=0)
            abnormal_batch = torch.stack(abnormal_list, dim=0)
            
            text_embeddings_dict = {
                'normal': normal_batch.to(args.device),
                'abnormal': abnormal_batch.to(args.device)
            }
            
            cls, patches = image_encoder(data['img'])

            text_guided_anomaly_map, global_context_regularizer_score = model(text_embeddings_dict, [cls, patches])

            mask = data['img_mask'].to(args.device).float()

            localization_loss = pixel_focal(text_guided_anomaly_map, mask) + loss_dice(text_guided_anomaly_map[:, 1, :, :], mask)
            global_context_regularizer_loss = F.cross_entropy(global_context_regularizer_score.squeeze(1), data['anomaly'].to(args.device).long())
            loss =  0.5 * localization_loss + 0.5 * global_context_regularizer_loss

            losses['localization_loss'] += localization_loss.item()
            losses['global_context_regularizer'] += global_context_regularizer_loss.item()
            losses['total_loss'] += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        losses = {k: v / len(dataloader) for k, v in losses.items()}
        print(
            f"Epoch {epoch+1}/{args.end_epochs}: "
            f"Localization Loss: {losses['localization_loss']:.4f}, "
            f"Global Context Regularizer Loss: {losses['global_context_regularizer']:.4f}, "
            f"Total Loss: {losses['total_loss']:.4f}",
            flush=True
        )
        save_model(model, optimizer, args.output_dir, epoch+1)
