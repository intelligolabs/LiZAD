import os
import torch
from datasets.constants import DatasetConstants
from utils.utils import generate_text_embeddings
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from backbones.encoders import ImageEncoder
from torch.utils.data import DataLoader
from datasets import get_data
from utils.transformations import get_transforms
from model.model import ZSADModel
import numpy as np
def inference_batches(dataloader, model, image_encoder, text_embeddings):
    for data in tqdm(dataloader):
        normal_list = []
        abnormal_list = []

        for cls in data['cls_name']:
            normal_list.append(text_embeddings['normal'][cls])
            abnormal_list.append(text_embeddings['abnormal'][cls])

        normal_batch = torch.stack(normal_list, dim=0)
        abnormal_batch = torch.stack(abnormal_list, dim=0)
        

        text_embeddings_dict = {
            'normal': normal_batch.to(next(model.parameters()).device),
            'abnormal': abnormal_batch.to(next(model.parameters()).device)
        }
        
        cls, patches = image_encoder(data['img'])

        text_guided_anomaly_scores, _ = model(text_embeddings_dict, [cls, patches])
    
        pixel_anomaly_map =  text_guided_anomaly_scores[:, 1, :, :]

        yield {
            "gt_masks": data['img_mask'].squeeze(1).cpu().detach().numpy(),
            "pred_masks": pixel_anomaly_map.cpu().detach().numpy().astype(np.float32, copy=False),
            "img_paths": list(data["img_path"]),
            "cls_names": list(data['cls_name']),
        }


def inference_epoch(dataloader, model, image_encoder, text_embeddings):
    pixel_gt = []
    pixel_pred = []
    img_list = []
    cls_list = []

    for outputs in inference_batches(dataloader, model, image_encoder, text_embeddings):
        pixel_gt.extend(outputs["gt_masks"])
        pixel_pred.extend(outputs["pred_masks"])
        img_list.extend(outputs["img_paths"])
        cls_list.extend(outputs["cls_names"])

    return {
        "gt_masks": np.asarray(pixel_gt),
        "pred_masks": np.asarray(pixel_pred, dtype=np.float32),
        "img_paths": np.asarray(img_list),
        "cls_names": np.asarray(cls_list),
    }


def testing_epoch(dataloader, model, image_encoder, text_embeddings):
    outputs = inference_epoch(dataloader, model, image_encoder, text_embeddings)

    gt_mask_list = outputs["gt_masks"]
    pred_mask_list = outputs["pred_masks"]
    cls_list = outputs["cls_names"]


    auroc_list = []

    for cls_name in np.unique(cls_list):
        idx = (cls_list == cls_name)

        gt_cls = gt_mask_list[idx].ravel().astype(np.uint8, copy=False)
        pred_cls = pred_mask_list[idx].ravel().astype(np.float32, copy=False)

        auroc = roc_auc_score(gt_cls, pred_cls)

        if not np.isnan(auroc):
            auroc_list.append(auroc)

        print(
            f"{cls_name}: "
            f"AUC_Pixel={auroc:.5f}"
        )

    mean_auroc = np.mean(auroc_list) if len(auroc_list) > 0 else float("nan")

    print(f"\nMean AUC_Pixel: {mean_auroc:.5f}")


def testing(args):
    dataset_constants = DatasetConstants(args.base_dir, args.dataset_name)

    text_embeddings = generate_text_embeddings(args, dataset_constants)

    transform_img, transform_mask = get_transforms(args.img_size)

    model = ZSADModel(args).to(args.device)
    image_encoder = ImageEncoder(args.vision_model_id, args.vision_layers, device=args.device)

    dataset = get_data(args.dataset_name, transform_img, transform_mask, training=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if args.model_name is not None:
        print(f"Testing with model: {args.model_name}")
        model_path = os.path.join(args.output_dir, args.model_name)
        checkpoint = torch.load(model_path)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        with torch.no_grad():
            testing_epoch(dataloader, model, image_encoder, text_embeddings)
        return
        
    for i in range(args.start_epochs, args.end_epochs):
        model_name = f'model_epoch_{i+1}.pth'
        print(f"Testing with model: {model_name}")
        model_path = os.path.join(args.output_dir, model_name)
        checkpoint = torch.load(model_path)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        with torch.no_grad():
            testing_epoch(dataloader, model, image_encoder, text_embeddings)
        
        
