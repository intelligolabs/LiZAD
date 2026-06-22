import os
from .base_dataset import BaseDataset

'''dataset source: https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar'''
VISA_CLS_NAMES = [
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
    'pcb4', 'pipe_fryum',
]

class VisaDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames, root, training=True):
        super(VisaDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, training=training
        )

