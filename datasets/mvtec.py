import os
from .base_dataset import BaseDataset

'''dataset source: https://paperswithcode.com/dataset/mvtecad'''

class MVTecDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames, root, training=True):
        super(MVTecDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, training=training
        )
