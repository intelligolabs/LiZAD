import os
from .base_dataset import BaseDataset

'''dataset source: https://github.com/stepanje/MPDD'''
MPDD_CLS_NAMES = [
    'bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate','tubes',
]

class MPDDDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames, root, training=True):
        super(MPDDDataset, self).__init__(
            clsnames=clsnames, transform=transform, root=root, target_transform=target_transform, training=training
        )

