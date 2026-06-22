import os
from .base_dataset import BaseDataset

'''dataset source: https://avires.dimi.uniud.it/papers/btad/btad.zip'''
BTAD_CLS_NAMES = [
    '01', '02', '03',
]
class BTADDataset(BaseDataset):
    def __init__(self, transform, target_transform, root, clsnames, training=True):
        super(BTADDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, training=training
        )
