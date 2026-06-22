from .constants import DatasetConstants
from .mvtec import MVTecDataset
from .visa import VisaDataset
from .btad import BTADDataset
from .mpdd import MPDDDataset


base_dir = ''


mvtec_dataset_constants = DatasetConstants(base_path=base_dir, dataset_name='MVTec')
visa_dataset_constants = DatasetConstants(base_path=base_dir, dataset_name='VisA')
btad_dataset_constants = DatasetConstants(base_path=base_dir, dataset_name='btad')
mpdd_dataset_constants = DatasetConstants(base_path=base_dir, dataset_name='MPDD')

dataset_dict = {
    'btad': (btad_dataset_constants.get_class_names(), BTADDataset, btad_dataset_constants.get_data_path()),
    'mpdd': (mpdd_dataset_constants.get_class_names(), MPDDDataset, mpdd_dataset_constants.get_data_path()),
    'mvtec': (mvtec_dataset_constants.get_class_names(), MVTecDataset, mvtec_dataset_constants.get_data_path()),
    'visa': (visa_dataset_constants.get_class_names(), VisaDataset, visa_dataset_constants.get_data_path()),
}

def get_data(dataset_name, transform, target_transform, training):
    dataset_cls_names, dataset_instance, dataset_root = dataset_dict[dataset_name]
    return dataset_instance(
                clsnames=dataset_cls_names,
                transform=transform,
                target_transform=target_transform,
                training=training,
                root=dataset_root
            )