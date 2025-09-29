from typing import List

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T

from dataloaders.VisLocDataset import VisLocDataset
from dataloaders.AerialVLValDataset import AerialVLValDataset

import dataloaders.VisLocSatelliteUavSeparatedDataset

#TODO: Check which dinov2's pretraning tensor transform was used for pretraining, and set mean correct mean
IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]}

VIT_MEAN_STD = {'mean': [0.5, 0.5, 0.5],
                'std': [0.5, 0.5, 0.5]}

TRAIN_DATASET_NAMES = [
    'visloc-Taizhou-1-03'
]

class MapsDataModule(pl.LightningDataModule):
    def __init__(self,
                 thumbnails_csv_file_paths: List[str]=[],
                 batch_size=32,
                 num_workers=1,
                 val_set_names=[],
                 shuffle_all=False,
                 mean_std=VIT_MEAN_STD,
                 random_sample_from_each_place=True,
                 image_size=(224, 224)
                 ):
        super().__init__()
        self.thumbnails_csv_file_paths: List[str]=thumbnails_csv_file_paths
        self.batch_size:int=batch_size
        self.num_workers=num_workers
        self.shuffle_all=shuffle_all
        self.mean_dataset = mean_std['mean']
        self.std_dataset = mean_std['std']
        self.val_set_names = val_set_names
        self.image_size = image_size

        self.random_sample_from_each_place = random_sample_from_each_place

        self.save_hyperparameters() # save hyperparameter with Pytorch Lightning

        self.train_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])

        self.valid_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset)])

        #TODO: ANALYSE IT, AND REWRITE
        self.train_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': self.shuffle_all}
        
        self.valid_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': 0,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': False}

    def setup(self, stage: str):
        if stage == 'fit' or stage == 'validate':
            self.reload()

            self.val_datasets=[]
            for valid_set_name in self.val_set_names:
                if "Shandong-1" in valid_set_name:
                    self.val_datasets.append(AerialVLValDataset(valid_set_name, 0.65, input_transform=self.valid_transform))
                elif "Shandan" in valid_set_name:
                    self.val_datasets.append(dataloaders.VisLocSatelliteUavSeparatedDataset.get_separated_test_set(valid_set_name, input_transform=self.valid_transform))

    def reload(self):
        self.train_dataset = VisLocDataset(
            thumbnails_csv_file_paths=self.thumbnails_csv_file_paths,
            random_sample_from_each_place=self.random_sample_from_each_place)
            # transform=self.train_transform)

    def train_dataloader(self):
        self.reload()
        return DataLoader(self.train_dataset, **self.train_loader_config)
    
    def val_dataloader(self):
        val_dataloaders = []
        for val_dataset in self.val_datasets:
            val_dataloaders.append(DataLoader(
                dataset=val_dataset, **self.valid_loader_config))
        return val_dataloaders