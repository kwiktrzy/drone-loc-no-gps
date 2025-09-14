from typing import List

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataloaders.VisLocDataset import VisLocDataset


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
                 thumbnails_csv_file_paths: List[str]=None,
                 batch_size=32,
                 num_workers=4,
                 shuffle_all=False,
                 mean_std=IMAGENET_MEAN_STD,
                 random_sample_from_each_place=True
                 ):
        super().__init__()
        self.thumbnails_csv_file_paths: List[str]=thumbnails_csv_file_paths
        self.batch_size:int=batch_size
        self.num_workers=num_workers
        self.shuffle_all=shuffle_all
        self.mean_dataset = mean_std['mean']
        self.std_dataset = mean_std['std']

        self.random_sample_from_each_place = random_sample_from_each_place

        self.save_hyperparameters() # save hyperparameter with Pytorch Lightning

        #TODO: ANALYSE IT, AND REWRITE
        self.train_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': self.shuffle_all}

    def setup(self, stage: str):
        if stage == 'fit':
            self.reload()

    def reload(self):
        self.train_dataset = VisLocDataset(
            thumbnails_csv_file_paths=self.thumbnails_csv_file_paths,
            random_sample_from_each_place=self.random_sample_from_each_place)
            # transform=self.train_transform)

    def train_dataloader(self):
        self.reload()
        return DataLoader(self.train_dataset, **self.train_loader_config)