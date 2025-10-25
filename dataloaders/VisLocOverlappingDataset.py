from typing import List

import torch
import pandas as pd
from PIL import UnidentifiedImageError
import PIL.Image
from torch.utils.data import Dataset

import torchvision.transforms as T

default_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class VisLocOverlappingDataset(Dataset):
    def __init__(
        self,
        # tiles_csv_file_paths: List[str] = None,
        dataframe: pd.DataFrame = None,  # Dataframe with labels
        random_sample_from_each_place=True,
        transform=default_transform,
        image_per_place=4,
    ):
        super(VisLocOverlappingDataset, self).__init__()
        # self.tiles_csv_file_paths = tiles_csv_file_paths
        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform
        self.image_per_place = image_per_place

        self.dataframe = dataframe
        self.places_ids = pd.unique(self.dataframe.index)

        # TODO: FINISH IT

    @staticmethod
    def __image_loader(path):
        try:
            return PIL.Image.open(path).convert("RGB")
        except UnidentifiedImageError:
            print(f"Image {path} could not be loaded")
            return PIL.Image.new("RGB", (224, 224))

    def __getitem__(self, index):

        # TODO LZ: THERE WE HAVE PICTURE INDEXES!
        place_id = self.places_ids[index]
        places = self.dataframe[self.dataframe.index == place_id]
        images = []

        for i, row in places.iterrows():
            img_path = row["img_path"]
            img = self.__image_loader(img_path)
            if self.transform is not None:
                img = self.transform(img)

            images.append(img)

        # NOTE: contrary to image classification where __getitem__ returns only one image
        # there, we return a place, which is a Tesor of K images (K=self.img_per_place)
        # this will return a Tensor of shape [K, channels, height, width]. This needs to be taken into account
        # in the Dataloader (which will yield batches of shape [BS, K, channels, height, width])
        return torch.stack(images), torch.tensor(place_id).repeat(self.image_per_place)

    def __len__(self):
        return len(self.places_ids)
