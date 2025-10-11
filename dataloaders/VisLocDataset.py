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


class VisLocDataset(Dataset):
    def __init__(
        self,
        tiles_csv_file_paths: List[str] = None,
        random_sample_from_each_place=True,
        transform=default_transform,
        image_per_place=4,
    ):
        super(VisLocDataset, self).__init__()
        self.tiles_csv_file_paths = tiles_csv_file_paths
        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform
        self.image_per_place = image_per_place

        self.dataframe = self.__get_data_frames()
        self.places_ids = pd.unique(self.dataframe.index)

        # TODO: FINISH IT

    def __get_data_frames(self):
        df = pd.read_csv(self.tiles_csv_file_paths[0])
        df = df.sample(frac=1)
        for idx, csv_path in enumerate(self.tiles_csv_file_paths[1:], start=1):
            temp_df = pd.read_csv(csv_path)
            temp_df["place_id"] = temp_df["place_id"] + (idx * 10**5)
            temp_df = temp_df.sample(frac=1)
            df = pd.concat([df, temp_df], ignore_index=True)
        return df.set_index("place_id")

    @staticmethod
    def __image_loader(path):
        try:
            return PIL.Image.open(path).convert("RGB")
        except UnidentifiedImageError:
            print(f"Image {path} could not be loaded")
            return PIL.Image.new("RGB", (224, 224))

    def __getitem__(self, index):
        place_id = self.places_ids[index]
        places = self.dataframe[self.dataframe.index == place_id]

        # places = places.sample(n=self.image_per_place, replace=True)
        # LZ change : forcing the addition of both domains at the same time has been added because we want to teach the model to search for similarities between domains. Only images from one domain can be included in the batch,
        # which artificially inflates b_acc, resulting in a weak eval score.
        uav_place_ids = places[places["friendly-name"].str.contains("uav")]
        sat_place_ids = places[places["friendly-name"].str.contains("satellite")]

        uav_samples = uav_place_ids.sample(n=self.image_per_place // 2, replace=True)
        sat_samples = sat_place_ids.sample(n=self.image_per_place // 2, replace=True)
        final_samples = pd.concat([uav_samples, sat_samples])

        images = []

        for i, row in final_samples.iterrows():
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
