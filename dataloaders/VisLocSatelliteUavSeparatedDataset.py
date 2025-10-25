from torch.utils.data import Dataset
import pandas as pd
import utm
import numpy as np

from sklearn.neighbors import NearestNeighbors
from PIL import Image, UnidentifiedImageError


def get_separated_test_set(dataframe_csv_path, input_transform):
    return SatelliteUavDataset(
        dataframe_csv_path=dataframe_csv_path, input_transform=input_transform
    )


class SatelliteUavDataset(Dataset):
    def __init__(
        self,
        dataframe_csv_path,
        q_ratio=0.65,
        posDistThr=100,
        input_transform=None,
        onlyDB=False,
        random_seed=10,
    ):
        super().__init__()
        self.df = pd.read_csv(dataframe_csv_path)
        self.posDistThr = posDistThr
        self.input_transform = input_transform
        self.q_ratio = q_ratio
        self.random_seed = random_seed

        self.positives = None
        self.distances = None

        self.__calculate_properties()

        self.images = self.db_image_paths.copy()
        if not onlyDB:
            self.images += self.q_image_paths.copy()

    def __getitem__(self, index):
        try:
            img = Image.open(self.images[index])
        except UnidentifiedImageError:
            print(f"Image {self.images[index]} could not be loaded")
            # img = Image.new('RGB', (224, 224))

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    def __calculate_properties(self):

        # TODO LZ: do not use magic strings
        db_satellite = (
            self.df[self.df["friendly-name"].str.contains("satellite")]
            .copy()
            .reset_index(drop=True)
        )
        q_uav_all = (
            self.df[self.df["friendly-name"].str.contains("uav")]
            .copy()
            .reset_index(drop=True)
        )
        num_q_sample = int(len(q_uav_all) * self.q_ratio)
        q_uav = q_uav_all.sample(
            n=num_q_sample, random_state=self.random_seed
        ).reset_index(drop=True)

        self.db_image_paths = db_satellite["img_path"].tolist()

        self.q_image_paths = q_uav["img_path"].tolist()

        self.db_utm_np = np.array(db_satellite[["e_utm", "n_utm"]].values)
        self.q_utm_np = np.array(db_satellite[["e_utm", "n_utm"]].values)

    def get_positives(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=1)
            knn.fit(self.db_utm_np)

            self.distances, self.positives = knn.radius_neighbors(
                self.q_utm_np, radius=self.posDistThr
            )

        return self.positives
