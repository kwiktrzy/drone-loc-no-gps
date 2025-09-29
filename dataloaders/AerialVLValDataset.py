from torch.utils.data import Dataset

from os.path import join, exists
from collections import namedtuple
from scipy.io import loadmat

import torchvision.transforms as T


from PIL import Image, UnidentifiedImageError
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import utm
import numpy as np


class AerialVLValDataset(Dataset):
    def __init__(self, dataframe_csv_path, db_ratio, posDistThr=25, input_transform=None, onlyDB=False, random_seed=10):
        super().__init__()
        self.db_ratio = db_ratio
        self.random_seed=random_seed
        self.df = pd.read_csv(dataframe_csv_path)
        self.posDistThr = posDistThr
        self.input_transform=input_transform
        
        self.positives = None
        self.distances = None
        self.__calculate_properties()
        self.images = self.db_image_paths
        if not onlyDB:
            self.images += self.q_image_paths

    def __getitem__(self, index):
        try:
            img = Image.open(self.images[index])
        except UnidentifiedImageError:
            print(f'Image {self.images[index]} could not be loaded')
            img = Image.new('RGB', (224, 224))

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    def __calculate_properties(self):
        df_s = self.df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        utm_coords = df_s.apply(
        lambda row: utm.from_latlon(row['lat'], row['lon'])[:2],
        axis=1)
        utm_np = np.stack(utm_coords.tolist())
        num_total = len(df_s)
        split_index = int(num_total * self.db_ratio)
        all_image_paths = df_s['img_path'].tolist()
        self.db_image_paths = all_image_paths[:split_index]
        self.q_image_paths = all_image_paths[split_index:]

        self.db_utm_np = utm_np[:split_index]
        self.q_utm_np = utm_np[split_index:]

    def get_positives(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if self.positives is None: 
            knn = NearestNeighbors(n_jobs=1)
            knn.fit(self.db_utm_np)

            self.distances, self.positives = knn.radius_neighbors(self.q_utm_np,
                                                                  radius=self.posDistThr)

        return self.positives
    