from typing import List
from torch.utils.data import BatchSampler, Sampler
import torch
import pandas as pd
import numpy as np
import random


class CustomVPRBatchSampler(Sampler):
    # P => Number of classes in batch
    # K => Number of pictures for one place_id - image per place
    def __init__(
        self, dataframe: pd.DataFrame, P: int, K: int, drop_last: bool = False
    ):
        self.dataframe = dataframe
        self.P = P
        self.K = K
        self.drop_last = drop_last
        self.places_ids = pd.unique(self.dataframe.index)

    def __iter__(self):
        np.random.shuffle(self.places_ids)

        num_batches = len(self.places_ids) // self.P
        if not self.drop_last and len(self.places_ids) % self.P != 0:
            num_batches += 1

        for i in range(num_batches):
            place_id = self.places_ids[i]
            places = self.dataframe[self.dataframe.index == place_id]
        # uav_place_ids = places[places["friendly-name"].str.contains("uav")]
        # sat_place_ids = places[places["friendly-name"].str.contains("satellite")]

        # uav_samples = uav_place_ids.sample(n=self.image_per_place // 2, replace=True)
        # sat_samples = sat_place_ids.sample(n=self.image_per_place // 2, replace=True)
        # final_samples = pd.concat([uav_samples, sat_samples])
        batch_indices = []
        yield batch_indices

    def __len__(self):
        return len(self.places_ids)

    zakazy = {4: [3, 5], 5: [4, 6], 3: [4], 6: [5]}
