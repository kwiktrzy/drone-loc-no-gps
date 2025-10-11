from typing import List
import pandas as pd
import numpy as np
from AbstractGenerator import GPSCoordinates, AbstractGenerator
import utm


class ManyToManyPlaceIdGenerator:
    def __init__(
        self,
        radius_neighbors_meters: int = 50,
        csv_tiles_path: str = "",
        force_regenerate=False,
    ):
        self.radius_neighbors_meters = radius_neighbors_meters
        self.csv_tiles_path = csv_tiles_path
        self.force_regenerate = force_regenerate
        self.converters = AbstractGenerator()
        self.generate_place_ids()

    def utm_to_columns(self, row):
        utm_cords = self.converters.gps_to_utm(row.lat, row.lon)
        return {"e_utm": utm_cords.e, "n_utm": utm_cords.n, "zone_utm": utm_cords.zone}

    def generate_place_ids(self):
        df = pd.read_csv(self.csv_tiles_path)
        if "place_id" in df.columns and not self.force_regenerate:
            print(
                f" Skipping Place ID generation for {self.csv_tiles_path}, 'place_id' column already exists."
            )
            return
        print(f" Generating Place ID for {self.csv_tiles_path}...")

        if not {"e_utm", "n_utm", "zone_utm"}.issubset(df.columns):
            df = df.join(df.apply(self.utm_to_columns, axis=1).apply(pd.Series))
        name = df["friendly-name"]
        sat_mask = name.str.contains("satellite", case=False, na=False)
        uav_mask = name.str.contains("uav", case=False, na=False)
