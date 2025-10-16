from typing import List
import pandas as pd
import numpy as np
import os
from dataset_splitter.AbstractGenerator import UTMPatchCoordinates, AbstractGenerator
from sklearn.neighbors import NearestNeighbors


class ManyToManyPlaceIdGenerator:
    def __init__(
        self,
        radius_neighbors_meters: int = 50,
        csv_tiles_path: str = "",
        csv_place_ids_output_path: str = "",
        force_regenerate=False,
    ):
        self.radius_neighbors_meters = radius_neighbors_meters
        self.csv_tiles_path = csv_tiles_path
        self.csv_place_ids_output_path = csv_place_ids_output_path
        self.force_regenerate = force_regenerate
        self.converters = AbstractGenerator()

    def __utm_to_columns(self, row):
        utm_cords = self.converters.gps_to_utm(row.lat, row.lon)
        return {"e_utm": utm_cords.e, "n_utm": utm_cords.n, "zone_utm": utm_cords.zone}

    def generate_place_ids(self):

        if os.path.exists(self.csv_place_ids_output_path):
            df_output = pd.read_csv(self.csv_place_ids_output_path)
            if "place_id" in df_output.columns and not self.force_regenerate:
                print(
                    f" Skipping Many to Many Place ID generation for {self.csv_place_ids_output_path}, 'place_id' column already exists."
                )
                return

        print(
            f" Generating Many to Many Place ID for {self.csv_place_ids_output_path}..."
        )

        try:
            df = pd.read_csv(self.csv_tiles_path, index_col=0)
        except IndexError:
            df = pd.read_csv(self.csv_tiles_path)

        df = df.reset_index(drop=True)
        df.to_csv(self.csv_tiles_path, index=True)

        if not {"e_utm", "n_utm", "zone_utm"}.issubset(df.columns):
            df = df.join(df.apply(self.__utm_to_columns, axis=1).apply(pd.Series))

        has_same_zone = df["zone_utm"].nunique() == 1

        if not has_same_zone:
            print(" \n Warning: region contains different zones!")

        tiles_coords = df[["e_utm", "n_utm"]].to_numpy()

        nn_r = NearestNeighbors(
            radius=self.radius_neighbors_meters, algorithm="kd_tree"
        )

        nn_r = nn_r.fit(tiles_coords)

        uav_records = df[df["friendly-name"].str.contains("uav", case=False, na=False)]
        uav_coords = uav_records[["e_utm", "n_utm"]].to_numpy()
        distances_r, indices_r = nn_r.radius_neighbors(uav_coords)
        csv_rows = []
        for index, (distances, indices) in enumerate(zip(distances_r, indices_r)):

            for dist, indic in zip(distances, indices):
                reference_point_uav = uav_records.iloc[index]
                neighbour = df.iloc[indic]
                row = {
                    "img_path": neighbour["img_path"],
                    "place_id": index,
                    "distance_between_tiles_meters": dist,
                    "e_utm": neighbour["e_utm"],
                    "n_utm": neighbour["n_utm"],
                    "zone_utm": neighbour["zone_utm"],
                    "friendly-name": neighbour["friendly-name"],
                    "region_name": neighbour["region_name"],
                    "source_tile_id": neighbour["index"],
                    "reference_tile_id": reference_point_uav["index"],
                    "reference_lon": reference_point_uav["lon"],
                    "reference_lat": reference_point_uav["lat"],
                    "lon": neighbour["lon"],
                    "lat": neighbour["lat"],
                    "source_file_path": self.csv_tiles_path,
                }
                csv_rows.append(row)

        self.converters.append_rows_csv(
            csv_rows, self.csv_place_ids_output_path, self.force_regenerate
        )
        print(
            f"\n Many to Many Place ID for {self.csv_place_ids_output_path} finished."
        )
        df_output = pd.read_csv(self.csv_place_ids_output_path)
        print(
            f'\n Mean distance between neighbours: {df_output["distance_between_tiles_meters"].mean()} m'
        )
        print(f"\fRows: {len(df_output)}")
