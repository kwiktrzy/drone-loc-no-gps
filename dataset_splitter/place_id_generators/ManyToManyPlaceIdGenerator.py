from pathlib import Path
import shutil
from typing import List
import pandas as pd
import numpy as np
import os
from dataset_splitter.AbstractGenerator import UTMPatchCoordinates, AbstractGenerator
from dataset_splitter.structs.InformativenessFilter import InformativenessFilter
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
# is_validation_set_v2 means we doing query like for train set but also added dont used sat images into Dataframe csv


class ManyToManyPlaceIdGenerator:
    def __init__(
        self,
        radius_neighbors_meters: int = 50,
        top_n_neighbors: int = 1,
        csv_tiles_path: str = "",
        csv_place_ids_output_path: str = "",
        tiles_trash_directory: Path = Path(),
        is_validation_set=False,
        is_validation_set_v2=False,
        force_regenerate=False,
        include_uav_in_output=True,
        is_non_overlaping_uavs=False,  # False
    ):
        self.radius_neighbors_meters = radius_neighbors_meters
        self.top_n_neighbors = top_n_neighbors
        self.csv_tiles_path = csv_tiles_path
        self.csv_place_ids_output_path = csv_place_ids_output_path
        self.force_regenerate = force_regenerate
        self.include_uav_in_output = include_uav_in_output
        self.is_validation_set = is_validation_set
        self.is_validation_set_v2 = is_validation_set_v2
        self.is_non_overlaping_uavs = is_non_overlaping_uavs
        self.used_sat_imgs = defaultdict(int)
        self.converters = AbstractGenerator()
        self.informativeness_calculation = InformativenessFilter()
        self.tiles_trash_directory = tiles_trash_directory
        
        self.tiles_trash_directory.mkdir(parents=True, exist_ok=True)


    def __utm_to_columns(self, row):
        utm_cords = self.converters.gps_to_utm(row.lat, row.lon)
        return {"e_utm": utm_cords.e, "n_utm": utm_cords.n, "zone_utm": utm_cords.zone}

    def get_non_overlapping_uavs(
        self, df_tiles, uav_records, radius_meters: float
    ) -> pd.DataFrame:

        if uav_records.empty:
            print(f"Error: do not find any rows UAV.")
            return pd.DataFrame()

        original_indices = uav_records.index
        uav_records.reset_index(drop=True, inplace=True)
        uav_coords = uav_records[["e_utm", "n_utm"]].to_numpy()

        if uav_coords.shape[0] == 0:
            print(f" ERROR: Do not find any rows UAV.")
            return pd.DataFrame()

        nn_r = NearestNeighbors(radius=radius_meters, algorithm="kd_tree")
        nn_r = nn_r.fit(uav_coords)
        distances_r, indices_r = nn_r.radius_neighbors(uav_coords)

        indices_to_skip = set()
        iloc_indices_to_keep = []
        for i in range(len(uav_records)):
            result = self.informativeness_calculation.analyze(
                uav_records.iloc[i]["img_path"]
            )
            if not result.is_informative:
                shutil.copy2(uav_records.iloc[i]["img_path"], self.tiles_trash_directory)
                print(f"✗ REJECT: {result.details}")
                continue
            if i in indices_to_skip:
                continue
            iloc_indices_to_keep.append(i)
            indices_to_skip.update(indices_r[i])

        original_indices_to_keep = original_indices[iloc_indices_to_keep]
        final_filtered_records = df_tiles.loc[original_indices_to_keep].copy()

        return final_filtered_records

    def _fit_informativeness_filter(self, paths):
        n = min(200, len(paths))
        sampled_paths = paths.sample(n=n)
        # self.informativeness_calculation.fit(sampled_paths)

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
        df = pd.read_csv(self.csv_tiles_path)

        df = df.loc[:, ~df.columns.duplicated()]
        if "index" in df.columns:
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])
        elif "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "index"})
        else:
            df.index.name = "index"
            df = df.reset_index()

        if not {"e_utm", "n_utm", "zone_utm"}.issubset(df.columns):
            df = df.join(df.apply(self.__utm_to_columns, axis=1).apply(pd.Series))

        has_same_zone = df["zone_utm"].nunique() == 1
        if not has_same_zone:
            print(" \n Warning: region contains different zones!")

        sat_records = df[
            df["friendly-name"].str.contains("satellite", case=False, na=False)
        ]

        uav_records = df[
            df["friendly-name"].str.contains("uav", case=False, na=False)
        ].copy()

        self._fit_informativeness_filter(uav_records["img_path"])

        if self.is_non_overlaping_uavs and not self.is_validation_set:
            uav_records = self.get_non_overlapping_uavs(
                df, uav_records, self.radius_neighbors_meters
            )
        else:
            indices_to_drop = []
            for index, row in uav_records.iterrows():
                result = self.informativeness_calculation.analyze(row["img_path"])
                if not result.is_informative:
                    try:
                        shutil.copy2(row["img_path"], self.tiles_trash_directory)
                        print(f"✗ REJECT: {result.details}")
                    except Exception as e:
                        print(f"Warning: Could not copy file to trash: {e}")
                    indices_to_drop.append(index)
            
            uav_records = uav_records.drop(indices_to_drop)

        if self.is_validation_set and not self.is_validation_set_v2:
            print(" -> MODE: Validation (query: SAT, Neighbor: UAV)")
            query_records = sat_records
            database_records = uav_records
        else:
            print(" -> MODE: Training (query: UAV, Neighbor: SAT)")
            query_records = uav_records
            database_records = sat_records

        query_coords = query_records[["e_utm", "n_utm"]].to_numpy()
        database_coords = database_records[["e_utm", "n_utm"]].to_numpy()

        if len(database_coords) == 0 or len(query_coords) == 0:
            print("Error: UAV or SAT is empty.")
            return

        if self.top_n_neighbors is not None:
            print(f" Using top {self.top_n_neighbors} nearest neighbors")
            nn_model = NearestNeighbors(
                n_neighbors=min(self.top_n_neighbors + 1, len(database_coords)),
                algorithm="kd_tree",
            )
            nn_model = nn_model.fit(database_coords)
            distances_r, indices_r = nn_model.kneighbors(query_coords)
        else:
            print(f" Using radius-based neighbors ({self.radius_neighbors_meters}m)")
            nn_model = NearestNeighbors(
                radius=self.radius_neighbors_meters, algorithm="kd_tree"
            )
            nn_model = nn_model.fit(database_coords)
            distances_r, indices_r = nn_model.radius_neighbors(query_coords)

        csv_rows = []

        for index, (distances, indices) in enumerate(zip(distances_r, indices_r)):
            reference_point = query_records.iloc[index]

            should_add_query = True

            if not self.is_validation_set:
                if not self.include_uav_in_output:
                    should_add_query = False
            else:
                pass

            neighbors_added = 0
            for dist, indic in zip(distances, indices):
                neighbour = database_records.iloc[indic]

                if neighbour["index"] == reference_point["index"]:
                    continue

                if (
                    self.top_n_neighbors is not None
                    and neighbors_added >= self.top_n_neighbors
                ):
                    break

                if (
                    self.top_n_neighbors is not None
                    and dist > self.radius_neighbors_meters
                ):
                    if not self.is_validation_set:
                        print(f"\nERROR: WE DO NOT FIND VALID NEIGHBOUR FOR TRAIN")
                        print(f"\n -> Nearest neighbour is {dist} m away")
                        continue
                    continue

                if self.is_validation_set_v2:
                    self.used_sat_imgs[neighbour["index"]] += 1
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
                    "reference_tile_id": reference_point["index"],
                    "reference_lon": reference_point["lon"],
                    "reference_lat": reference_point["lat"],
                    "lon": neighbour["lon"],
                    "lat": neighbour["lat"],
                    "width": neighbour["patch_width"],
                    "height": neighbour["patch_height"],
                    "source_file_path": self.csv_tiles_path,
                }
                csv_rows.append(row)
                neighbors_added += 1

            if should_add_query and (neighbors_added > 0 or self.is_validation_set):
                query_row = {
                    "img_path": reference_point["img_path"],
                    "place_id": index,
                    "distance_between_tiles_meters": 0.0,
                    "e_utm": reference_point["e_utm"],
                    "n_utm": reference_point["n_utm"],
                    "zone_utm": reference_point["zone_utm"],
                    "friendly-name": reference_point["friendly-name"],
                    "region_name": reference_point["region_name"],
                    "source_tile_id": reference_point["index"],
                    "reference_tile_id": reference_point["index"],
                    "reference_lon": reference_point["lon"],
                    "reference_lat": reference_point["lat"],
                    "lon": reference_point["lon"],
                    "lat": reference_point["lat"],
                    "width": reference_point["patch_width"],
                    "height": reference_point["patch_height"],
                    "source_file_path": self.csv_tiles_path,
                }
                csv_rows.append(query_row)
        latest_max_id = csv_rows[-1]["place_id"] + 1 if csv_rows else 0
        if self.is_validation_set_v2:
            for idx, iter_row in database_records.iterrows():
                if self.used_sat_imgs[iter_row["index"]] == 0:
                    row = {
                        "img_path": iter_row["img_path"],
                        "place_id": latest_max_id,
                        "distance_between_tiles_meters": 0.0,
                        "e_utm": iter_row["e_utm"],
                        "n_utm": iter_row["n_utm"],
                        "zone_utm": iter_row["zone_utm"],
                        "friendly-name": iter_row["friendly-name"],
                        "region_name": iter_row["region_name"],
                        "source_tile_id": iter_row["index"],
                        "reference_tile_id": iter_row["index"],
                        "reference_lon": iter_row["lon"],
                        "reference_lat": iter_row["lat"],
                        "lon": iter_row["lon"],
                        "lat": iter_row["lat"],
                        "width": iter_row["patch_width"],
                        "height": iter_row["patch_height"],
                        "source_file_path": self.csv_tiles_path,
                    }
                    csv_rows.append(row)
                    latest_max_id += 1

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
        print(f"\n Rows: {len(df_output)}")

        place_id_counts = df_output.groupby("place_id").size()
        print(f"\n Number of unique place_ids: {len(place_id_counts)}")
        print(f" Mean images per place_id: {place_id_counts.mean():.2f}")
        print(
            f" Min/Max images per place_id: {place_id_counts.min()}/{place_id_counts.max()}"
        )
