from collections import namedtuple
from typing import List
from pandas import DataFrame
import rasterio
import rasterio.windows
from rasterio.windows import Window
from dataset_splitter.MapSatellite import MapSatellite
from dataset_splitter.AbstractGenerator import (
    PixelBoundingBox,
    Tile,
    UTMCoordinates,
    UTMPatchCoordinates,
    AbstractGenerator,
)
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# gt: The ground-truth bounding box.
# pred: The predicted bounding box.
Detection = namedtuple("Detection", ["image_path", "gt", "pred"])


class Analyse:
    def __init__(self, csv_path, satellite_map: MapSatellite):
        self.csv_path = csv_path
        self.map: MapSatellite = satellite_map
        self.converters = AbstractGenerator()

    def __load_map_coords(self, csv_path, mapname):
        df = pd.read_csv(csv_path)
        row = df.loc[df["mapname"] == mapname].iloc[0]
        lt_lon, rb_lon = row["LT_lon_map"], row["RB_lon_map"]
        lt_lat, rb_lat = row["LT_lat_map"], row["RB_lat_map"]
        self.map.set_coordinates(lt_lat, lt_lon, rb_lat, rb_lon)

    def __calculate_pixel_resolution(self):
        with rasterio.open(self.map.map_tif_path) as src:
            self.lt_utm_sat = self.converters.gps_to_utm(
                self.map.coordinates.lt_lat, self.map.coordinates.lt_lon
            )
            self.rb_utm_sat = self.converters.gps_to_utm(
                self.map.coordinates.rb_lat, self.map.coordinates.rb_lon
            )

        width_meters = abs(self.lt_utm_sat.e - self.rb_utm_sat.e)
        height_meters = abs(self.lt_utm_sat.n - self.rb_utm_sat.n)

        self.meters_per_pixel_x = width_meters / src.width
        self.meters_per_pixel_y = height_meters / src.height

    def __bounding_box_for_picture(self, tile: Tile):
        e_meters_pic = abs(self.lt_utm_sat.e - tile.utm_centroid.e)
        n_meters_pic = abs(self.lt_utm_sat.n - tile.utm_centroid.n)

        centroid_pixel_x = e_meters_pic / self.meters_per_pixel_x
        centroid_pixel_y = n_meters_pic / self.meters_per_pixel_y

        box_top_left_x = centroid_pixel_x - tile.width / 2
        box_top_left_y = centroid_pixel_y - tile.height / 2

        box_bottom_right_x = centroid_pixel_x + tile.width / 2
        box_bottom_right_y = centroid_pixel_y + tile.height / 2

        m_width = abs(box_bottom_right_x - box_top_left_x) * self.meters_per_pixel_x
        m_height = abs(box_bottom_right_y - box_top_left_y) * self.meters_per_pixel_y

        return PixelBoundingBox(
            lt_x=box_top_left_x,
            lt_y=box_top_left_y,
            rb_x=box_bottom_right_x,
            rb_y=box_bottom_right_y,
            m_width=m_width,
            m_height=m_height,
        )

    def load_map(self):
        self.__load_map_coords(self.map.csv_path, self.map.map_name)
        self.__calculate_pixel_resolution()

    def draw_boxes(self):
        df = pd.read_csv(self.csv_path)
        filtered_df = self.load_top_place_id(df)
        if filtered_df is None:
            print("Error: Empty dataframe")
            return

        tiles_list = self.map_to_tiles(filtered_df)

        if not tiles_list:
            print("No tiles found after mapping.")
            return

        pixel_boxes_info = []
        for tile in tiles_list:
            print(f"Index: {tile.id}")
            box = self.__bounding_box_for_picture(tile)
            pixel_boxes_info.append((box, tile.friendly_name))

        if not pixel_boxes_info:
            print("No bounding boxes calculated.")
            return

        min_x = min(b[0].lt_x for b in pixel_boxes_info)
        min_y = min(b[0].lt_y for b in pixel_boxes_info)
        max_x = max(b[0].rb_x for b in pixel_boxes_info)
        max_y = max(b[0].rb_y for b in pixel_boxes_info)

        padding = 100

        with rasterio.open(self.map.map_tif_path) as src:
            crop_x_min = int(max(0, min_x - padding))
            crop_y_min = int(max(0, min_y - padding))
            crop_x_max = int(min(src.width, max_x + padding))
            crop_y_max = int(min(src.height, max_y + padding))

            window_width = crop_x_max - crop_x_min
            window_height = crop_y_max - crop_y_min

            if window_width <= 0 or window_height <= 0:
                print("Error: Invalid crop window dimensions.")
                return

            window = Window(crop_x_min, crop_y_min, window_width, window_height)

            image_crop = src.read(window=window)

        image_crop = np.moveaxis(image_crop, 0, -1)

        if image_crop.shape[2] > 3:
            image_rgb = image_crop[:, :, :3]
        else:
            image_rgb = image_crop

        image_normalized = cv2.normalize(
            image_rgb, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        if image_normalized.ndim == 2:
            image_bgr = cv2.cvtColor(image_normalized, cv2.COLOR_GRAY2BGR)
        elif image_normalized.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_normalized, cv2.COLOR_RGB2BGR)
        else:
            print(f"Error: Unexpected image shape {image_normalized.shape}")
            return

        image_bgr = np.ascontiguousarray(image_bgr, dtype=np.uint8)

        color_red_bgr = (0, 0, 255)
        color_green_bgr = (0, 255, 0)
        default_color = (255, 0, 0)
        thickness = 2

        for box, friendly_name in pixel_boxes_info:
            if "uav" in friendly_name:
                color = color_red_bgr
            elif "satellite" in friendly_name:
                color = color_green_bgr
            else:
                color = default_color

            adj_lt_x = int(box.lt_x - crop_x_min)
            adj_lt_y = int(box.lt_y - crop_y_min)
            adj_rb_x = int(box.rb_x - crop_x_min)
            adj_rb_y = int(box.rb_y - crop_y_min)

            cv2.rectangle(
                image_bgr,
                (adj_lt_x, adj_lt_y),
                (adj_rb_x, adj_rb_y),
                color,
                thickness,
            )
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        plt.imshow(img_rgb)

    def map_to_tiles(self, filtered_df):
        tiles_list: List[Tile] = []

        for index, row in filtered_df.iterrows():
            utm_coords = UTMCoordinates(
                e=row["e_utm"], n=row["n_utm"], zone=row["zone_utm"]
            )

            tile_obj = Tile(
                id=index,
                utm_centroid=utm_coords,
                width=int(row["patch_width"]),
                height=int(row["patch_height"]),
                friendly_name=row["friendly-name"],
                place_id=row["place_id"],
            )
            tiles_list.append(tile_obj)

        return tiles_list
        # uav_t_records = df_r_t[df_r_t["friendly-name"].str.contains("uav")]

    def load_top_place_id(self, df: DataFrame):
        column_name = "place_id"
        if column_name not in df.columns:
            print(f"Error: Csv do not have column '{column_name}'.")
        else:
            place_id_counts = df[column_name].value_counts()
            most_frequent_place_id = place_id_counts.idxmax()
            count = place_id_counts.max()
            filtered_df = df[df[column_name] == most_frequent_place_id]
            return filtered_df
