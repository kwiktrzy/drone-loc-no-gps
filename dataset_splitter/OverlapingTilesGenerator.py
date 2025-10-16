import os
from collections import namedtuple
from typing import List

import pandas as pd

import rasterio
from rasterio.windows import Window
import cv2
import numpy as np
import uuid
import utm

from dataset_splitter.MapSatellite import MapSatellite
from dataset_splitter.AbstractGenerator import UTMPatchCoordinates, AbstractGenerator


class OverlapingTilesGenerator:
    def __init__(
        self,
        output_dir,
        width_size: int = 224,
        height_size: int = 224,
        patch_format_compress=(".jpg", 100),
        is_rebuild_csv=False,
        satellite_map_names: List[MapSatellite] = [],
    ):
        self.output_dir = output_dir
        self.width_size = width_size
        self.height_size = height_size
        self.patch_format_compress = patch_format_compress
        self.is_rebuild_csv = is_rebuild_csv
        self.satellite_map_names: List[MapSatellite] = satellite_map_names
        self.csv_tiles_paths: List[str] = []
        self.crop_range = 448  # Pixels to crop from tiff
        self.overlap_stride_meters = 25
        self.converters = AbstractGenerator()

    def __get_map_coordinates_csv(self, csv_path, mapname):
        df = pd.read_csv(csv_path)
        row = df.loc[df["mapname"] == mapname].iloc[0]
        lt_lon, rb_lon = row["LT_lon_map"], row["RB_lon_map"]
        lt_lat, rb_lat = row["LT_lat_map"], row["RB_lat_map"]
        coordinates = namedtuple(
            "coordinates", ["lt_lat", "lt_lon", "rb_lat", "rb_lon"]
        )
        return coordinates(lt_lat, lt_lon, rb_lat, rb_lon)

    def __save_patch(self, patch, patch_dir_ext):
        success = False
        if self.patch_format_compress[0] == ".png":
            success = cv2.imwrite(
                patch_dir_ext,
                patch,
                [cv2.IMWRITE_PNG_COMPRESSION, self.patch_format_compress[1]],
            )
        elif self.patch_format_compress[0] == ".jpg":
            resized_img = cv2.resize(
                patch,
                dsize=(self.width_size, self.height_size),
                interpolation=cv2.INTER_LANCZOS4,
            )
            success = cv2.imwrite(
                patch_dir_ext,
                resized_img,
                [cv2.IMWRITE_JPEG_QUALITY, self.patch_format_compress[1]],
            )
        else:
            print(f"\n ERROR: Unsupported tile format: {self.patch_format_compress[0]}")
            return False

        if not success:
            print(f"\n ERROR: Failed to save image to: {patch_dir_ext}")
            return False

        return True

    def get_patch_utm_coords(self, src, window: Window) -> UTMPatchCoordinates:
        lt_e, lt_n = src.transform * (window.col_off, window.row_off)
        rb_e, rb_n = src.transform * (
            window.col_off + window.width,
            window.row_off + window.height,
        )

        center_col = window.col_off + window.width / 2
        center_row = window.row_off + window.height / 2
        center_e, center_n = src.transform * (center_col, center_row)

        return UTMPatchCoordinates(
            lt_e=lt_e,
            lt_n=lt_n,
            rb_e=rb_e,
            rb_n=rb_n,
            center_e=center_e,
            center_n=center_n,
        )

    def __calculate_pixel_resolution(self, src, map: MapSatellite):
        lt_utm = self.converters.gps_to_utm(
            map.coordinates.lt_lat, map.coordinates.lt_lon
        )
        rb_utm = self.converters.gps_to_utm(
            map.coordinates.rb_lat, map.coordinates.rb_lon
        )

        width_meters = abs(lt_utm.e - rb_utm.e)
        height_meters = abs(lt_utm.n - rb_utm.n)

        meters_per_pixel_x = width_meters / src.width
        meters_per_pixel_y = height_meters / src.height

        return meters_per_pixel_x, meters_per_pixel_y

    def __generate_patches(
        self,
        src,
        map: MapSatellite,
        overlap_pixels_x,
        overlap_pixels_y,
    ):
        step_x = self.crop_range - overlap_pixels_x
        step_y = self.crop_range - overlap_pixels_y

        for y_off in range(0, src.height - self.crop_range + 1, step_y):
            for x_off in range(0, src.width - self.crop_range + 1, step_x):
                window = Window(x_off, y_off, self.crop_range, self.crop_range)
                patch_data = src.read(window=window, boundless=True)
                patch_cv2 = np.moveaxis(patch_data, 0, -1)
                if patch_cv2.shape[-1] == 3:
                    patch_cv2 = cv2.cvtColor(patch_cv2, cv2.COLOR_RGB2BGR)
                yield (patch_cv2, window)

    def __crop_map(self, map: MapSatellite):
        with rasterio.open(map.map_tif_path) as src:
            width_image, height_image = src.width, src.height
            print("\n width:  ", width_image)
            print("\n height: ", height_image)
            print("\n channel: ", src.count)

            center_lat, center_lon = self.converters.get_center(map.coordinates)
            _, _, zone_num, zone_let = utm.from_latlon(center_lat, center_lon)
            zone_str = f"{zone_num}{zone_let}"

            meters_per_pixel_x, meters_per_pixel_y = self.__calculate_pixel_resolution(
                src, map
            )
            csv_rows = []
            overlap_pixels_x = int(self.overlap_stride_meters / meters_per_pixel_x)
            overlap_pixels_y = int(self.overlap_stride_meters / meters_per_pixel_y)
            for i, (patch, window) in enumerate(
                self.__generate_patches(
                    map=map,
                    src=src,
                    overlap_pixels_x=overlap_pixels_x,
                    overlap_pixels_y=overlap_pixels_y,
                )
            ):
                utm_coords = self.get_patch_utm_coords(src, window)
                dir_output = f"{self.output_dir}/{map.region_name}"
                os.makedirs(dir_output, exist_ok=True)

                patch_filename = f"patch_{int(utm_coords.center_e)}_{int(utm_coords.center_n)}_{uuid.uuid4().hex[:8]}"
                patch_dir_ext = (
                    f"{dir_output}/{patch_filename}{self.patch_format_compress[0]}"
                )

                row = {
                    "img_path": patch_dir_ext,
                    "center_e": utm_coords.center_e,
                    "center_n": utm_coords.center_n,
                    "lt_e": utm_coords.lt_e,
                    "lt_n": utm_coords.lt_n,
                    "rb_e": utm_coords.rb_e,
                    "rb_n": utm_coords.rb_n,
                    "zone_utm": zone_str,
                    "patch_width": self.width_size,
                    "patch_height": self.height_size,
                    "region_name": map.region_name,
                    "friendly-name": map.friendly_name,
                }
                csv_rows.append(row)

                if not self.__save_patch(patch, patch_dir_ext):
                    print(f"Skipping patch due to save error: {patch_dir_ext}")

            self.converters.append_rows_csv(
                csv_rows, map.tiles_satellite_csv_output_path, self.is_rebuild_csv
            )

    def generate_tiles(self):
        for map in self.satellite_map_names:
            print(f"\n Processing map: {map.friendly_name}")
            coords = self.__get_map_coordinates_csv(map.csv_path, map.map_name)
            map.set_coordinates(
                lt_lat=coords.lt_lat,
                lt_lon=coords.lt_lon,
                rb_lat=coords.rb_lat,
                rb_lon=coords.rb_lon,
            )
            self.__crop_map(map)
            print(f"\n Processed {map.map_name}")
