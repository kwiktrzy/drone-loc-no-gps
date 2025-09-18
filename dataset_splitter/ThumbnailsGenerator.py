import os
from collections import namedtuple
from typing import List

import pandas as pd

import rasterio
from rasterio.windows import Window
import cv2
import numpy as np
import uuid
import math

from dataset_splitter.MapSatellite import MapSatellite

#Hyperparameters
# Map parameters
# config_csv_path='/kaggle/input/uav-visloc-example/satellite_ coordinates_range.csv'y
# # mapname = '03.tif' # automatize process... because i thats bad idea to extract by magic string ids longitude and latitide map params.
# high_resolution_satelite_map_path_tif = '/kaggle/input/uav-visloc-example/03/satellite03.tif'
# width_crop_count=50
# height_crop_count=50
# # thumbnails_satelite_map_output_directory ='/kaggle/working/map_thumbnails'
# # # thumbnails_satelite_csv_output_path ='/kaggle/working/satelite_thumbnails_coords.csv'
# # is_recreate_csv = True
# # is_visualize_crop= False

# patch_format = '.png'#".jpg"
# compress_for_png = 5 #PNG: 0 (none) - 9 (max)
# compress_for_jpg = 92 # JPEG

class ThumbnailsGenerator:
    def __init__(self,
                 output_dir,
                 width_size: int=224,
                 height_size: int=224,
                 patch_format_compress=('.png', 5),
                 is_rebuild_csv=True,
                 satellite_map_names: List[MapSatellite]=None):
        self.output_dir=output_dir
        self.width_size=width_size
        self.height_size=height_size
        self.patch_format_compress=patch_format_compress
        self.is_rebuild_csv=is_rebuild_csv
        self.satellite_map_names: List[MapSatellite] = satellite_map_names
        self.csv_thumbnails_paths: List[str] = []

    def __get_map_coordinates_csv(self, csv_path, mapname):
        df = pd.read_csv(csv_path)
        row = df.loc[df['mapname'] == mapname].iloc[0]
        lt_lon, rb_lon = row['LT_lon_map'], row['RB_lon_map']
        lt_lat, rb_lat = row['LT_lat_map'], row['RB_lat_map']
        coordinates = namedtuple('coordinates', ['lt_lat', 'lt_lon', 'rb_lat', 'rb_lon'])
        return coordinates(lt_lat, lt_lon, rb_lat, rb_lon)

    def __split_sizes(self, total, size):
        parts = math.ceil(total / size)
        return [size*i for i in range(parts)]


    def __generate_patches(self, src, width_size=256, height_size=256):

        height, width = src.height, src.width
        x_sizes = self.__split_sizes(width, width_size)
        y_sizes = self.__split_sizes(height, height_size)

        for y_off in y_sizes:
            for x_off in x_sizes:

                window = Window(x_off, y_off, width_size, height_size)

                x_end = x_off + width_size
                y_end = y_off + height_size

                patch_data = src.read(window=window, boundless=True)  # shape: (bands, h, w)

                # Change to HWC
                patch_cv2 = np.moveaxis(patch_data, 0, -1)

                if patch_cv2.shape[-1] == 3:
                    patch_cv2 = cv2.cvtColor(patch_cv2, cv2.COLOR_RGB2BGR)

                yield patch_cv2, (x_off, y_off, x_end, y_end, width_size, height_size)

    def __get_gps_coords(self, x_start, y_start, x_end, y_end,
                       img_width, img_height,
                       top_left_lat, top_left_lon,
                       bottom_right_lat, bottom_right_lon):

        lat_per_pixel = (top_left_lat - bottom_right_lat) / img_height
        lon_per_pixel = (bottom_right_lon - top_left_lon) / img_width

        top_left_crop_lat = top_left_lat - y_start * lat_per_pixel
        top_left_crop_lon = top_left_lon + x_start * lon_per_pixel

        bottom_right_crop_lat = top_left_lat - y_end * lat_per_pixel
        bottom_right_crop_lon = top_left_lon + x_end * lon_per_pixel

        return (top_left_crop_lat, top_left_crop_lon,
                bottom_right_crop_lat, bottom_right_crop_lon)

    def __append_row_csv(self, row_append, file_path, is_recreate):
        # ex. row = { 'id': '1', 'value': 10 }
        is_file_exists = os.path.isfile(file_path)
        if is_file_exists and is_recreate:
            os.remove(file_path)
            is_file_exists = False
        df = pd.DataFrame([row_append])
        df.to_csv(file_path, mode='a', index=False, header=not is_file_exists)

    def __crop_map(self, map: MapSatellite):
        with rasterio.open(map.map_tif_path) as src:
            width_image, height_image = src.width, src.height  # size of patches...
            print('width:  ', width_image)
            print('height: ', height_image)
            print('channel: ', src.count)

            for i, (patch, coords) in enumerate(self.__generate_patches(src, self.width_size,
                                                                 self.height_size)):  # height_crop_count, width_crop_count - hyperparameters
                gps_coords = self.__get_gps_coords(coords[0],
                                                   coords[1],
                                                   coords[2],
                                                   coords[3],
                                                   width_image,
                                                   height_image,
                                                   map.coordinates.lt_lat,
                                                   map.coordinates.lt_lon,
                                                   map.coordinates.rb_lat,
                                                   map.coordinates.rb_lon)
                dir_output=f"{self.output_dir}/{map.region_name}"
                if(i == 0):
                    os.makedirs(dir_output, exist_ok=True)

                patch_dir=f"{dir_output}/patch__{gps_coords[0]}__{gps_coords[1]}__{gps_coords[2]}__{gps_coords[3]}__{uuid.uuid4().hex}"
                patch_dir_ext = f"{patch_dir.replace('.', '_')}{self.patch_format_compress[0]}"
                # place_id, source
                row = {'img_path': patch_dir_ext,
                       'LT_lat': gps_coords[0], 'LT_lon': gps_coords[1],
                       'RB_lat': gps_coords[2], 'RB_lon': gps_coords[3],
                       'patch_width': coords[4], 'patch_height': coords[5],
                       'region_name': map.region_name, 'friendly-name': map.friendly_name }

                if i == 1:
                    self.is_rebuild_csv=False #todo fix it because if we have many csvs it could work just for first csv
                self.__append_row_csv(row, map.thumbnails_satellite_csv_output_path, self.is_rebuild_csv)

                if self.patch_format_compress[0] == '.png':
                    cv2.imwrite(patch_dir_ext, patch, [cv2.IMWRITE_PNG_COMPRESSION, self.patch_format_compress[1]])
                elif self.patch_format_compress[0] == '.jpg':
                    cv2.imwrite(patch_dir_ext, patch, [cv2.IMWRITE_JPEG_QUALITY, self.patch_format_compress[1]])
                else:
                    print("OH NO...u used unsupported thumbnail format!")
                    return
                
                print(f"\rGenerated:{i}", end='', flush=True)

    def __calculate_place_id(self, csv_thumbnails_paths):
        # Assumption the regions are in the same csv file
        for csv_path in csv_thumbnails_paths:
            df = pd.read_csv(csv_path)
            df['place_id'] = pd.factorize(df[['LT_lat', 'LT_lon', 'RB_lat', 'RB_lon']].apply(tuple, axis=1))[0]
            df.to_csv(csv_path, index=False)


    def generate_thumbnails(self):
        for map in self.satellite_map_names:
            print(f"Processing map: {map.friendly_name}")
            coords = self.__get_map_coordinates_csv(map.csv_path, map.map_name)
            map.set_coordinates(lt_lat=coords.lt_lat, lt_lon=coords.lt_lon, rb_lat=coords.rb_lat,rb_lon=coords.rb_lon)
            self.__crop_map(map)
            print("Processed")

        self.csv_thumbnails_paths = self.get_csv_thumbnails_paths()
        self.__calculate_place_id(self.csv_thumbnails_paths)
        print("Generation thumbnails completed!")

    def get_csv_thumbnails_paths(self):
        # If we have already prepared thumbnails, we can just take csv_thumbnails_paths without requirement to generate them. 
        return list(set([obj.thumbnails_satellite_csv_output_path for obj in self.satellite_map_names]))