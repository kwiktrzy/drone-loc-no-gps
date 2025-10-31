from dataclasses import dataclass
from typing import Tuple
import utm
import os
import pandas as pd

from dataset_splitter.structs.MapSatellite import MapSatellite
from dataset_splitter.structs.datatypes import GPSCoordinates


@dataclass
class UTMCoordinates:
    e: float
    n: float
    zone: str


@dataclass
class UTMPatchCoordinates:
    lt_e: float
    lt_n: float
    rb_e: float
    rb_n: float
    center_e: float
    center_n: float


@dataclass
class Tile:
    id: int
    utm_centroid: UTMCoordinates
    width: int
    height: int
    friendly_name: str
    place_id: str
    img_path: str


@dataclass
class PixelBoundingBox:
    id: int
    lt_x: float
    lt_y: float
    rb_x: float
    rb_y: float
    m_width: int
    m_height: int


class AbstractGenerator:
    @staticmethod
    def gps_to_utm(lat: float, lon: float) -> UTMCoordinates:
        e, n, zone_num, zone_let = utm.from_latlon(lat, lon)
        zone = f'{zone_num}{zone_let or ""}'
        return UTMCoordinates(e, n, zone)

    @staticmethod
    def utm_to_gps(
        e: float, n: float, zone_num: int, zone_let: str = ""
    ) -> Tuple[float, float]:
        lat, lon = utm.to_latlon(e, n, zone_num, zone_let)
        return lat, lon

    @staticmethod
    def get_center(gps_coords: GPSCoordinates) -> Tuple[float, float]:
        lat = (gps_coords.lt_lat + gps_coords.rb_lat) / 2
        lon = (gps_coords.lt_lon + gps_coords.rb_lon) / 2
        return lat, lon

    @staticmethod
    def append_rows_csv(append_rows: list, file_path, is_recreate):
        # ex. row = { 'id': '1', 'value': 10 }
        is_file_exists = os.path.isfile(file_path)
        if is_file_exists and is_recreate:
            os.remove(file_path)
            is_file_exists = False
        df = pd.DataFrame(append_rows)
        df.to_csv(file_path, mode="a", index=False, header=not is_file_exists)

    @staticmethod
    def calculate_pixel_resolution(src, map: MapSatellite):
        lt_utm = AbstractGenerator.gps_to_utm(
            map.coordinates.lt_lat, map.coordinates.lt_lon
        )
        rb_utm = AbstractGenerator.gps_to_utm(
            map.coordinates.rb_lat, map.coordinates.rb_lon
        )

        width_meters = abs(lt_utm.e - rb_utm.e)
        height_meters = abs(lt_utm.n - rb_utm.n)

        meters_per_pixel_x = width_meters / src.width
        meters_per_pixel_y = height_meters / src.height

        return meters_per_pixel_x, meters_per_pixel_y

    @staticmethod
    def get_uav_square_meter():
        pass
