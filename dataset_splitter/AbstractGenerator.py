from dataclasses import dataclass
from typing import Tuple
import utm


@dataclass
class UTMCoordinates:
    e: float
    n: float
    zone: str


@dataclass
class GPSCoordinates:
    lt_lat: float
    lt_lon: float
    rb_lat: float
    rb_lon: float


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
