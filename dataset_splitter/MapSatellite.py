from collections import namedtuple
from pathlib import Path

Coordinates = namedtuple("Coordinates", ["lt_lat", "lt_lon", "rb_lat", "rb_lon"])


class MapSatellite:
    def __init__(
        self,
        csv_path,
        tiles_satellite_csv_output_path,
        map_tif_path,
        region_name,
        friendly_name=None,
    ):
        self.csv_path = csv_path
        self.tiles_satellite_csv_output_path = tiles_satellite_csv_output_path
        self.map_tif_path = map_tif_path
        self.map_name = Path(map_tif_path).name
        self.region_name = region_name
        self.friendly_name = friendly_name if friendly_name else self.map_name
        self.coordinates = None

    def set_coordinates(
        self, lt_lat: float, lt_lon: float, rb_lat: float, rb_lon: float
    ):
        self.coordinates: Coordinates = Coordinates(lt_lat, lt_lon, rb_lat, rb_lon)
