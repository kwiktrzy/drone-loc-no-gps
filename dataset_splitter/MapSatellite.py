from collections import namedtuple

Coordinates = namedtuple('Coordinates', ['lt_lat', 'lt_lon', 'rb_lat', 'rb_lon'])


class MapSatellite:
    def __init__(self,
                 csv_path,
                 thumbnails_satellite_csv_output_path,
                 map_tif_path,
                 map_name,# must be mapname from the row in file selected in csv_path
                 region_name,
                 friendly_name=None):
        self.csv_path = csv_path
        self.thumbnails_satellite_csv_output_path=thumbnails_satellite_csv_output_path
        self.map_tif_path=map_tif_path
        self.map_name = map_name
        self.region_name = region_name
        self.friendly_name = friendly_name if friendly_name else map_name
        self.coordinates = None


    def set_coordinates(self, lt_lat: float, lt_lon: float, rb_lat: float, rb_lon: float):
        self.coordinates = Coordinates(lt_lat, lt_lon, rb_lat, rb_lon)