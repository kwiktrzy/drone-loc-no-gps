from dataclasses import dataclass


@dataclass
class GPSCoordinates:
    lt_lat: float
    lt_lon: float
    rb_lat: float
    rb_lon: float
