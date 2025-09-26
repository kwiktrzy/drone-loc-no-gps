from typing import List
import pandas as pd

# place_id_round = 4 # accuracy 3 ~ 100m,  4 ~ 10m, 5 ~ 1m

class AbstractGenerator:
    def __init__(self,
                 csv_thumbnails_paths: List[str] = None,
                 place_id_round=4):
        self.csv_thumbnails_paths = csv_thumbnails_paths
        self.place_id_round=place_id_round
        self.__calculate_place_id()


    def __calculate_place_id(self):
        # Assumption the regions are in the same csv file
        for csv_path in self.csv_thumbnails_paths:
            df = pd.read_csv(csv_path)

            df['lon_round'] = df['lon'].round(self.place_id_round)
            df['lat_round'] = df['lat'].round(self.place_id_round)

            df['place_id'] = pd.factorize(df[['lon_round', 'lat_round']].apply(tuple, axis=1))[0]
            df.to_csv(csv_path, index=False)
            print("Calculated place_ids!")
