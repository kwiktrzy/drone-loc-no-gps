from typing import List
import pandas as pd
import numpy as np
import utm

# place_id_round = 4 # accuracy 3 ~ 100m,  4 ~ 10m, 5 ~ 1m

class PlaceIdGenerator:
    def __init__(self,
                 csv_thumbnails_paths: List[str] = [],
                 place_id_round=4,
                 is_validation_set=False,
                 force_regenerate=False):
        self.csv_thumbnails_paths = csv_thumbnails_paths
        self.place_id_round=place_id_round
        self.is_validation_set = is_validation_set
        self.force_regenerate = force_regenerate
        self.__calculate_place_id_for_all()

    def get_utm(self, row):
        e, n, zone_num, zone_let = utm.from_latlon(row.lat, row.lon)
        zone = f'{zone_num}{zone_let or ""}'

        return {'e_utm': e, 'n_utm': n, 'zone_utm': zone}

    def __calculate_place_id_for_all(self):

        for csv_path in self.csv_thumbnails_paths:
            df = pd.read_csv(csv_path)

            if 'place_id' in df.columns and not self.force_regenerate:
                print(f"Skipping Place ID generation for {csv_path}, 'place_id' column already exists.")
                continue
            print(f"Generating Place ID for {csv_path}...")

            if not {'e_utm', 'n_utm', 'zone_utm'}.issubset(df.columns):
                df = df.join(df.apply(self.get_utm, axis=1).apply(pd.Series))

            name = df['friendly-name']
            sat_mask = name.str.contains('satellite', case=False, na=False)
            uav_mask = name.str.contains('uav', case=False, na=False)

            df.loc[sat_mask, 'place_id'] = df.index.to_series()[sat_mask].astype('Int64')

            for col in ['uav_to_nearest_sat_m', 'uav_to_second_sat_m', 'nearest_sats_pair_distance_m']:
                if col not in df.columns:
                    df[col] = np.nan

            good_coords = df['e_utm'].notna() & df['n_utm'].notna() & df['zone_utm'].notna()

            sub = df[good_coords & (sat_mask | uav_mask)]
            for zone, idxs in sub.groupby('zone_utm').groups.items():
                idxs = pd.Index(idxs)
                sat_idx = idxs[sat_mask.loc[idxs].to_numpy()]
                uav_idx = idxs[uav_mask.loc[idxs].to_numpy()]

                if len(sat_idx) == 0 or len(uav_idx) == 0:
                    continue

                sat_xy = df.loc[sat_idx, ['e_utm', 'n_utm']].to_numpy()
                uav_xy = df.loc[uav_idx, ['e_utm', 'n_utm']].to_numpy()


                diff = uav_xy[:, None, :] - sat_xy[None, :, :]
                dist2 = np.einsum('ijk,ijk->ij', diff, diff)  

                U = uav_xy.shape[0]
                S = sat_xy.shape[0]
                K = 2 if S >= 2 else 1


                nearest_rel = np.argpartition(dist2, K - 1, axis=1)[:, :K]  # (U, K)
                rows = np.arange(U)[:, None]
                nearest_rel_sorted = nearest_rel[rows, np.argsort(dist2[rows, nearest_rel], axis=1)]

                sat_idx_arr = sat_idx.to_numpy()
                nearest_sat_global = sat_idx_arr[nearest_rel_sorted[:, 0]]
                
                df.loc[uav_idx, 'place_id'] = df.loc[nearest_sat_global, 'place_id'].to_numpy()

                d1 = np.sqrt(dist2[np.arange(U), nearest_rel_sorted[:, 0]])
                df.loc[uav_idx, 'uav_to_nearest_sat_m'] = d1

                if K == 2:
                    d2 = np.sqrt(dist2[np.arange(U), nearest_rel_sorted[:, 1]])
                    df.loc[uav_idx, 'uav_to_second_sat_m'] = d2

                    sat1_xy = sat_xy[nearest_rel_sorted[:, 0]]
                    sat2_xy = sat_xy[nearest_rel_sorted[:, 1]]
                    pair = np.sqrt(np.sum((sat1_xy - sat2_xy) ** 2, axis=1))
                    df.loc[uav_idx, 'nearest_sats_pair_distance_m'] = pair


            print(f'\n Mean distance between satelite and uav: {df["uav_to_nearest_sat_m"].mean()} m');

            ranking = df['place_id'].value_counts()
            print(f'\n Ranking {csv_path}:\n{ranking}\n')
            print(f"\fRows: {len(csv_path)}")

            if not self.is_validation_set:
                print(f'\nRemoving single place_id entries for training set...')
                df = df[df['place_id'].duplicated(keep=False)]
                print(f"\fRows after clear up: {len(df)}")
            else:
                print("\nSkipping removal of single place_id entries for validation set.")

            df.to_csv(csv_path, index=False)
        print("\nPlace ID generation finished.")