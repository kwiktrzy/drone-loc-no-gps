import os
import PIL.Image
import pandas as pd
import uuid
import cv2


class UavSmallerCropGenerator:
    def __init__(
        self,
        csv_path="",
        cropped_uav_csv_output_path="",
        cropped_output_dir="",
        uav_images_dir="",
        region_name="",
        friendly_name="",
        grid_size=1,
        patch_format_compress=(".jpg", 100),
        target_width_size=224,
        target_height_size=224,
    ):
        self.csv_path = csv_path
        self.cropped_uav_csv_output_path = cropped_uav_csv_output_path
        self.cropped_output_dir = cropped_output_dir
        self.uav_images_dir = uav_images_dir
        self.region_name = region_name
        self.friendly_name = friendly_name
        self.patch_format_compress = patch_format_compress
        self.target_width_size = target_width_size
        self.target_height_size = target_height_size
        self.grid_size = grid_size
        self.csv_header_written = True

    def generate_tiles(self):
        dir_output = f"{self.cropped_output_dir}/{self.region_name}"
        if not os.path.exists(dir_output):
            os.makedirs(dir_output)

        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            print(f"Error unable to find csv: {self.csv_path}")
            return
            
        total_images = len(df)
        print(f"Starting generate crops {total_images} images...")

        for index, row in df.iterrows():
            filename = row["filename"]
            input_path = f"{self.uav_images_dir}/{filename}"

            patch_dir_base = (
                f"patch__{row['lat']}__{row['lon']}__{uuid.uuid4().hex}"
            )
            patch_dir_ext_base = patch_dir_base.replace('.', '_')

            if self.__process_image(input_path, f"{dir_output}/{patch_dir_ext_base}", row):
                print(f"\rGenerated uav crops:{index + 1}/{total_images}: {filename}", end="", flush=True)
            else:
                print(f"\nSkipped: {filename}")

        print("\n Generation tiles completed!")


    def __process_image(self, input_path, base_output_path, source_row):
        try:
            img = cv2.imread(input_path)
            if img is None:
                print(f"\nError. Unable to read image: {input_path}")
                return False 

            height, width, channels = img.shape
            
            crop_height = height // self.grid_size
            crop_width = width // self.grid_size
            
            if crop_height == 0 or crop_width == 0:
                print(f"\Error. Picture {input_path} (size: {width}x{height}) is too small for grid: {self.grid_size}x{self.grid_size}")
                return False

            indx = 0
            patch_format, patch_quality = self.patch_format_compress
            target_h = self.target_height_size
            target_w = self.target_width_size

            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    indx += 1
                    y_lt = y * crop_height
                    y_rb = (y + 1) * crop_height
                    x_lt = x * crop_width
                    x_rb = (x + 1) * crop_width
                    
                    crop = img[y_lt:y_rb, x_lt:x_rb]

                    try:
                        src_h, src_w = crop.shape[:2]
                        scale = max(target_w / src_w, target_h / src_h)
                        new_w = int(src_w * scale)
                        new_h = int(src_h * scale)
                        resized_full = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        y_start = (new_h - target_h) // 2
                        x_start = (new_w - target_w) // 2
                        resized_crop = resized_full[y_start : y_start + target_h, x_start : x_start + target_w]
                        
                    except cv2.error as e:
                         print(f"\nError resize for patch {indx} from {input_path}: {e}")
                         continue 

                    full_patch_path = f"{base_output_path}_{indx}{patch_format}"

                    if patch_format == ".png":
                        cv2.imwrite(
                            full_patch_path,
                            resized_crop,
                            [cv2.IMWRITE_PNG_COMPRESSION, patch_quality]
                        )
                    elif patch_format == ".jpg":
                        cv2.imwrite(
                            full_patch_path,
                            resized_crop,
                            [cv2.IMWRITE_JPEG_QUALITY, patch_quality]
                        )
                    else:
                        print("\nOH NO...u used unsupported tile format!")
                        return False

                    new_row = {
                        "img_path": full_patch_path, 
                        "LT_lat": 0, 
                        "LT_lon": 0,
                        "RB_lat": 0,
                        "RB_lon": 0,
                        "lon": source_row["lon"], 
                        "lat": source_row["lat"], 
                        "patch_width": self.target_width_size, 
                        "patch_height": self.target_height_size,
                        "region_name": self.region_name,
                        "friendly-name": f"{self.friendly_name}-{source_row['filename']}-{indx}",
                    }

                    self.__append_row_csv(new_row, self.cropped_uav_csv_output_path)

        except Exception as e:
            print(f"Error {input_path}: {e}")
            return False
        
        return True
    
    def __append_row_csv(self, row_data, csv_path):
        # ex. row = { 'id': '1', 'value': 10 }
        try:
            df_row = pd.DataFrame([row_data])
            file_exists = os.path.isfile(csv_path)
            
            write_header = not file_exists or not self.csv_header_written
            
            df_row.to_csv(csv_path, mode='a', header=write_header, index=False)

            if write_header:
                self.csv_header_written = True
        except Exception as e:
            print(f"\nCSV save error {csv_path}: {e}")
