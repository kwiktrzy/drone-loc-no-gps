import os
import PIL.Image
import pandas as pd
import uuid
import cv2


# crop_stale = 0.9 -> small zoom
# crop_stale = 0.7 -> normal zoom
# crop_stale = 0.5 -> big zoom


class UavCropGenerator:
    def __init__(
        self,
        csv_path="",
        cropped_uav_csv_output_path="",
        cropped_output_dir="",
        uav_images_dir="",
        region_name="",
        friendly_name="",
        patch_format_compress=(".jpg", 70),
        target_width_size=224,
        target_height_size=224,
        crop_scale=0.8,
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
        self.crop_scale = crop_scale

    def __process_image(self, input_path, output_path):
        try:
            img = cv2.imread(input_path)
            if img is None:
                print(f"\nError. Unable to read image: {input_path}")
                return

            width, height, channels = img.shape
            shorter_side = min(width, height)
            crop_size = shorter_side * self.crop_scale
            left = (width - crop_size) / 2
            top = (height - crop_size) / 2
            right = (width + crop_size) / 2
            bottom = (height + crop_size) / 2
            cropped_img = img[int(top) : int(bottom), int(left) : int(right)]
            resized_img = cv2.resize(
                cropped_img,
                dsize=(self.target_width_size, self.target_height_size),
                interpolation=cv2.INTER_LANCZOS4,
            )
            if self.patch_format_compress[0] == ".png":
                cv2.imwrite(
                    output_path,
                    resized_img,
                    [cv2.IMWRITE_PNG_COMPRESSION, self.patch_format_compress[1]],
                )
            elif self.patch_format_compress[0] == ".jpg":
                cv2.imwrite(
                    output_path,
                    resized_img,
                    [cv2.IMWRITE_JPEG_QUALITY, self.patch_format_compress[1]],
                )
            else:
                print("\nOH NO...u used unsupported tile format!")
                return False

        except Exception as e:
            print(f"Error {input_path}: {e}")
            return False

        return True

    def __append_row_csv(self, row_append, file_path):
        # ex. row = { 'id': '1', 'value': 10 }

        df = pd.DataFrame([row_append])
        df.to_csv(file_path, mode="a", index=False, header=False)

    def generate_tiles(self):
        dir_output = f"{self.cropped_output_dir}/{self.region_name}"
        if not os.path.exists(dir_output):
            os.makedirs(dir_output)

        df = pd.read_csv(self.csv_path)
        for index, row in df.iterrows():
            input_path = f"{self.uav_images_dir}/{row['filename']}"

            patch_dir = (
                f"{dir_output}/patch__{row['lat']}__{row['lon']}__{uuid.uuid4().hex}"
            )
            patch_dir_ext = (
                f"{patch_dir.replace('.', '_')}{self.patch_format_compress[0]}"
            )

            if self.__process_image(input_path, patch_dir_ext):
                row = {
                    "img_path": patch_dir_ext,
                    "LT_lat": 0,
                    "LT_lon": 0,
                    "RB_lat": 0,
                    "RB_lon": 0,
                    "lon": row["lon"],
                    "lat": row["lat"],
                    "patch_width": self.target_width_size,
                    "patch_height": self.target_height_size,
                    "region_name": self.region_name,
                    "friendly-name": self.friendly_name,
                }

                self.__append_row_csv(row, self.cropped_uav_csv_output_path)
                print(f"\rGenerated uav crops:{index}", end="", flush=True)

        print("\n Generation tiles completed!")
