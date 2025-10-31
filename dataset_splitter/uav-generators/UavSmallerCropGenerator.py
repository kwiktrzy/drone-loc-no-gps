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
        patch_format_compress=(".jpg", 70),
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

    def generate_tiles(self):
        dir_output = f"{self.cropped_output_dir}/{self.region_name}"
        if not os.path.exists(dir_output):
            os.makedirs(dir_output)

        df = pd.read_csv(self.csv_path)
        for index, row in df.iterrows():
            filename = row["filename"]
            input_path = f"{self.uav_images_dir}/{filename}"

            patch_dir = (
                f"{dir_output}/patch__{row['lat']}__{row['lon']}__{uuid.uuid4().hex}"
            )
            patch_dir_ext = (
                f"{patch_dir.replace('.', '_')}{self.patch_format_compress[0]}"
            )

    def __process_image(self, input_path, output_path):
        try:
            img = cv2.imread(input_path)
            if img is None:
                print(f"\nError. Unable to read image: {input_path}")
                return

            height, width, channels = img.shape
            single_picture_crop_num = 4
            for i in range(single_picture_crop_num):
                f1_x_lt = 0
                f1_y_lt = 0
                f1_x_rb = height / 2
                f1_y_rb = width / 2
                crop = img[f1_x_lt:f1_x_rb, f1_y_lt:f1_y_rb]

        except Exception as e:
            print(f"Error {input_path}: {e}")
            return False
