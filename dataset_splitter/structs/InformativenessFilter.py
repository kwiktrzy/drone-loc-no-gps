import base64
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Union
import numpy as np
import cv2
import shutil
import torch
import torchvision.models as models
from torchvision import transforms as T
import os
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import cdist

IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
VIT_MEAN_STD = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}


@dataclass
class FilterResult:
    is_informative: bool
    details: dict

    # def __repr__(self):
    #     status = "✓ KEEP" if self.is_informative else "✗ REJECT"
    #     return f"{status} | {self.details}"


class InformativenessFilter:
    def __init__(self):
        self.THRESH_DOMAIN_SPLIT = 3.0
        self.WATER_MIN_LOUD = 5.5
        self.WATER_MIN_RATIO = 6.0
        self.LAND_MIN_LOUD = 12.0

    def _check_linear_rescue(self, gray: np.ndarray) -> bool:
        h, w = gray.shape

        pre_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(pre_blur)

        edges = cv2.Canny(enhanced, 30, 90)

        min_len = int(min(h, w) * 0.60)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 40, minLineLength=min_len, maxLineGap=15
        )

        if lines is not None:
            valid_lines = sum(
                1
                for line in lines
                if np.sqrt(
                    (line[0][2] - line[0][0]) ** 2 + (line[0][3] - line[0][1]) ** 2
                )
                > min_len
            )
            if valid_lines >= 1:
                return True

        return False

    def analyze(self, img_path_or_array: Union[str, Path, np.ndarray]) -> FilterResult:
        if isinstance(img_path_or_array, (str, Path)):
            img = cv2.imread(str(img_path_or_array))
            if img is None:
                return FilterResult(False, {"error": "read_error"})
        else:
            img = img_path_or_array

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_float = gray.astype(np.float32)

        blur = cv2.blur(gray_float, (25, 25))
        blur_sq = cv2.blur(gray_float**2, (25, 25))
        variance = blur_sq - blur**2
        variance[variance < 0] = 0
        std_map = np.sqrt(variance)

        quiet = np.percentile(std_map, 5)
        loud = np.percentile(std_map, 95)
        ratio = loud / (quiet + 1.0)

        is_water_domain = quiet < self.THRESH_DOMAIN_SPLIT

        stats = {
            "quiet": round(quiet, 2),
            "loud": round(loud, 2),
            "ratio": round(ratio, 2),
            "domain": "WATER" if is_water_domain else "LAND",
            "linear_rescue": False,
        }

        is_kept = False
        reason = ""

        if is_water_domain:
            if loud >= self.WATER_MIN_LOUD and ratio >= self.WATER_MIN_RATIO:
                is_kept = True
                reason = "High Contrast Object on Water"
            elif loud > 30.0:
                is_kept = True
                reason = "Massive Structure"
            else:
                reason = "Flat Water / Floating Noise"
        else:
            if loud >= self.LAND_MIN_LOUD:
                is_kept = True
                reason = "Structural Terrain"
            else:
                reason = "Barren Terrain/Sand"

        if not is_kept:

            if ratio > 2.0:
                if self._check_linear_rescue(gray):
                    stats["linear_rescue"] = True
                    return FilterResult(
                        True, {**stats, "reason": "Linear Infrastructure Rescue"}
                    )

        return FilterResult(is_kept, {**stats, "reason": reason})

    def __call__(self, img_path_or_array) -> bool:
        return self.analyze(img_path_or_array).is_informative

        # def analyze(self, img_path):
        #     conditions_met = 0
        #     checks = []
        #     img = cv2.imread(str(img_path))
        #     if img is None:
        #         return False


def test_informativeness_filter():

    filter = InformativenessFilter()

    should_reject = [
        "/home/user/PycharmProjects/mgr-repack/debug-04022026/Changjiang-20/patch__29_73744038__115_9747973__0c77db0211bf4fe7a54442862e9cf281_1.jpg",
        "/home/user/PycharmProjects/mgr-repack/debug-04022026/Changjiang-20/patch__29_73492178__115_9768595__84ec621b756e4a29ab3b2831e7aa0e28_1.jpg",
        "/home/user/PycharmProjects/mgr-repack/debug-04022026/Changjiang-20/patch__29_72993132__115_976848__2505bd532f794868b1ef681dac0d9894_1.jpg",
        "/home/user/PycharmProjects/mgr-repack/debug-04022026/Changjiang-20/patch__29_73815084__115_97474__c2ab27a808cd40e69b48f46266e46eb5_1.jpg",
        "/home/user/PycharmProjects/mgr-repack/debug-04022026/Changjiang-20/patch__29_72051132__115_9809645__502297e11d2846c7aee4baa9cf2303ca_1.jpg",
        "/home/user/PycharmProjects/mgr-repack/debug-04022026/Changjiang-20/patch__29_73102325__115_9747858__5a390fd76d0c490a8609d2311bae5f0a_1.jpg",
        "/home/user/PycharmProjects/mgr-repack/debug-04022026/Changjiang-20/patch__29_73476651__115_9810332__43276595061a45639bf8431938f12e91_1.jpg",
        "/home/user/PycharmProjects/mgr-repack/debug-04022026/Changjiang-20/patch__29_740414__115_981184__29_737624__115_984481__975ff322.jpg",
        "/home/user/PycharmProjects/mgr-repack/debug-04022026/Taizhou-1/patch__29_740414__115_981184__29_737624__115_984481__975ff322.jpg",
    ]

    should_keep = [
        "/home/user/PycharmProjects/mgr-repack/debug-04022026/Changjiang-20/patch__29_74456797__115_9747744__2bbf5e63724049488a595c8f8eaea944_1.jpg",
        "/home/user/PycharmProjects/mgr-repack/debug-04022026/Changjiang-20/patch__29_73886704__115_97474__c164f603e7aa42fdad00a65da0528891_1.jpg",
        "/home/user/PycharmProjects/mgr-repack/debug-04022026/Changjiang-20/patch__29_771096__115_974151__29_768306__115_977448__3d8ab068.jpg",
        "/home/user/PycharmProjects/mgr-repack/debug-04022026/Taizhou-1/patch__32_293884__119_818415__32_291135__119_821480__a63a8265.jpg",
        "/home/user/PycharmProjects/mgr-repack/debug-04022026/Taizhou-1/patch__32_293884__119_819455__32_291135__119_822521__06b20594.jpg",
        "/home/user/PycharmProjects/mgr-repack/debug-04022026/Taizhou-1/patch__32_293884__119_820496__32_291135__119_823562__e63d25e9.jpg",
        "/home/user/PycharmProjects/mgr-repack/debug-04022026/Taizhou-1/patch__32_294818__119_818415__32_292068__119_821480__ebfd65d5.jpg",
        "/home/user/PycharmProjects/mgr-repack/debug-04022026/Taizhou-1/patch__32_30487883__119_8881757__1885b3010d684bcc95bd6ec423e51100_1.jpg",
        "/home/user/PycharmProjects/mgr-repack/debug-04022026/Taizhou-1/patch__32_30540022__119_8873736__8c3c17770e4f45aa8935d968dfb7784d_1.jpg",
        "/home/user/PycharmProjects/mgr-repack/debug-04022026/Taizhou-1/patch__32_30616225__119_8903644__238d5c10ab62495898a5ca16700a5597_1.jpg",
        "/home/user/PycharmProjects/mgr-repack/debug-04022026/Taizhou-1/patch__32_34891636__119_8288402__2f599f5ded494bd78f66840cc684ee32_1.jpg",
        "/home/user/PycharmProjects/mgr-repack/debug-04022026/Taizhou-1/patch__32_30564659__119_8911665__a44a3025b7044460bceda053358ef3ce_1.jpg",
        "/home/user/PycharmProjects/mgr-repack/debug-04022026/Taizhou-1/patch__32_34912263__119_8160174__7950b84fccca4a3287c888e76ae45743_1.jpg",
    ]

    print("=" * 60)
    print("TESTING REJECT (should be False)")
    print("=" * 60)
    for path in should_reject:
        result = filter.analyze(path)
        status = "✓ PASS" if not result.is_informative else "✗ FAIL"
        print(f"{status}: {Path(path).name}")
        print(f"       {result}")
        assert not result.is_informative, f"Should REJECT: {path}"

    print("\n" + "=" * 60)
    print("TESTING KEEP (should be True)")
    print("=" * 60)
    for path in should_keep:
        result = filter.analyze(path)
        status = "✓ PASS" if result.is_informative else "✗ FAIL"
        print(f"{status}: {Path(path).name}")
        print(f"       {result}")
        assert result.is_informative, f"Should KEEP: {path}"

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


test_informativeness_filter()

# filter = InformativenessFilter()
# folder = "/home/user/PycharmProjects/mgr-repack/debug-04022026/Taizhou-1"
# destination = "/home/user/PycharmProjects/mgr-repack/debug-04022026/to_delete/"
# paths = [os.path.join(folder, f) for f in os.listdir(folder)]
# for path in paths:
#     result = filter.analyze(path)
#     if not result.is_informative:
#         shutil.copy2(path, destination)
#     print(result)
