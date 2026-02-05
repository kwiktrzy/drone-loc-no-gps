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

    def __repr__(self):
        status = "✓ KEEP" if self.is_informative else "✗ REJECT"
        return f"{status} | {self.details}"


class InformativenessFilter:
    def __init__(self):
        self.THRESH_LOUD_MIN = 5.5

        self.THRESH_RATIO_MIN = 6.0

        self.THRESH_LOUD_HUGE = 15.0

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

        stats = {
            "quiet": round(quiet, 2),
            "loud": round(loud, 2),
            "ratio": round(ratio, 2),
        }

        if loud < self.THRESH_LOUD_MIN:
            return FilterResult(False, stats)

        if loud > self.THRESH_LOUD_HUGE:
            return FilterResult(True, stats)

        if ratio > self.THRESH_RATIO_MIN:
            return FilterResult(True, stats)

        return FilterResult(False, stats)

    def __call__(self, img_path_or_array) -> bool:
        return self.analyze(img_path_or_array).is_informative

    # def analyze(self, img_path):
    #     conditions_met = 0
    #     checks = []
    #     img = cv2.imread(str(img_path))
    #     if img is None:
    #         return False


# def test_informativeness_filter():

#     filter = InformativenessFilter()

#     should_reject = [
#         "/home/user/PycharmProjects/mgr-repack/debug-04022026/Changjiang-20/patch__29_73744038__115_9747973__0c77db0211bf4fe7a54442862e9cf281_1.jpg",
#         "/home/user/PycharmProjects/mgr-repack/debug-04022026/Changjiang-20/patch__29_73492178__115_9768595__84ec621b756e4a29ab3b2831e7aa0e28_1.jpg",
#         "/home/user/PycharmProjects/mgr-repack/debug-04022026/Changjiang-20/patch__29_72993132__115_976848__2505bd532f794868b1ef681dac0d9894_1.jpg",
#         "/home/user/PycharmProjects/mgr-repack/debug-04022026/Changjiang-20/patch__29_73815084__115_97474__c2ab27a808cd40e69b48f46266e46eb5_1.jpg",
#         "/home/user/PycharmProjects/mgr-repack/debug-04022026/Changjiang-20/patch__29_72051132__115_9809645__502297e11d2846c7aee4baa9cf2303ca_1.jpg",
#         "/home/user/PycharmProjects/mgr-repack/debug-04022026/Changjiang-20/patch__29_73102325__115_9747858__5a390fd76d0c490a8609d2311bae5f0a_1.jpg",
#         "/home/user/PycharmProjects/mgr-repack/debug-04022026/Changjiang-20/patch__29_73476651__115_9810332__43276595061a45639bf8431938f12e91_1.jpg",
#         "/home/user/PycharmProjects/mgr-repack/debug-04022026/Changjiang-20/patch__29_740414__115_981184__29_737624__115_984481__975ff322.jpg",
#     ]

#     should_keep = [
#         "/home/user/PycharmProjects/mgr-repack/debug-04022026/Changjiang-20/patch__29_74456797__115_9747744__2bbf5e63724049488a595c8f8eaea944_1.jpg",
#         "/home/user/PycharmProjects/mgr-repack/debug-04022026/Changjiang-20/patch__29_73886704__115_97474__c164f603e7aa42fdad00a65da0528891_1.jpg",
#         "/home/user/PycharmProjects/mgr-repack/debug-04022026/Changjiang-20/patch__29_771096__115_974151__29_768306__115_977448__3d8ab068.jpg",
#     ]

#     print("=" * 60)
#     print("TESTING REJECT (should be False)")
#     print("=" * 60)
#     for path in should_reject:
#         result = filter.analyze(path)
#         status = "✓ PASS" if not result.is_informative else "✗ FAIL"
#         print(f"{status}: {Path(path).name}")
#         print(f"       {result}")
#         assert not result.is_informative, f"Should REJECT: {path}"

#     print("\n" + "=" * 60)
#     print("TESTING KEEP (should be True)")
#     print("=" * 60)
#     for path in should_keep:
#         result = filter.analyze(path)
#         status = "✓ PASS" if result.is_informative else "✗ FAIL"
#         print(f"{status}: {Path(path).name}")
#         print(f"       {result}")
#         assert result.is_informative, f"Should KEEP: {path}"

#     print("\n" + "=" * 60)
#     print("ALL TESTS PASSED!")
#     print("=" * 60)


# test_informativeness_filter()

# filter = InformativenessFilter()
# folder = "/home/user/PycharmProjects/mgr-repack/debug-04022026/Changjiang-20"
# destination = "/home/user/PycharmProjects/mgr-repack/debug-04022026/to_delete/"
# paths = [os.path.join(folder, f) for f in os.listdir(folder)]
# for path in paths:
#     result = filter.analyze(path)
#     if not result.is_informative:
#         shutil.copy2(path, destination)
#     print(result)
