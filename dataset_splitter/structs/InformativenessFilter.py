import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Union
import numpy as np
import cv2
import torch
import torchvision.models as models
from torchvision import transforms as T
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import cdist

IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
VIT_MEAN_STD = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}

@dataclass
class FilterResult:
    is_informative: bool
    votes_for_rejection: int
    votes_required: int
    details: dict
    
    def __repr__(self):
        status = "✓ KEEP" if self.is_informative else "✗ REJECT"
        return f"{status} | Votes for reject: {self.votes_for_rejection}/{self.votes_required} | {self.details}"
    
class InformativenessFilter:
    
    def __init__(self, votes_to_reject: int = 3):

        self.votes_to_reject = votes_to_reject
    
    
    def _check_low_gradient_variance(self, gray: np.ndarray):
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        
        mean_grad = gradient_mag.mean()
        std_grad = gradient_mag.std()
        
        if mean_grad < 1.0:
            return True, 0.0
        
        cv = std_grad / mean_grad
    
        is_suspicious = cv < 0.8 and mean_grad < 15
        
        return is_suspicious, cv
    
    def _check_low_local_contrast(self, gray: np.ndarray):
        block_size = 32
        h, w = gray.shape
        
        high_contrast_blocks = 0
        total_blocks = 0
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                local_std = block.std()
                total_blocks += 1
                
                if local_std > 20:
                    high_contrast_blocks += 1
        
        if total_blocks == 0:
            return True, 0.0
        
        ratio = high_contrast_blocks / total_blocks
        
        is_suspicious = ratio < 0.05
        
        return is_suspicious, ratio
    
    def _check_low_edge_density(self, gray: np.ndarray):
        median_val = np.median(gray)
        lower = int(max(0, 0.66 * median_val))
        upper = int(min(255, 1.33 * median_val))
        
        edges = cv2.Canny(gray, lower, upper)
        edge_density = np.sum(edges > 0) / edges.size
        
        is_suspicious = edge_density < 0.01
        
        return is_suspicious, edge_density
    
    def _check_low_entropy_blocks(self, gray: np.ndarray):
        block_size = 64
        h, w = gray.shape
        
        entropies = []
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                
                hist, _ = np.histogram(block.ravel(), bins=64, range=(0, 256))
                hist = hist[hist > 0]
                hist = hist / hist.sum()
                entropy = -np.sum(hist * np.log2(hist))
                entropies.append(entropy)
        
        if not entropies:
            return True, 0.0
        
        mean_entropy = np.mean(entropies)
        
        is_suspicious = mean_entropy < 3.5
        
        return is_suspicious, mean_entropy
    
    def _check_extreme_uniformity(self, gray: np.ndarray):

        mean_val = gray.mean()
        std_val = gray.std()
        
        if mean_val < 1:
            return True, 0.0
        
        cv = std_val / mean_val
        
        is_suspicious = cv < 0.1
        
        return is_suspicious, cv
    
    def _check_has_strong_edges(self, gray: np.ndarray) -> bool:
        edges = cv2.Canny(gray, 100, 200)
        
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )
        
        if lines is not None and len(lines) > 5:
            return True
        
        return False

    def _check_has_corners(self, gray: np.ndarray) -> bool:
        maxCorners = 300

        g1 = cv2.GaussianBlur(gray, (5, 5), 0)
        c1 = cv2.goodFeaturesToTrack(
            g1,
            maxCorners=maxCorners,
            qualityLevel=0.08,
            minDistance=20,
            blockSize=5,
            useHarrisDetector=True,
            k=0.04
        )
        if c1 is None:
            return False
        n1 = len(c1)
        if n1 < 15:
            return False

        if n1 >= 0.90 * maxCorners:
            return False

        h, w = gray.shape
        pts = c1.reshape(-1, 2)

        corner_density = n1 / (gray.size / 10000.0)

        if corner_density > 60:
            return False

        grid = np.zeros((4, 4), dtype=np.int32)
        for x, y in pts:
            gi = min(3, int(y / h * 4))
            gj = min(3, int(x / w * 4))
            grid[gi, gj] += 1

        p = grid.flatten().astype(np.float32)
        p = p / (p.sum() + 1e-6)
        entropy = -(p * np.log(p + 1e-12)).sum() / np.log(len(p))  # 0..1

        if entropy > 0.92:
            return False

        if p.max() < 0.18:
            return False

        g2 = cv2.GaussianBlur(gray, (11, 11), 0)
        c2 = cv2.goodFeaturesToTrack(
            g2,
            maxCorners=maxCorners,
            qualityLevel=0.08,
            minDistance=20,
            blockSize=5,
            useHarrisDetector=True,
            k=0.04
        )
        n2 = 0 if c2 is None else len(c2)
        ratio = n2 / (n1 + 1e-6)

        if n2 < 8:
            return False
        if ratio < 0.20:
            return False

        return True


    def _check_has_significant_contours(self, gray: np.ndarray) -> bool:
        edges = cv2.Canny(gray, 30, 100)

        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False

        significant = 0
        min_area = gray.size * 0.0005

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:

                x, y, w, h = cv2.boundingRect(cnt)
                aspect = max(w, h) / (min(w, h) + 1)
                if aspect < 10:
                    significant += 1

        return significant >= 2
    
    def _check_has_texture_variation(self, gray: np.ndarray):
        h, w = gray.shape
        block = 32
        step = 32

        gh = (h - block) // step + 1
        gw = (w - block) // step + 1
        if gh < 3 or gw < 3:
            return False

        var_grid = np.zeros((gh, gw), dtype=np.float32)

        for i in range(gh):
            for j in range(gw):
                y = i * step
                x = j * step
                b = gray[y:y+block, x:x+block]
                var_grid[i, j] = b.var()

        med = float(np.median(var_grid))
        if med < 1.0:
            return False

        high = var_grid > med * 2.5
        low = var_grid < med * 0.3
        out = high | low

        outlier_ratio = out.mean()
        if outlier_ratio < 0.10:
            return False
        if outlier_ratio > 0.35:
            return False
        high_ratio = high.mean()
        if high_ratio < 0.08:
            return False

        mask = high.astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        if num <= 1:
            return False

        sizes = stats[1:, cv2.CC_STAT_AREA]
        largest = int(sizes.max())
        comp_count = int(num - 1)
        total_high_cells = int(high.sum())
        if largest < 4:
            return False

        if total_high_cells < 6:
            return False

        if comp_count > 25 and largest < 8:
            return False

        idx = 1 + int(np.argmax(sizes))
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect > 8 and largest > 8:
            return False

        edges = cv2.Canny(gray, 30, 100)
        edge_density = (edges > 0).mean()
        if edge_density < 0.012:
            return False

        return True


    def _check_has_straight_lines(self, gray: np.ndarray) -> bool:

        edges = cv2.Canny(gray, 30, 100)
    
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=30,
            minLineLength=40,
            maxLineGap=15
        )
    
        if lines is None:
            return False
    
        h, w = gray.shape
        long_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length > 50:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                long_lines.append((cx, cy, length))
    
        if len(long_lines) < 3:
            return False
    
        grid = np.zeros((3, 3), dtype=int)
        
        for cx, cy, _ in long_lines:
            gi = min(2, int(cy / h * 3))
            gj = min(2, int(cx / w * 3))
            grid[gi, gj] += 1
        
        cells_with_lines = np.sum(grid > 0)
        
        if cells_with_lines < 3:
            return False
        
        total_length = sum(l for _, _, l in long_lines)
        min_total_length = min(h, w) * 1.5  
        
        if total_length < min_total_length:
            return False
    
        return True
    
    def analyze(self, img_path) -> FilterResult:

        if isinstance(img_path, (str, Path)):
            img = cv2.imread(str(img_path))
            if img is None:
                return FilterResult(
                    is_informative=False,
                    votes_for_rejection=999,
                    votes_required=self.votes_to_reject,
                    details={"error": "cannot_read"}
                )
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        votes = 0
        details = {}
        
        suspicious, value = self._check_low_gradient_variance(gray)
        details["gradient_cv"] = round(value, 3)
        if suspicious:
            votes += 1
            details["gradient_cv_flag"] = True
        
        suspicious, value = self._check_low_local_contrast(gray)
        details["high_contrast_ratio"] = round(value, 3)
        if suspicious:
            votes += 1
            details["local_contrast_flag"] = True
        
        suspicious, value = self._check_low_edge_density(gray)
        details["edge_density"] = round(value, 4)
        if suspicious:
            votes += 1
            details["edge_density_flag"] = True
        
        suspicious, value = self._check_low_entropy_blocks(gray)
        details["mean_block_entropy"] = round(value, 3)
        if suspicious:
            votes += 1
            details["entropy_flag"] = True
        
        suspicious, value = self._check_extreme_uniformity(gray)
        details["global_cv"] = round(value, 3)
        if suspicious:
            votes += 1
            details["uniformity_flag"] = True

        has_veto = False
        
        if votes >= self.votes_to_reject:
            if self._check_has_strong_edges(gray):
                has_veto = True
                details["veto"] = "strong_edges"
            elif self._check_has_corners(gray):
                has_veto = True
                details["veto"] = "corners"
            elif self._check_has_straight_lines(gray):
                has_veto = True
                details["veto"] = "straight_lines"
            elif self._check_has_texture_variation(gray):
                has_veto = True
                details["veto"] = "texture_variation"

        if has_veto:
            is_informative = True
        else:
            is_informative = votes < self.votes_to_reject
        
        return FilterResult(
            is_informative=is_informative,
            votes_for_rejection=votes,
            votes_required=self.votes_to_reject,
            details=details
        )
    
    def __call__(self, img_path_or_array: Union[str, Path, np.ndarray]) -> bool:
        return self.analyze(img_path_or_array).is_informative



    # def analyze(self, img_path):
    #     conditions_met = 0
    #     checks = []
    #     img = cv2.imread(str(img_path))
    #     if img is None:
    #         return False




def test_informativeness_filter():
    
    filter = InformativenessFilter(votes_to_reject=3)
    
    should_reject = [
        '/workspace/datasets/train_tiles_one_to_one/Changjiang-20/patch__29_73744038__115_9747973__0c77db0211bf4fe7a54442862e9cf281_1.jpg',
        '/workspace/datasets/train_tiles_one_to_one/Changjiang-20/patch__29_73492178__115_9768595__84ec621b756e4a29ab3b2831e7aa0e28_1.jpg',
        '/workspace/datasets/train_tiles_one_to_one/Changjiang-20/patch__29_72993132__115_976848__2505bd532f794868b1ef681dac0d9894_1.jpg',
        '/workspace/datasets/train_tiles_one_to_one/Changjiang-20/patch__29_73815084__115_97474__c2ab27a808cd40e69b48f46266e46eb5_1.jpg',
        '/workspace/datasets/train_tiles_one_to_one/Changjiang-20/patch__29_72051132__115_9809645__502297e11d2846c7aee4baa9cf2303ca_1.jpg',
        '/workspace/datasets/train_tiles_one_to_one/Changjiang-20/patch__29_73102325__115_9747858__5a390fd76d0c490a8609d2311bae5f0a_1.jpg',
        '/workspace/datasets/train_tiles_one_to_one/Changjiang-20/patch__29_73476651__115_9810332__43276595061a45639bf8431938f12e91_1.jpg'
    ]
    
    should_keep = [
        '/workspace/datasets/train_tiles_one_to_one/Changjiang-20/patch__29_74456797__115_9747744__2bbf5e63724049488a595c8f8eaea944_1.jpg',
        '/workspace/datasets/train_tiles_one_to_one/Changjiang-20/patch__29_73886704__115_97474__c164f603e7aa42fdad00a65da0528891_1.jpg'

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



# test_informativeness_filter()

# filter = InformativenessFilter(votes_to_reject=3)

# test_informativeness_filter()

img_path = '/workspace/datasets/train_tiles_one_to_one/Changjiang-20/patch__29_73476651__115_9810332__43276595061a45639bf8431938f12e91_1.jpg'


img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # # # Zakoduj jako PNG → base64
_, buffer = cv2.imencode('.png', gray)
b64 = base64.b64encode(buffer).decode('utf-8')
# # result = filter.analyze(img_path)
print(b64)

# # test_informativeness_filter()