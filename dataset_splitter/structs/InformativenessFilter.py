import numpy as np
import cv2
import torch
import torchvision.models as models
from torchvision import transforms as T
from sklearn.ensemble import IsolationForest

IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
VIT_MEAN_STD = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}


class InformativenessFilter:
    def __init__(
        self,
        contamination=0.05,
        device="cpu",
        entropy_threshold: float = 4.5,
        mean_std=IMAGENET_MEAN_STD,
    ):
        self.device = device
        self.feature_extractor = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
        self.feature_extractor.fc = torch.nn.Identity()
        self.feature_extractor.eval().to(device)

        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize((224, 224)),
                T.Normalize(mean=mean_std["mean"], std=mean_std["std"]),
            ]
        )

        self.detector = IsolationForest(
            contamination=contamination, random_state=42, n_estimators=100
        )
        self._is_fitted = False
        self.entropy_threshold = entropy_threshold
        self.blue_ratio_threshold: float = 0.6
        self.edge_density_threshold: float = 0.02

    def _extract_features(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.feature_extractor(tensor)
        return features.cpu().numpy().flatten()

    def _compute_entropy(self, gray: np.ndarray):
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        return entropy

    def _compute_edge_density(self, gray: np.ndarray):
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        return edge_density

    def _compute_color_statistics(self, img_rgb: np.ndarray):
        mean_rgb = img_rgb.mean(axis=(0, 1))
        std_rgb = img_rgb.std(axis=(0, 1))

        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        mean_hsv = img_hsv.mean(axis=(0, 1))

        hue = img_hsv[:, :, 0]
        sat = img_hsv[:, :, 1]
        blue_mask = (hue > 90) & (hue < 130) & (sat > 30)
        blue_ratio = blue_mask.sum() / blue_mask.size

        sand_mask = (hue > 15) & (hue < 35) & (sat > 20)
        sand_ratio = sand_mask.sum() / sand_mask.size

        val = img_hsv[:, :, 2]
        white_mask = (sat < 30) & (val > 200)
        white_ratio = white_mask.sum() / white_mask.size

        return {
            "mean_r": mean_rgb[0],
            "mean_g": mean_rgb[1],
            "mean_b": mean_rgb[2],
            "std_r": std_rgb[0],
            "std_g": std_rgb[1],
            "std_b": std_rgb[2],
            "std_overall": std_rgb.mean(),
            "mean_saturation": mean_hsv[1] / 255.0,
            "mean_value": mean_hsv[2] / 255.0,
            "blue_ratio": blue_ratio,
            "sand_ratio": sand_ratio,
            "white_ratio": white_ratio,
        }

    def fit(self, sample_image_paths):
        """
        Call once on a representative batch (first 500-1000 tiles).
        Calibrates the model dataset.
        """
        # paths = list(Path("path/").glob("*.jpg"))[:1000]
        print(f"Fitting on {len(sample_image_paths)} samples...")
        features = []
        for path in sample_image_paths:
            img = cv2.imread(str(path))
            if img is not None:
                features.append(self._extract_features(img))
        if len(features) == 0:
            raise RuntimeError("0 features to fit!")
        self.detector.fit(features)
        self._is_fitted = True
        print("Filter calibrated")

    def analyze(self, img_path):
        checks = []
        img = cv2.imread(str(img_path))
        if img is None:
            return False

        if not self._is_fitted:
            raise RuntimeError("Call fit() first")

        features = self._extract_features(img)

        # IsolationForest: 1 = inlier (informative), -1 = outlier (low-info)
        is_informative = self.detector.predict([features])[0] == 1
        if not is_informative:
            checks.append(("low_informativeness", is_informative))

        color_stats = self._compute_color_statistics(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        entropy = self._compute_entropy(gray)
        edge_density = self._compute_edge_density(gray)

        if entropy < self.entropy_threshold:
            checks.append(("low_entropy", 1 - entropy / self.entropy_threshold))

        if edge_density < self.edge_density_threshold:
            checks.append(
                (
                    "low_edge_density",
                    1 - edge_density / self.edge_density_threshold,
                )
            )
        if color_stats["blue_ratio"] > self.blue_ratio_threshold:
            checks.append(("water_detected", color_stats["blue_ratio"]))

        if color_stats["sand_ratio"] > 0.5:
            checks.append(("desert_sand", color_stats["sand_ratio"]))

        return bool(is_informative)
