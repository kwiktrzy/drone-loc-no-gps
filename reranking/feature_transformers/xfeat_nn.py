import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from reranking.feature_transformers.abstract.modules.xfeat import XFeat


class XFeatLightGlueFT:
    def __init__(self, ratio_thresh=0.8, ransac_thresh=5.0):
        self.xfeat_model = XFeat()
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.ratio_thresh = ratio_thresh
        self.ransac_thresh = ransac_thresh

    def _prepare_image(self, img_path):
        im = Image.open(img_path).convert("RGB")
        img_arr = np.array(im)
        img_tensor = self.xfeat_model.parse_input(img_arr)
        return im, img_arr, img_tensor

    def get_xfeat_lightglue_match_score(
        self, img_query_path, img_candidate_path, visualize=False, verbose=False
    ):
        im1, img1_arr, img1_tensor = self._prepare_image(img_query_path)
        im2, img2_arr, img2_tensor = self._prepare_image(img_candidate_path)

        output0 = self.xfeat_model.detectAndCompute(img1_tensor)[0]
        output1 = self.xfeat_model.detectAndCompute(img2_tensor)[0]

        output0["image_size"] = im1.size
        output1["image_size"] = im2.size

        mkpts_0, mkpts_1, _ = self.xfeat_model.match_lighterglue(output0, output1)

        # --- RANSAC ---
        score = 0
        inliers_mask = None

        if len(mkpts_0) >= 4:
            _, mask = cv2.findHomography(
                mkpts_0, mkpts_1, cv2.RANSAC, self.ransac_thresh
            )
            if mask is not None:
                inliers_mask = mask.ravel()
                score = int(inliers_mask.sum())
        elif verbose:
            print("Not enough points for RANSAC")

        if verbose:
            print(f"Matches (LightGlue): {len(mkpts_0)}  |  Inliers (RANSAC): {score}")

        if visualize and score > 0:
            self._visualize(
                img1_arr,
                img2_arr,  # PIL / np.array RGB
                mkpts_0,
                mkpts_1,
                inliers_mask,
                title=f"XFeat + LightGlue — inliers: {score}",
            )

        return score

    def get_xfeat_bf_match_score(
        self, img_query_path, img_candidate_path, visualize=False, verbose=False
    ):
        im1, img1_arr, img1_tensor = self._prepare_image(img_query_path)
        im2, img2_arr, img2_tensor = self._prepare_image(img_candidate_path)

        output1 = self.xfeat_model.detectAndCompute(img1_tensor)[0]
        output2 = self.xfeat_model.detectAndCompute(img2_tensor)[0]

        desc1_np = output1["descriptors"].cpu().numpy().astype(np.float32)
        desc2_np = output2["descriptors"].cpu().numpy().astype(np.float32)

        kp1_np = output1["keypoints"].cpu().numpy()
        kp2_np = output2["keypoints"].cpu().numpy()

        if (
            desc1_np is None
            or desc2_np is None
            or len(desc1_np) == 0
            or len(desc2_np) == 0
        ):
            return 0

        matches_knn = self.bf.knnMatch(desc1_np, desc2_np, k=2)
        if not matches_knn:
            return 0

        good = []
        for m_n in matches_knn:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < self.ratio_thresh * n.distance:
                good.append(m)

        # --- RANSAC ---
        score = 0
        inliers_mask = None

        if len(good) >= 4:
            src = np.float32([kp1_np[m.queryIdx] for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp2_np[m.trainIdx] for m in good]).reshape(-1, 1, 2)

            _, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_thresh)
            if mask is not None:
                inliers_mask = mask.ravel()
                score = int(inliers_mask.sum())
        elif verbose:
            print("Not enough good points for RANSAC")

        if verbose:
            print(f"Good matches (ratio): {len(good)}  |  Inliers: {score}")

        if visualize and score > 0:
            mkpts_0 = np.float32([kp1_np[m.queryIdx] for m in good])
            mkpts_1 = np.float32([kp2_np[m.trainIdx] for m in good])
            self._visualize(
                img1_arr,
                img2_arr,
                mkpts_0,
                mkpts_1,
                inliers_mask,
                title=f"XFeat + BF — inliers: {score}",
            )

        return score

    def _visualize(self, img1_rgb, img2_rgb, mkpts_0, mkpts_1, inliers_mask, title=""):

        kp1_cv = [cv2.KeyPoint(float(p[0]), float(p[1]), size=5) for p in mkpts_0]
        kp2_cv = [cv2.KeyPoint(float(p[0]), float(p[1]), size=5) for p in mkpts_1]
        dmatches = [cv2.DMatch(i, i, 0) for i in range(len(mkpts_0))]

        mask_list = (
            inliers_mask.astype(int).tolist() if inliers_mask is not None else None
        )

        vis = cv2.drawMatches(
            img1_rgb,
            kp1_cv,
            img2_rgb,
            kp2_cv,
            dmatches,
            None,
            matchColor=(0, 255, 0),
            singlePointColor=None,
            matchesMask=mask_list,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        plt.figure(figsize=(15, 10))
        plt.imshow(vis)
        plt.title(title)
        plt.axis("off")
        plt.show()
