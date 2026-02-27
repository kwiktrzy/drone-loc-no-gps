import cv2
import numpy as np
import matplotlib.pyplot as plt


class SitfFT:
    def __init__(self, nfeatures=5000, ratio_thresh=0.8, ransac_thresh=5.0):
        self.sift = cv2.SIFT_create(nfeatures=nfeatures)
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.ratio_thresh = ratio_thresh
        self.ransac_thresh = ransac_thresh

    def get_sift_match_score(
        self, img_query_path, img_candidate_path, visualize=False, verbose=False
    ):

        img1 = cv2.imread(img_query_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_candidate_path, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            if verbose:
                print("Error: Enable to load picture.")
            return 0

        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)

        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            return 0

        matches_knn = self.bf.knnMatch(des1, des2, k=2)
        if matches_knn is None or len(matches_knn) == 0:
            return 0

        good_matches = []
        for m_n in matches_knn:
            if m_n is None or len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < self.ratio_thresh * n.distance:
                good_matches.append(m)

        score = 0
        inliers_mask = None

        if len(good_matches) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
                -1, 1, 2
            )
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
                -1, 1, 2
            )

            M, mask = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh
            )

            if mask is not None:
                inliers_mask = mask.ravel().tolist()
                score = int(np.sum(mask))
        else:
            if verbose:
                print("Not enough points for RANSAC")

        if verbose:
            print(f"Number of matches (before RANSAC): {len(good_matches)}")
            print(f"Number of inliers (after RANSAC): {score}")

        if visualize and score > 0:
            draw_params = dict(
                matchColor=(0, 255, 0),
                singlePointColor=None,
                matchesMask=inliers_mask,
                flags=2,
            )
            img3 = cv2.drawMatches(
                img1, kp1, img2, kp2, good_matches, None, **draw_params
            )

            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(15, 10))
            plt.imshow(img3)
            plt.title(f"Inliers: {score}")
            plt.axis("off")
            plt.show()

        return score
