import cv2
import numpy as np

from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops


def extract_features(image_path):

    # -----------------------------
    # 1. Read image
    # -----------------------------
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not read image")

    # Resize
    image = cv2.resize(image, (224, 224))

    # -----------------------------
    # 2. HSV FEATURES
    # -----------------------------
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    hsv_features = [
        np.mean(h),
        np.std(h),
        np.mean(s),
        np.std(s),
        np.mean(v),
        np.std(v)
    ]

    # -----------------------------
    # 3. GRAYSCALE
    # -----------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # -----------------------------
    # 4. LBP FEATURES
    # -----------------------------
    radius = 1
    points = 8 * radius

    lbp = local_binary_pattern(
        gray,
        points,
        radius,
        method="uniform"
    )

    # Histogram of LBP
    n_bins = points + 2

    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins)
    )

    # Normalize histogram
    lbp_hist = lbp_hist.astype(float)
    lbp_hist /= (lbp_hist.sum() + 1e-7)

    # -----------------------------
    # 5. GLCM FEATURES
    # -----------------------------
    # Reduce gray levels
    gray_small = (gray / 32).astype(np.uint8)

    glcm = graycomatrix(
        gray_small,
        distances=[1],
        angles=[0],
        levels=8,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, "contrast")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]

    glcm_features = [
        contrast,
        correlation,
        energy,
        homogeneity
    ]

    # -----------------------------
    # 6. COMBINE FEATURES
    # -----------------------------
    features = np.concatenate([
        hsv_features,
        lbp_hist,
        glcm_features
    ])

    return features


# ---------------------------------
# TEST WITH ONE IMAGE
# ---------------------------------

IMAGE_PATH = (
    "data/tomato/plantvillage/plantvillage/"
    "Tomato___Early_blight/"
)

import os

image_file = None

for file in os.listdir(IMAGE_PATH):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        image_file = os.path.join(IMAGE_PATH, file)
        break

features = extract_features(image_file)

print("Feature extraction successful!")
print("Number of features:", len(features))
print("\nFeature values:")
print(features)