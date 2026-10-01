import cv2
import numpy as np
import joblib

from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops


def extract_features(image_path):

    # -----------------------------
    # 1. Read image
    # -----------------------------
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not read image")

    # IMPORTANT:
    # Same size used during training
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

    n_bins = points + 2

    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins)
    )

    lbp_hist = lbp_hist.astype(float)

    lbp_hist /= (lbp_hist.sum() + 1e-7)

    # -----------------------------
    # 5. GLCM FEATURES
    # -----------------------------
    gray_small = (gray / 32).astype(np.uint8)

    glcm = graycomatrix(
        gray_small,
        distances=[1],
        angles=[0],
        levels=8,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(
        glcm, "contrast"
    )[0, 0]

    correlation = graycoprops(
        glcm, "correlation"
    )[0, 0]

    energy = graycoprops(
        glcm, "energy"
    )[0, 0]

    homogeneity = graycoprops(
        glcm, "homogeneity"
    )[0, 0]

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
# Load trained Random Forest
# ---------------------------------

model = joblib.load(
    "models/random_forest_model.pkl"
)


# ---------------------------------
# Get image path
# ---------------------------------

image_path = input(
    "\nEnter the path of the tomato leaf image: "
)


# ---------------------------------
# Extract features
# ---------------------------------

features = extract_features(image_path)

features = features.reshape(1, -1)

print("\nNumber of features:", features.shape[1])


# ---------------------------------
# Prediction
# ---------------------------------

probabilities = model.predict_proba(features)[0]

class_names = model.classes_

top_indices = np.argsort(
    probabilities
)[::-1][:3]


# ---------------------------------
# Display result
# ---------------------------------

print("\n================================")
print("       CROP HEALTH RESULT")
print("================================")

print("\nPredicted class:")
print(class_names[top_indices[0]])

print(
    f"\nConfidence: "
    f"{probabilities[top_indices[0]] * 100:.2f}%"
)

print("\nTop 3 predictions:")

for rank, index in enumerate(
    top_indices,
    start=1
):

    print(
        f"{rank}. {class_names[index]} "
        f"- {probabilities[index] * 100:.2f}%"
    )

print("\n================================")