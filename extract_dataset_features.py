import pandas as pd
import numpy as np
from features import extract_features
from tqdm import tqdm
from pathlib import Path


def process_dataset(csv_file, output_file):

    # Read CSV
    df = pd.read_csv(csv_file)

    feature_list = []
    labels = []

    print(f"\nProcessing: {csv_file}")
    print("Number of images:", len(df))

    # Process every image
    for image_path, label in tqdm(
        zip(df["image_path"], df["label"]),
        total=len(df)
    ):

        try:
            features = extract_features(image_path)

            feature_list.append(features)
            labels.append(label)

        except Exception as e:
            print(f"\nError processing: {image_path}")
            print(e)

    # Convert to NumPy array
    X = np.array(feature_list)
    y = np.array(labels)

    # Create output folder
    Path("features").mkdir(exist_ok=True)

    # Save features
    np.savez(
        output_file,
        X=X,
        y=y
    )

    print("\nCompleted!")
    print("Feature shape:", X.shape)
    print("Label shape:", y.shape)


# Process training data
process_dataset(
    "splits/train.csv",
    "features/train_features.npz"
)

# Process validation data
process_dataset(
    "splits/validation.csv",
    "features/validation_features.npz"
)

# Process testing data
process_dataset(
    "splits/test.csv",
    "features/test_features.npz"
)