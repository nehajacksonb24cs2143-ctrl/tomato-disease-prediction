from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Dataset location
DATASET_PATH = Path("data/tomato/plantvillage/plantvillage")

# Store image paths and labels
data = []

# Read each disease folder
for class_folder in sorted(DATASET_PATH.iterdir()):

    if not class_folder.is_dir():
        continue

    class_name = class_folder.name

    for image_file in class_folder.iterdir():

        if image_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            data.append({
                "image_path": str(image_file),
                "label": class_name
            })

# Create DataFrame
df = pd.DataFrame(data)

print("Total images:", len(df))

# First split: 70% train, 30% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

# Second split: 15% validation, 15% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

# Create splits folder
Path("splits").mkdir(exist_ok=True)

# Save CSV files
train_df.to_csv("splits/train.csv", index=False)
val_df.to_csv("splits/validation.csv", index=False)
test_df.to_csv("splits/test.csv", index=False)

# Display results
print("\nDataset split completed!")

print("\nTraining images:", len(train_df))
print("Validation images:", len(val_df))
print("Testing images:", len(test_df))

print("\nFiles created:")
print("splits/train.csv")
print("splits/validation.csv")
print("splits/test.csv")