from pathlib import Path

DATASET_PATH = Path("data/tomato/plantvillage/plantvillage")

classes = sorted([
    folder for folder in DATASET_PATH.iterdir()
    if folder.is_dir()
])

print("Number of classes:", len(classes))
print("\nImages in each class:\n")

total_images = 0

for folder in classes:

    images = [
        file for file in folder.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    print(f"{folder.name}: {len(images)} images")

    total_images += len(images)

print("\nTotal images:", total_images)