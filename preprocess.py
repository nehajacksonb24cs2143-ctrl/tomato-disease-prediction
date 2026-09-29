import cv2
import matplotlib.pyplot as plt

# Change this to one image from your dataset
IMAGE_PATH = "data/tomato/plantvillage/plantvillage/Tomato___Early_blight/"

# Get the first image from the folder
import os

image_file = None

for file in os.listdir(IMAGE_PATH):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        image_file = os.path.join(IMAGE_PATH, file)
        break

# Read image
image = cv2.imread(image_file)

# Convert BGR to RGB for displaying
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize image
resized = cv2.resize(image_rgb, (224, 224))

# Convert RGB to HSV
hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)

# Create a simple green-leaf mask
lower_green = (25, 30, 30)
upper_green = (100, 255, 255)

mask = cv2.inRange(hsv, lower_green, upper_green)

# Apply mask
masked_image = cv2.bitwise_and(resized, resized, mask=mask)

# Display results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("Leaf Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(masked_image)
plt.title("Masked Leaf")
plt.axis("off")

plt.tight_layout()
plt.show()