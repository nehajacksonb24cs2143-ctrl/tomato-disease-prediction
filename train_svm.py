import numpy as np
import os
import joblib

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------
# 1. Load training data
# --------------------------------

train_data = np.load("features/train_features.npz")

X_train = train_data["X"]
y_train = train_data["y"]

print("Training data:", X_train.shape)


# --------------------------------
# 2. Load validation data
# --------------------------------

val_data = np.load("features/validation_features.npz")

X_val = val_data["X"]
y_val = val_data["y"]

print("Validation data:", X_val.shape)


# --------------------------------
# 3. Load test data
# --------------------------------

test_data = np.load("features/test_features.npz")

X_test = test_data["X"]
y_test = test_data["y"]

print("Test data:", X_test.shape)


# --------------------------------
# 4. Create SVM model
# --------------------------------

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    class_weight="balanced"
)


# --------------------------------
# 5. Train SVM
# --------------------------------

print("\nTraining SVM...")

model.fit(X_train, y_train)

print("Training completed!")


# --------------------------------
# 6. Validation prediction
# --------------------------------

y_val_pred = model.predict(X_val)

val_accuracy = accuracy_score(y_val, y_val_pred)

print("\nValidation Accuracy:")
print(f"{val_accuracy * 100:.2f}%")


# --------------------------------
# 7. Test prediction
# --------------------------------

print("\nEvaluating on test data...")

y_test_pred = model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")


# --------------------------------
# 8. Classification report
# --------------------------------

print("\nTest Classification Report:")

print(
    classification_report(
        y_test,
        y_test_pred
    )
)


# --------------------------------
# 9. Save model
# --------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/svm_model.pkl"
)

print("\nSVM model saved successfully!")

print("models/svm_model.pkl")