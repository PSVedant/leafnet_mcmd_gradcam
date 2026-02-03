import tensorflow as tf
import numpy as np
import os

from sklearn.metrics import confusion_matrix, classification_report

# ---------------- CONFIG ---------------- #
MODEL_PATH = "leafnet_custom_cnn.keras"
TEST_DIR = "data/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
OUTPUT_DIR = "outputs/evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD MODEL ---------------- #
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------- LOAD TEST DATA ---------------- #
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names
num_classes = len(class_names)

# Save class names
with open(f"{OUTPUT_DIR}/class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

# ---------------- PREDICTIONS ---------------- #
y_true = []
y_pred = []
y_prob = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))
    y_prob.extend(preds)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

# ---------------- SAVE OUTPUTS ---------------- #
np.save(f"{OUTPUT_DIR}/y_true.npy", y_true)
np.save(f"{OUTPUT_DIR}/y_pred.npy", y_pred)
np.save(f"{OUTPUT_DIR}/y_prob.npy", y_prob)

# ---------------- METRICS ---------------- #
cm = confusion_matrix(y_true, y_pred)
np.save(f"{OUTPUT_DIR}/confusion_matrix.npy", cm)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
    f.write(report)

print(" Evaluation complete")
print(f" Results saved to: {OUTPUT_DIR}")
