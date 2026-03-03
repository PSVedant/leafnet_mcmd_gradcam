import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ---------------- CONFIG ---------------- #
MODEL_PATH = "outputs/best_fast_cnn.keras"
TEST_DIR = "data/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
OUTPUT_DIR = "outputs/evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- CUSTOM LAYER DEFINITION ---------------- #
# This must match the exact block used during your 99% training run


@tf.keras.saving.register_keras_serializable()
class FastMicroBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.dwconv = tf.keras.layers.DepthwiseConv2D(
            3, padding="same", use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pwconv = tf.keras.layers.Conv2D(filters, 1, use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.se_dense1 = tf.keras.layers.Dense(filters // 4, activation="relu")
        self.se_dense2 = tf.keras.layers.Dense(filters, activation="sigmoid")

    def call(self, inputs):
        x = self.dwconv(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.pwconv(x)
        x = self.bn2(x)
        se = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        se = self.se_dense1(se)
        se = self.se_dense2(se)
        return tf.nn.relu(x * se)

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config


# ---------------- LOAD MODEL & DATA ---------------- #
print("Loading model and test dataset...")
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={
                                   "FastMicroBlock": FastMicroBlock})

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names
y_true, y_pred = [], []

print("Running evaluation...")
for images, labels in test_ds:
    images = images / 255.0
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ---------------- GENERATE REPORTS ---------------- #
report = classification_report(
    y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(20, 15))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('LeafNet-MCMD Confusion Matrix')
plt.ylabel('Actual Disease')
plt.xlabel('Predicted Disease')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_heatmap.png")

print(f"\n[SUCCESS] Evaluation complete. Results saved to: {OUTPUT_DIR}")
