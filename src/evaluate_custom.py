import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ---------------- CONFIG ---------------- #
MODEL_PATH = "outputs_novel/best_model.keras"
TEST_DIR = "data/test"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
OUTPUT_DIR = "outputs_novel"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- CUSTOM LAYER ---------------- #
# FIXED: Using utils instead of saving
@tf.keras.utils.register_keras_serializable()
class FastMicroBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(FastMicroBlock, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.dwconv = tf.keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pwconv = tf.keras.layers.Conv2D(self.filters, 1, use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.se_dense1 = tf.keras.layers.Dense(self.filters // 4, activation="relu")
        self.se_dense2 = tf.keras.layers.Dense(self.filters, activation="sigmoid")
        super(FastMicroBlock, self).build(input_shape)

    def call(self, inputs):
        x = self.dwconv(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.pwconv(x)
        x = self.bn2(x)
        se = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        se = self.se_dense1(se)
        se = self.se_dense2(se)
        x = x * se
        return tf.nn.relu(x)

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config

# ---------------- LOAD MODEL ---------------- #
print("Loading model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"FastMicroBlock": FastMicroBlock}
)

# ---------------- LOAD TEST DATA ---------------- #
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names
num_classes = len(class_names)

print("Test images:", len(test_ds.file_paths))
print("Number of classes:", num_classes)

test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# ---------------- PREDICTIONS ---------------- #
print("Running predictions...")
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# ---------------- ACCURACY ---------------- #
test_acc = np.mean(y_true == y_pred)
print("Test Accuracy:", test_acc)

# ---------------- CLASSIFICATION REPORT ---------------- #
report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
print(report)

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# ---------------- CONFUSION MATRIX ---------------- #
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap="Blues", xticklabels=False, yticklabels=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

print("Confusion matrix saved.")

# ---------------- ROC CURVE (Multiclass) ---------------- #
print("Generating ROC curve...")

y_true_bin = label_binarize(y_true, classes=range(num_classes))

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
mean_tpr = np.zeros_like(all_fpr)

for i in range(num_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= num_classes
macro_auc = auc(all_fpr, mean_tpr)

plt.figure(figsize=(8, 6))
plt.plot(all_fpr, mean_tpr, label=f"Macro-average ROC (AUC = {macro_auc:.4f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
plt.close()

print("ROC curve saved.")
print("Evaluation complete.")