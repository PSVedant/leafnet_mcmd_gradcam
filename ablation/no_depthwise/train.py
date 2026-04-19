import tensorflow as tf
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from sklearn.preprocessing import label_binarize

# Add project root to path (one level up from src/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your custom layer from custom_cnn.py (same directory)
try:
    from src.custom_cnn import FastMicroBlock
except ImportError:
    raise ImportError("FastMicroBlock not found in src/custom_cnn.py")

# Paths relative to project root (where src/ is located)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = {
    "no_attention": os.path.join(PROJECT_ROOT, "ablation/no_attention/model.keras"),
    "no_depthwise": os.path.join(PROJECT_ROOT, "ablation/no_depthwise/model.keras"),
    "shallow": os.path.join(PROJECT_ROOT, "ablation/shallow/model.keras")
}

TEST_DIR = os.path.join(PROJECT_ROOT, "data/test")
OUT = os.path.join(PROJECT_ROOT, "ablation_results")

os.makedirs(os.path.join(OUT, "cm"), exist_ok=True)
os.makedirs(os.path.join(OUT, "roc"), exist_ok=True)
os.makedirs(os.path.join(OUT, "reports"), exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load test dataset (only once)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names
num_classes = len(class_names)

# Register custom layer
custom_objects = {'FastMicroBlock': FastMicroBlock}

for name, path in MODELS.items():
    print(f"\n=== Evaluating {name} ===")
    model = tf.keras.models.load_model(path, custom_objects=custom_objects)

    y_true, y_pred, y_prob = [], [], []

    for images, labels in test_ds:
        images = images / 255.0
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
        y_prob.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0)
    with open(os.path.join(OUT, "reports", f"{name}.txt"), "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues", annot=False)
    plt.title(f"{name} - Confusion Matrix")
    plt.savefig(os.path.join(OUT, "cm", f"{name}.png"), bbox_inches="tight")
    plt.close()

    # ROC curves (one per class)
    y_bin = label_binarize(y_true, classes=range(num_classes))
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, lw=1, alpha=0.7)
    plt.title(f"{name} - ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(os.path.join(OUT, "roc", f"{name}.png"), bbox_inches="tight")
    plt.close()

print("\n✅ Ablation evaluation completed successfully.")
