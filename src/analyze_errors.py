import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# ---------------- CONFIG ---------------- #
MODEL_PATH = "outputs_novel/best_model.keras"
TEST_DIR = "data/test"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
OUTPUT_DIR = "outputs_novel/error_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

# ---------------- CUSTOM LAYER DEFINITION ---------------- #
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

# ---------------- LOAD MODEL & DATA ---------------- #
print("Loading model for Error Analysis...")
model = tf.keras.models.load_model(
    MODEL_PATH, 
    custom_objects={"FastMicroBlock": FastMicroBlock}
)

# shuffle=False is CRUCIAL here so the file paths line up with the predictions
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False 
)

class_names = test_ds.class_names
file_paths = test_ds.file_paths

# ---------------- RUN PREDICTIONS ---------------- #
print("Hunting for misclassified images...")
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# Find where the model guessed wrong
misclassified_indices = np.where(y_true != y_pred)[0]
print(f"Found {len(misclassified_indices)} misclassified images out of {len(y_true)}.")

# ---------------- GENERATE TABLE AND IMAGES ---------------- #
table_data = []

for idx in misclassified_indices:
    true_id = y_true[idx]
    pred_id = y_pred[idx]
    
    true_class = class_names[true_id]
    pred_class = class_names[pred_id]
    confidence = y_pred_probs[idx][pred_id]
    img_path = file_paths[idx]
    img_name = os.path.basename(img_path)
    
    # 1. Add to Table Data
    table_data.append({
        "True Class (Actual)": true_class,
        "Misclassified As (Predicted)": pred_class,
        "Confidence": f"{confidence*100:.2f}%",
        "Image Filename": img_name
    })
    
    # 2. Save Visual Sample for the Professor
    img = tf.keras.preprocessing.image.load_img(img_path)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"TRUE: {true_class}\nPRED: {pred_class} ({confidence*100:.1f}%)", 
              color='red' if true_class != pred_class else 'green', fontsize=10)
    plt.axis('off')
    
    # Save the image using the true and predicted names for easy sorting
    safe_true = true_class.replace("___", "_").replace(" ", "")
    safe_pred = pred_class.replace("___", "_").replace(" ", "")
    save_name = f"True_{safe_true}_Pred_{safe_pred}_{img_name}"
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "images", save_name))
    plt.close()

# ---------------- SAVE CSV TABLE ---------------- #
df = pd.DataFrame(table_data)
csv_path = os.path.join(OUTPUT_DIR, "misclassifications_table.csv")
df.to_csv(csv_path, index=False)

print(f"\n[SUCCESS] Error Analysis Complete!")
print(f"1. A table of all errors was saved to: {csv_path}")
print(f"2. Visual images of the errors were saved to: {OUTPUT_DIR}/images/")