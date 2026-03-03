import tensorflow as tf
import numpy as np
import cv2
import os

# ---------------- CONFIG ---------------- #
MODEL_PATH = "outputs_novel/best_model.keras"
TEST_DIR = "data/test"
IMG_SIZE = (224, 224)
OUTPUT_DIR = "outputs_novel/gradcam"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# ---------------- LOAD MODEL ---------------- #
print("Loading model...")
model = tf.keras.models.load_model(
    MODEL_PATH, 
    custom_objects={"FastMicroBlock": FastMicroBlock}
)
print("Model loaded successfully.")

# Dynamically find the last spatial layer (either Conv2D or FastMicroBlock)
last_conv_layer_name = None
for layer in reversed(model.layers):
    # FIXED: Removed the output_shape check that crashed on the Dense layer
    if isinstance(layer, FastMicroBlock) or isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer_name = layer.name
        break

print(f"Targeting layer for Grad-CAM: {last_conv_layer_name}")

# Create Grad-CAM model
grad_model = tf.keras.models.Model(
    [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
)

# ---------------- GRAD-CAM LOGIC ---------------- #
def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Calculate gradients of the predicted class with respect to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel in the feature map array by "how important" this channel is
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)
    cv2.imwrite(cam_path, superimposed_img)

# ---------------- PROCESS IMAGES ---------------- #
print("\nGenerating Grad-CAM visualizations across all classes...")

processed_count = 0
error_count = 0

for cls in sorted(os.listdir(TEST_DIR)):
    cls_path = os.path.join(TEST_DIR, cls)
    if not os.path.isdir(cls_path):
        continue
    
    img_files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG'))]
    if not img_files:
        continue
        
    img_name = img_files[0]
    img_path = os.path.join(cls_path, img_name)
    
    # Prepare image exactly as the model expects
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    
    # Get prediction
    preds = model.predict(img_array, verbose=0)
    pred_class = np.argmax(preds[0])
    confidence = preds[0][pred_class]
    
    try:
        heatmap = make_gradcam_heatmap(img_array, grad_model, pred_class)
        save_path = os.path.join(OUTPUT_DIR, f"{cls}_gradcam.jpg")
        save_and_display_gradcam(img_path, heatmap, save_path)
        
        print(f"✓ {cls} -> Saved (Confidence: {confidence:.2f})")
        processed_count += 1
    
    except Exception as e:
        print(f"✗ Failed on {cls}: {str(e)}")
        error_count += 1

print(f"\n{'='*40}")
print(f"Grad-CAM Complete!")
print(f"Successfully processed: {processed_count}")
print(f"Errors: {error_count}")
print(f"Results saved to: {OUTPUT_DIR}/")
print(f"{'='*40}")