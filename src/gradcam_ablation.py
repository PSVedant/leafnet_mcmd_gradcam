import tensorflow as tf
import numpy as np
import cv2
import os
import h5py
import zipfile
import tempfile
from tensorflow.keras import layers, models

# ---------------- CONFIG ---------------- #
DATA_DIR = "data/test"
OUTPUT_DIR = "gradcam_ablation_outputs"
IMG_SIZE = (224, 224)
IMAGES_PER_CLASS = 5
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT.endswith("src"):
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

# ---------------- WEIGHT LOADER ---------------- #

def extract_h5(keras_path):
    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(keras_path, 'r') as z:
        z.extract("model.weights.h5", tmpdir)
    return os.path.join(tmpdir, "model.weights.h5")


def load_weights_by_layer_name(model, keras_path):
    h5_path = extract_h5(keras_path)
    with h5py.File(h5_path, 'r') as f:
        layer_group = f['layers']
        saved = {}
        for layer_name in layer_group.keys():
            lg = layer_group[layer_name]
            w_list = []
            def collect(name, obj, w=w_list):
                if isinstance(obj, h5py.Dataset):
                    w.append((name, np.array(obj)))
            lg.visititems(collect)
            if w_list:
                saved[layer_name] = [v for _, v in sorted(w_list)]

    for layer in model.layers:
        lname = layer.name
        if lname in saved and layer.weights:
            h5_weights = saved[lname]
            model_weights = layer.weights
            if len(h5_weights) == len(model_weights):
                for w, val in zip(model_weights, h5_weights):
                    if w.shape == val.shape:
                        w.assign(val)
    print(f"  Weights loaded.")


# ---------------- BUILD FUNCTIONS ---------------- #

def build_no_attention(n):
    class FastMicroBlock(layers.Layer):
        def __init__(self, filters, **kwargs):
            super().__init__(**kwargs)
            self.filters = filters
        def build(self, input_shape):
            self.dw  = layers.DepthwiseConv2D(3, padding="same", use_bias=False)
            self.bn1 = layers.BatchNormalization()
            self.pw  = layers.Conv2D(self.filters, 1, use_bias=False)
            self.bn2 = layers.BatchNormalization()
            super().build(input_shape)
        def call(self, x):
            x = self.dw(x); x = self.bn1(x); x = tf.nn.relu(x)
            x = self.pw(x); x = self.bn2(x)
            return tf.nn.relu(x)

    i = layers.Input((224, 224, 3))
    x = layers.Rescaling(1./255)(i)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = FastMicroBlock(64)(x);  x = layers.MaxPooling2D()(x)
    x = FastMicroBlock(128)(x); x = layers.MaxPooling2D()(x)
    x = FastMicroBlock(256)(x); x = layers.MaxPooling2D()(x)
    x = FastMicroBlock(512)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    o = layers.Dense(n, activation="softmax")(x)
    return models.Model(i, o)


def build_no_depthwise(n):
    class FastMicroBlock(layers.Layer):
        def __init__(self, filters, **kwargs):
            super().__init__(**kwargs)
            self.filters = filters
        def build(self, input_shape):
            self.conv = layers.Conv2D(self.filters, 3, padding="same", use_bias=True)
            self.bn   = layers.BatchNormalization()
            super().build(input_shape)
        def call(self, x):
            x = self.conv(x); x = self.bn(x)
            return tf.nn.relu(x)

    i = layers.Input((224, 224, 3))
    x = layers.Rescaling(1./255)(i)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = FastMicroBlock(64)(x);  x = layers.MaxPooling2D()(x)
    x = FastMicroBlock(128)(x); x = layers.MaxPooling2D()(x)
    x = FastMicroBlock(256)(x); x = layers.MaxPooling2D()(x)
    x = FastMicroBlock(512)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    o = layers.Dense(n, activation="softmax")(x)
    return models.Model(i, o)


def build_shallow(n):
    class FastMicroBlock(layers.Layer):
        def __init__(self, filters, **kwargs):
            super().__init__(**kwargs)
            self.filters = filters
        def build(self, input_shape):
            self.dw = layers.DepthwiseConv2D(3, padding="same", use_bias=False)
            self.bn = layers.BatchNormalization()
            super().build(input_shape)
        def call(self, x):
            x = self.dw(x); x = self.bn(x)
            return tf.nn.relu(x)

    i = layers.Input((224, 224, 3))
    x = layers.Rescaling(1./255)(i)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = FastMicroBlock(64)(x);  x = layers.MaxPooling2D()(x)
    x = FastMicroBlock(128)(x)
    x = layers.GlobalAveragePooling2D()(x)
    o = layers.Dense(n, activation="softmax")(x)
    return models.Model(i, o)


# ---------------- GRADCAM ---------------- #

def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        loss = preds[:, pred_index]
    grads = tape.gradient(loss, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros((7, 7))
    heatmap = heatmap / max_val
    return heatmap.numpy()


def save_gradcam(img_path, heatmap, save_path, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    cv2.imwrite(save_path, superimposed)


# ---------------- MAIN ---------------- #

num_classes = len([
    d for d in os.listdir(os.path.join(PROJECT_ROOT, DATA_DIR))
    if os.path.isdir(os.path.join(PROJECT_ROOT, DATA_DIR, d))
])
print(f"Detected {num_classes} classes.")

ABLATION_MODELS = {
    "no_attention": (
        os.path.join(PROJECT_ROOT, "ablation/no_attention/model.keras"),
        build_no_attention
    ),
    "no_depthwise": (
        os.path.join(PROJECT_ROOT, "ablation/no_depthwise/model.keras"),
        build_no_depthwise
    ),
    "shallow": (
        os.path.join(PROJECT_ROOT, "ablation/shallow/model.keras"),
        build_shallow
    ),
}

test_dir = os.path.join(PROJECT_ROOT, DATA_DIR)

for model_name, (model_path, build_fn) in ABLATION_MODELS.items():
    print(f"\n{'='*40}")
    print(f"GradCAM: {model_name}")
    print(f"{'='*40}")

    tf.keras.backend.clear_session()
    model = build_fn(num_classes)
    load_weights_by_layer_name(model, model_path)

    last_conv = get_last_conv_layer(model)
    print(f"  Last conv layer: {last_conv}")

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv).output, model.output]
    )

    processed = 0
    errors = 0

    for cls in sorted(os.listdir(test_dir)):
        cls_path = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        img_files = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ][:IMAGES_PER_CLASS]

        if not img_files:
            continue

        save_cls_dir = os.path.join(OUTPUT_DIR, model_name, cls)
        os.makedirs(save_cls_dir, exist_ok=True)

        for img_name in img_files:
            img_path = os.path.join(cls_path, img_name)
            try:
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)

                preds = model.predict(img_array, verbose=0)
                pred_class = np.argmax(preds[0])

                heatmap = make_gradcam_heatmap(img_array, grad_model, pred_class)
                save_path = os.path.join(save_cls_dir, img_name)
                save_gradcam(img_path, heatmap, save_path)
                processed += 1
            except Exception as e:
                print(f"  ✗ {cls}/{img_name}: {e}")
                errors += 1

        print(f"  ✓ {cls} ({len(img_files)} images)")

    print(f"\n  Done {model_name}: {processed} saved, {errors} errors")

print("\n✅ Ablation GradCAM complete.")
print(f"Results in: {OUTPUT_DIR}/")