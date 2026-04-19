import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import h5py
import zipfile
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(PROJECT_ROOT, "data/test")
OUT = os.path.join(PROJECT_ROOT, "ablation_results")
os.makedirs(os.path.join(OUT, "cm"), exist_ok=True)
os.makedirs(os.path.join(OUT, "roc"), exist_ok=True)
os.makedirs(os.path.join(OUT, "reports"), exist_ok=True)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


# -------- CLASSES -------- #

class FastMicroBlockNoAttention(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        self.dw  = layers.DepthwiseConv2D(3, padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.pw  = layers.Conv2D(self.filters, 1, use_bias=False)
        self.bn2 = layers.BatchNormalization()
        super().build(input_shape)

    def call(self, x):
        x = self.dw(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.pw(x)
        x = self.bn2(x)
        return tf.nn.relu(x)


class FastMicroBlockShallow(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        self.dw = layers.DepthwiseConv2D(3, padding="same", use_bias=False)
        self.bn = layers.BatchNormalization()
        super().build(input_shape)

    def call(self, x):
        x = self.dw(x)
        x = self.bn(x)
        return tf.nn.relu(x)


class FastMicroBlockFull(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        self.dwconv    = layers.DepthwiseConv2D(3, padding="same", use_bias=False)
        self.bn1       = layers.BatchNormalization()
        self.pwconv    = layers.Conv2D(self.filters, 1, use_bias=False)
        self.bn2       = layers.BatchNormalization()
        self.se_dense1 = layers.Dense(self.filters // 4, activation="relu")
        self.se_dense2 = layers.Dense(self.filters, activation="sigmoid")
        super().build(input_shape)

    def call(self, inputs):
        x  = self.dwconv(inputs)
        x  = self.bn1(x)
        x  = tf.nn.relu(x)
        x  = self.pwconv(x)
        x  = self.bn2(x)
        se = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        se = self.se_dense1(se)
        se = self.se_dense2(se)
        x  = x * se
        return tf.nn.relu(x)


# -------- BUILD FUNCTIONS -------- #

def build_no_attention(n):
    i = layers.Input((224, 224, 3))
    x = layers.Rescaling(1./255)(i)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = FastMicroBlockNoAttention(64)(x);  x = layers.MaxPooling2D()(x)
    x = FastMicroBlockNoAttention(128)(x); x = layers.MaxPooling2D()(x)
    x = FastMicroBlockNoAttention(256)(x); x = layers.MaxPooling2D()(x)
    x = FastMicroBlockNoAttention(512)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    o = layers.Dense(n, activation="softmax")(x)
    return models.Model(i, o)


def build_shallow(n):
    i = layers.Input((224, 224, 3))
    x = layers.Rescaling(1./255)(i)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = FastMicroBlockShallow(64)(x);  x = layers.MaxPooling2D()(x)
    x = FastMicroBlockShallow(128)(x)
    x = layers.GlobalAveragePooling2D()(x)
    o = layers.Dense(n, activation="softmax")(x)
    return models.Model(i, o)


# no_depthwise: saved only 28 weights, so it was trained WITHOUT depthwise
# meaning FastMicroBlock in that script used only pw+bn (no dw at all)
def build_no_depthwise(n):
    i = layers.Input((224, 224, 3))
    x = layers.Rescaling(1./255)(i)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = FastMicroBlockNoAttention(64)(x);  x = layers.MaxPooling2D()(x)
    x = FastMicroBlockNoAttention(128)(x); x = layers.MaxPooling2D()(x)
    x = FastMicroBlockNoAttention(256)(x); x = layers.MaxPooling2D()(x)
    x = FastMicroBlockNoAttention(512)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    o = layers.Dense(n, activation="softmax")(x)
    return models.Model(i, o)


# -------- WEIGHT LOADER (by layer name matching H5 structure) -------- #

def extract_h5(keras_path):
    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(keras_path, 'r') as z:
        z.extract("model.weights.h5", tmpdir)
    return os.path.join(tmpdir, "model.weights.h5")


def load_weights_ordered(model, keras_path):
    h5_path = extract_h5(keras_path)

    with h5py.File(h5_path, 'r') as f:
        layer_group = f['layers']

        # Collect weights per saved layer name in H5 order
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

        # Match model layers to saved layers by name
        unmatched = []
        for layer in model.layers:
            lname = layer.name
            if lname in saved and layer.weights:
                h5_weights = saved[lname]
                model_weights = layer.weights
                if len(h5_weights) == len(model_weights):
                    for w, val in zip(model_weights, h5_weights):
                        if w.shape == val.shape:
                            w.assign(val)
                        else:
                            raise ValueError(
                                f"Shape mismatch in layer '{lname}': "
                                f"model={w.shape} vs h5={val.shape}"
                            )
                else:
                    unmatched.append(
                        f"  Count mismatch in '{lname}': "
                        f"model={len(model_weights)} h5={len(h5_weights)}"
                    )
            elif layer.weights:
                unmatched.append(f"  Layer '{lname}' not found in H5")

        if unmatched:
            print("  WARNING — unmatched layers:")
            for m in unmatched:
                print(m)
        else:
            print("  All weights loaded successfully.")


# -------- DATA -------- #
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)
class_names = test_ds.class_names
num_classes  = len(class_names)

MODELS = {
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

# -------- LOOP -------- #
for name, (path, build_fn) in MODELS.items():
    print(f"\n=== Evaluating {name} ===")

    tf.keras.backend.clear_session()
    model = build_fn(num_classes)

    try:
        load_weights_ordered(model, path)
    except Exception as e:
        print(f"  Failed: {e}. Skipping.")
        continue

    y_true, y_pred, y_prob = [], [], []
    for images, labels in test_ds:
        images = images / 255.0
        preds  = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
        y_prob.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # -------- REPORT -------- #
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    with open(os.path.join(OUT, "reports", f"{name}.txt"), "w") as f:
        f.write(report)
    print(report)

    # -------- CONFUSION MATRIX -------- #
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues", annot=False)
    plt.title(f"{name} - Confusion Matrix")
    plt.savefig(os.path.join(OUT, "cm", f"{name}.png"), bbox_inches="tight")
    plt.close()

    # -------- ROC -------- #
    y_bin = label_binarize(y_true, classes=range(num_classes))
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, alpha=0.7, label=f"{class_names[i]} ({roc_auc:.2f})")
    plt.title(f"{name} - ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(fontsize=5, loc="lower right")
    plt.savefig(os.path.join(OUT, "roc", f"{name}.png"), bbox_inches="tight")
    plt.close()

print("\n✅ Ablation evaluation completed successfully.")
