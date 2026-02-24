import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# ---------------- CONFIG ---------------- #
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
OUTPUT_DIR = "outputs_novel"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4

# ---------------- NOVEL MICRO BLOCK ---------------- #
class FastMicroBlock(layers.Layer):
    def __init__(self, filters):
        super(FastMicroBlock, self).__init__()
        self.filters = filters

    def build(self, input_shape):
        self.dwconv = layers.DepthwiseConv2D(3, padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization()

        self.pwconv = layers.Conv2D(self.filters, 1, use_bias=False)
        self.bn2 = layers.BatchNormalization()

        self.se_dense1 = layers.Dense(self.filters // 4, activation="relu")
        self.se_dense2 = layers.Dense(self.filters, activation="sigmoid")

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

# ---------------- BUILD MODEL ---------------- #
def build_custom_net(num_classes):
    inputs = layers.Input(shape=(224, 224, 3))

    x = layers.Rescaling(1./255)(inputs)

    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = FastMicroBlock(64)(x)
    x = layers.MaxPooling2D()(x)

    x = FastMicroBlock(128)(x)
    x = layers.MaxPooling2D()(x)

    x = FastMicroBlock(256)(x)
    x = layers.MaxPooling2D()(x)

    x = FastMicroBlock(512)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)

# ---------------- LOAD DATA ---------------- #
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

num_classes = len(train_ds.class_names)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# ---------------- COMPILE ---------------- #
model = build_custom_net(num_classes)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------- CALLBACKS ---------------- #
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(OUTPUT_DIR, "best_model.keras"),
        save_best_only=True
    )
]

# ---------------- TRAIN ---------------- #
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

model.save(os.path.join(OUTPUT_DIR, "final_model.keras"))

# ---------------- PLOT ---------------- #
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"))
plt.close()

print("Training complete.")