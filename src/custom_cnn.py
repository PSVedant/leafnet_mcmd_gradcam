import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# ---------------- PATHS ---------------- #
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- CONFIG ---------------- #
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 40
LR = 1e-3

# ---------------- ATTENTION MODULE ---------------- #
class DualPathAttention(layers.Layer):
    def __init__(self, filters, reduction=8):
        super().__init__()
        self.filters = filters

        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()

        self.fc1 = layers.Dense(filters // reduction, activation="relu")
        self.fc2 = layers.Dense(filters, activation="sigmoid")

        self.spatial_conv = layers.Conv2D(1, 7, padding="same", activation="sigmoid")

    def call(self, inputs):
        avg = self.fc2(self.fc1(self.avg_pool(inputs)))
        max_ = self.fc2(self.fc1(self.max_pool(inputs)))

        channel_att = avg + max_
        channel_att = tf.reshape(channel_att, (-1, 1, 1, self.filters))
        x = inputs * channel_att

        avg_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_spatial = tf.reduce_max(x, axis=-1, keepdims=True)
        spatial_att = self.spatial_conv(tf.concat([avg_spatial, max_spatial], axis=-1))

        return x * spatial_att

# ---------------- RESIDUAL BLOCK ---------------- #
class ResidualBlock(layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = layers.Conv2D(filters, 3, padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, 3, padding="same", use_bias=False)
        self.bn2 = layers.BatchNormalization()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        return tf.nn.relu(x + inputs)

# ---------------- MODEL ---------------- #
def build_leafattentionnet(num_classes):
    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.Rescaling(1./255)(inputs)

    # Stem
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Block 1
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = ResidualBlock(64)(x)
    x = layers.MaxPooling2D()(x)
    x = DualPathAttention(64)(x)

    # Block 2
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = ResidualBlock(128)(x)
    x = layers.MaxPooling2D()(x)
    x = DualPathAttention(128)(x)

    # Block 3
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = ResidualBlock(256)(x)
    x = layers.MaxPooling2D()(x)
    x = DualPathAttention(256)(x)

    # Block 4 (deeper as requested)
    x = layers.Conv2D(512, 3, padding="same")(x)
    x = ResidualBlock(512)(x)
    x = layers.MaxPooling2D()(x)
    x = DualPathAttention(512)(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)

# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    print("="*60)
    print("Custom LeafAttentionNet (CPU Friendly)")
    print("="*60)

    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    val_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        shuffle=True
    )

    val_data = val_gen.flow_from_directory(
        VAL_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        shuffle=False
    )

    num_classes = train_data.num_classes

    model = build_leafattentionnet(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        tf.keras.callbacks.ModelCheckpoint("best_custom_cnn.keras", save_best_only=True)
    ]

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model.save("leafattentionnet_custom_cnn.keras")

    # Plot
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

    print("Training complete. Model saved.")
