import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10

# -------- LOAD DATA -------- #
train = tf.keras.preprocessing.image_dataset_from_directory(
    "data/train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val = tf.keras.preprocessing.image_dataset_from_directory(
    "data/val",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# SAVE CLASS NAMES BEFORE PREFETCH
class_names = train.class_names
num_classes = len(class_names)

# PREFETCH AFTER
train = train.prefetch(tf.data.AUTOTUNE)
val = val.prefetch(tf.data.AUTOTUNE)

# -------- MODEL -------- #
base_model = EfficientNetB0(
    include_top=False,
    input_shape=(*IMAGE_SIZE, 3),
    weights="imagenet"
)
base_model.trainable = False   # freeze base

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -------- TRAIN -------- #
history = model.fit(
    train,
    validation_data=val,
    epochs=EPOCHS
)

# -------- SAVE MODEL -------- #
model.save("leafnet_baseline.h5")

print("Training complete. Model saved.")
