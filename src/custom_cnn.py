import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-4

class FastMicroBlock(layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.dwconv = layers.DepthwiseConv2D(3, padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.pwconv = layers.Conv2D(filters, 1, use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.se_dense1 = layers.Dense(filters // 4, activation="relu")
        self.se_dense2 = layers.Dense(filters, activation="sigmoid")

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

def build_fast_cpu_net(num_classes):
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
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)

if __name__ == "__main__":
    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
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

    model = build_fast_cpu_net(train_data.num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(OUTPUT_DIR, "best_fast_cnn.keras"), save_best_only=True)
    ]

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model.save(os.path.join(OUTPUT_DIR, "final_fast_cnn.keras"))

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    
    plt.subplot(1,2,2)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"))
    plt.close()

    print("Training complete. Model saved.")