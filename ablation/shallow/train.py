import tensorflow as tf
from tensorflow.keras import layers, models
import os

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
OUT = "ablation/shallow"
os.makedirs(OUT, exist_ok=True)

IMAGE_SIZE=(224,224)
BATCH_SIZE=32
EPOCHS=15
LR=1e-4

class FastMicroBlock(layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.filters = filters

    def build(self, input_shape):
        self.dw = layers.DepthwiseConv2D(3,padding="same",use_bias=False)
        self.bn = layers.BatchNormalization()

    def call(self,x):
        x=self.dw(x)
        x=self.bn(x)
        return tf.nn.relu(x)

def build(n):
    i=layers.Input((224,224,3))
    x=layers.Rescaling(1./255)(i)

    x=layers.Conv2D(32,3,strides=2,padding="same")(x)
    x=FastMicroBlock(64)(x)
    x=layers.MaxPooling2D()(x)

    x=FastMicroBlock(128)(x)

    x=layers.GlobalAveragePooling2D()(x)
    o=layers.Dense(n,activation="softmax")(x)

    return models.Model(i,o)

train=tf.keras.preprocessing.image_dataset_from_directory(TRAIN_DIR,image_size=IMAGE_SIZE,batch_size=BATCH_SIZE)
val=tf.keras.preprocessing.image_dataset_from_directory(VAL_DIR,image_size=IMAGE_SIZE,batch_size=BATCH_SIZE)

model=build(len(train.class_names))
model.compile(optimizer=tf.keras.optimizers.Adam(LR),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

cb=[
    tf.keras.callbacks.EarlyStopping(patience=8,restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(f"{OUT}/model.keras",save_best_only=True)
]

model.fit(train,validation_data=val,epochs=EPOCHS,callbacks=cb)