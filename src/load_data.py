import tensorflow as tf

train = tf.keras.preprocessing.image_dataset_from_directory(
    "data/train",
    image_size=(224,224),
    batch_size=32
)

val = tf.keras.preprocessing.image_dataset_from_directory(
    "data/val",
    image_size=(224,224),
    batch_size=32
)

test = tf.keras.preprocessing.image_dataset_from_directory(
    "data/test",
    image_size=(224,224),
    batch_size=32
)

print("Classes detected:", train.class_names)
