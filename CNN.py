import keras
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

train_set = train_datagen.flow_from_directory('dataset/training_set',
                                              target_size=(64, 64),
                                              batch_size=32,
                                              class_mode="binary"
                                              )

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory("dataset/test_set",
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary'
                                            )

cnn = tf.keras.models.Sequential()

cnn.add(keras.layers.Conv2D(kernel_size=3, filters=35, activation="relu", input_shape=[64, 64, 3]))
cnn.add(keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(keras.layers.Conv2D(kernel_size=3, filters=35, activation="relu"))
cnn.add(keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(keras.layers.Flatten())

cnn.add(keras.layers.Dense(units=128, activation="relu"))
cnn.add(keras.layers.Dense(units=1, activation="sigmoid"))

cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
cnn.fit(x=train_set, validation_data=test_set, epochs=32)

test_image_1 = image.load_img("dataset/single_prediction/dog.jpg", target_size=(64, 64))
test_image_1 = image.img_to_array(test_image_1)
test_image_1 = np.expand_dims(test_image_1, axis=0)
result_1 = cnn.predict(test_image_1)
train_set.class_indices

if result_1[0][0] == 1:
    prediction_1 = "dog"
else:
    prediction_1 = "cat"

test_image_2 = image.load_img("dataset/single_prediction/dog.jpg", target_size=(64, 64))
test_image_2 = image.img_to_array(test_image_2)
test_image_2 = np.expand_dims(test_image_2, axis=0)
result_2 = cnn.predict(test_image_2)
train_set.class_indices

if result_2[0][0] == 1:
    prediction_2 = "dog"
else:
    prediction_2 = "cat"

print("so for the first picture i think its", prediction_1, "then for the second image i think its", prediction_2)
