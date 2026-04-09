import pandas as pd
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np 

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=0.1,
    height_shift_range=0.1,
    width_shift_range=0.1
)

test_datagen = ImageDataGenerator(
    rescale=1/255
)

train_set  = train_datagen.flow(
    x_train,
    y_train,
    batch_size=32
)

test_set= test_datagen.flow(
    x_test,
    y_test,
    batch_size=32
)

mamosa = ReduceLROnPlateau(
    monitor="val_accuracy",
    patience=4,
    factor=0.5,
    min_lr=0.00001,
)

cnn = keras.Sequential()

cnn.add(keras.layers.Conv2D(kernel_size=3,filters=32,activation=None,padding="same",input_shape=(28,28,1)))
cnn.add(keras.layers.BatchNormalization())
cnn.add(keras.layers.Activation("relu"))
cnn.add(keras.layers.Dropout(0.25))
cnn.add(keras.layers.MaxPool2D(pool_size=2,strides=2))

cnn.add(keras.layers.Conv2D(kernel_size=3,filters=64,padding="same",activation=None))
cnn.add(keras.layers.BatchNormalization())
cnn.add(keras.layers.Activation("relu"))
cnn.add(keras.layers.Dropout(0.25))
cnn.add(keras.layers.MaxPool2D(pool_size=2,strides=2))

cnn.add(keras.layers.Conv2D(kernel_size=3,filters=128,padding="same",activation=None))
cnn.add(keras.layers.BatchNormalization())
cnn.add(keras.layers.Activation("relu"))
cnn.add(keras.layers.Dropout(0.25))
cnn.add(keras.layers.MaxPool2D(pool_size=2,strides=2))

cnn.add(keras.layers.GlobalAveragePooling2D())

cnn.add(keras.layers.Dense(units=128,activation="relu"))
cnn.add(keras.layers.Dropout(0.5))

cnn.add(keras.layers.Dense(units=10,activation="softmax"))

cnn.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
cnn.fit(x = train_set,validation_data=test_set,epochs=50,callbacks=mamosa)
