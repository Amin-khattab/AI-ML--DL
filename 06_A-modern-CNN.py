import pandas as pd
import keras
from keras.src.layers import Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
from keras.callbacks import ReduceLROnPlateau

(x_train,y_train),(x_test,y_test ) = cifar10.load_data()

train_datagen = ImageDataGenerator(
    rescale= 1/255,
    horizontal_flip= True,
    rotation_range= 0.1,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    zoom_range=0.05
)

test_datagen = ImageDataGenerator(
    rescale=1/255
)

train_set = train_datagen.flow(
    x_train,
    y_train,
    batch_size=64
)

test_set = test_datagen.flow(
    x_test,
    y_test,
    batch_size=64
)

tracker= ReduceLROnPlateau(
    monitor="val_accuracy",
    patience=3,
    factor=0.5,
    min_lr=0.0001
)

cnn = keras.Sequential()

cnn.add(keras.layers.Conv2D(kernel_size=3,filters=32,padding="same",activation = None, input_shape=(32,32,3)))
cnn.add(keras.layers.BatchNormalization())
cnn.add(keras.layers.Activation("relu"))
cnn.add(keras.layers.Dropout(0.25))
cnn.add(keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(keras.layers.Conv2D(kernel_size=3,filters=64,padding="same",activation = None))
cnn.add(keras.layers.BatchNormalization())
cnn.add(keras.layers.Activation("relu"))
cnn.add(keras.layers.Dropout(0.25))
cnn.add(keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(keras.layers.Conv2D(kernel_size=3,filters=128,padding="same",activation = None))
cnn.add(keras.layers.BatchNormalization())
cnn.add(keras.layers.Activation("relu"))
cnn.add(keras.layers.Dropout(0.25))
cnn.add(keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(keras.layers.GlobalAveragePooling2D())

cnn.add(keras.layers.Dense(units=128,activation="relu"))
cnn.add(keras.layers.Dropout(0.5))

cnn.add(keras.layers.Dense(units=10,activation="softmax"))

cnn.compile(optimizer="Adam", loss="sparse_categorical_crossentropy",metrics=["accuracy"])
cnn.fit(x = train_set,validation_data=test_set,epochs = 10,callbacks = tracker)
