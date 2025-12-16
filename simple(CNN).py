import keras
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_Datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range =0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip=True)
train_set = train_Datagen.flow_from_directory('dataset/training_set',
                                              target_size=(64,64),
                                              batch_size=32,
                                              class_mode="binary")

test_Datagen = ImageDataGenerator(rescale=1./255)
test_set = test_Datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64,64),
                                            batch_size=32,
                                            class_mode="binary")

cnn = keras.models.Sequential()

cnn.add(keras.layers.Conv2D(kernel_size=3 , activation="relu", filters=32, input_shape=[64,64,3]))
cnn.add(keras.layers.MaxPool2D(pool_size=2,strides=2))

cnn.add(keras.layers.Conv2D(kernel_size=3,activation="relu",filters=32))
cnn.add(keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(keras.layers.Flatten())

cnn.add(keras.layers.Dense(activation="relu",units=128))
cnn.add(keras.layers.Dense(activation="sigmoid",units=1))

cnn.compile(optimizer="adam",metrics=["accuracy"],loss="binary_crossentropy")
cnn.fit(x =train_set,validation_data = test_set, epochs = 25 )

test_image_1 = image.load_img("dataset/single_prediction/dog.jpg", target_size=(64,64))
test_image_1 = image.img_to_array(test_image_1)
test_image_1 = np.expand_dims(test_image_1,axis=0)
result_1 = cnn.predict(test_image_1)

class_indices = train_set.class_indices

print(train_set.class_indices)

if result_1[0][0] > 0.5:
    prediction_1 = "dog"
else:
    prediction_1 = "cat"

test_image_2 = image.load_img("dataset/single_prediction/image-79322-800.jpg",target_size=(64,64))
test_image_2 = image.img_to_array(test_image_2)
test_image_2 = np.expand_dims(test_image_2, axis=0)
result_2 = cnn.predict(test_image_2)

class_indices = train_set.class_indices

if result_2[0][0] > 0.5:
    prediction_2 = "dog"
else:
    prediction_2 = "cat"

print(train_set.class_indices)

print("okay i think that the first image is ",prediction_1,"and the second image is ",prediction_2)
