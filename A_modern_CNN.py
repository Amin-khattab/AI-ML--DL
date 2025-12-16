import keras
import numpy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

train_datagen = ImageDataGenerator(
    rescale = 1/255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range= 0.1,
    horizontal_flip= True,
    zoom_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1/255)

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

cnn = keras.Sequential()

cnn.add(keras.layers.Conv2D(kernel_size=3, filters=32, activation=None,padding="same", input_shape=(32,32,3)))
cnn.add(keras.layers.BatchNormalization())
cnn.add(keras.layers.Activation("relu"))
cnn.add(keras.layers.Dropout(0.25))
cnn.add(keras.layers.MaxPool2D(pool_size=2,strides=2))

cnn.add(keras.layers.Conv2D(kernel_size=3,filters=64,activation=None,padding="same"))
cnn.add(keras.layers.BatchNormalization())
cnn.add(keras.layers.Activation("relu"))
cnn.add(keras.layers.Dropout(0.25))
cnn.add(keras.layers.MaxPool2D(strides=2,pool_size=2))

cnn.add(keras.layers.Conv2D(kernel_size=3,filters=128,activation=None,padding="same"))
cnn.add(keras.layers.BatchNormalization())
cnn.add(keras.layers.Activation("relu"))
cnn.add(keras.layers.Dropout(0.25))
cnn.add(keras.layers.MaxPool2D(strides=2,pool_size=2))

cnn.add(keras.layers.Conv2D(kernel_size=3,filters=128,activation=None,padding="same"))
cnn.add(keras.layers.BatchNormalization())
cnn.add(keras.layers.Activation("relu"))
cnn.add(keras.layers.Dropout(0.25))
cnn.add(keras.layers.MaxPool2D(strides=2,pool_size=2))

cnn.add(keras.layers.GlobalAveragePooling2D())

cnn.add(keras.layers.Dense(units=256,activation="relu"))
cnn.add(keras.layers.BatchNormalization())
cnn.add(keras.layers.Dropout(0.5))
cnn.add(keras.layers.Dense(units=10,activation="softmax"))

cnn.compile(optimizer="Adam", loss = "sparse_categorical_crossentropy",metrics=["accuracy"])
cnn.fit(x = train_set,validation_data=test_set,epochs = 50)

import matplotlib.pyplot as plt

# 1. Capture the history from your training
history = cnn.history.history

# 2. Plot Accuracy
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 3. Plot Loss
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

import numpy as np

# CIFAR-10 Class Names (so we can read the output)
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# 1. Pick a random image from the test set (e.g., image #7)
index = 7
test_image = x_test[index]
true_label = y_test[index][0]

# 2. Preprocess it (Remember to Rescale like you did in training!)
# The model expects a batch, so we add an extra dimension: (1, 32, 32, 3)
input_image = np.expand_dims(test_image / 255.0, axis=0)

# 3. Predict
predictions = cnn.predict(input_image)
predicted_class_index = np.argmax(predictions) # Get the index of the highest score

print(f"True Label: {class_names[true_label]}")
print(f"Prediction: {class_names[predicted_class_index]}")

# Show the image so you can judge for yourself
plt.imshow(test_image)
plt.show()
