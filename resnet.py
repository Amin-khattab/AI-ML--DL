import tensorflow
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Lambda,GlobalAveragePooling2D,Dense,Input
from keras.applications import ResNet50 , preprocess_input
from keras.utils import to_categorical
from keras import mixed_precision
import numpy as np

mixed_precision.set_global_policy('mixed_float16')

BATCH_SIZE = 256
EPOCH = 10

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

input_tensor = Input(shape=(32,32,3))
resized_image = Lambda(lambda image :tensorflow.image.resize(image,(96,96)))(input_tensor)
base_model = ResNet50(weights="imagenet", include_top=False,input_tensor=resized_image)
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256,activation="relu")(x)

predictions = Dense(10,activation="softmax", dtype="float32")(x)

model = Model(input_tensor = input_tensor,outputs = predictions)

model.compile(optimizer="adam",loss = "categorical_crossentropy", metrics=["accuracy"])

print("4. Saving Model...")
model.save('cifar10_godmode.h5')
print("✅ DONE! Saved as cifar10_godmode.h5")
