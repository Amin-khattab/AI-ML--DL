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

#PIPELINE

import keras
import numpy as np
import os
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Input, Resizing # <--- Added Resizing

model_path = "cifar10_godmode.h5"
image_path="test_images"

images_that_i_know = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# --- 2. REBUILD THE BRAIN (ARCHTECTURE) ---
# We build the empty shell exactly like we did in training.
print("Building model architecture locally...")

input_tensor = Input(shape=(32, 32, 3))

# Re-create the resize layer (This creates it with YOUR Python version)
resized_image = Resizing(96,96)(input_tensor)

# Re-download the ResNet structure (without weights for now)
base_model = ResNet50(include_top=False, input_tensor=resized_image, weights=None)

# Add the tail
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
predictions = Dense(10, activation="softmax", dtype="float32")(x)

model = Model(inputs=input_tensor, outputs=predictions)

# --- 3. LOAD THE MEMORIES (WEIGHTS) ---
# Instead of load_model(), we use load_weights()
# This ignores the broken bytecode in the file and just grabs the numbers.
print(f"Loading weights from {model_path}...")
model.load_weights(model_path)

files_found = [f for f in os.listdir(image_path)]

for file in files_found:
    img_path = os.path.join(image_path,file)

    try:
        img = image.load_img(img_path,target_size=(32,32))

        img_array = image.img_to_array(img)

        img_array = preprocess_input(img_array)

        img_array = np.expand_dims(img_array,0)

        prediction = model.predict(img_array)
        score = prediction[0]

        prediction_class = images_that_i_know[np.argmax(score)]
        condidence = np.max(score) *100

        print("image was: ",file)
        print("prdiction is: ",prediction_class)
        print("confidence is: ",condidence)

    except Exception as e:
        print("alas nothing happend")

