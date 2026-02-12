import keras
import tensorflow as tf
from tensorflow.keras import layers,models,mixed_precision
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10

policy = mixed_precision.Policy("mixed_bfloat16")
mixed_precision.set_global_policy(policy)

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

train_datagen = ImageDataGenerator(
    rescale=1/255,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode="nearest",
    rotation_range=10
)

test_datagen = ImageDataGenerator(rescale=1/255)

train_set = train_datagen.flow(x_train,y_train,batch_size=64)
test_set = test_datagen.flow(x_test,y_test,batch_size=64)

def build_pretrained_resnet():
    input_tensor = layers.Input(shape=(32,32,3))

    x = layers.UpSampling2D(size=(3,3))(input_tensor)

    base_model = ResNet50V2(include_top=False,weights="imagenet",input_tensor=x)
    base_model.trainable = False

    x = base_model.outputs
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(units=256,activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=10,activation="softmax",dtype="float32")(x)

    return models.Model(input_tensor,outputs)

resnet_pt = build_pretrained_resnet()

tracker = ReduceLROnPlateau(
    patience=3,
    monitor="val_accuracy",
    factor=0.5,
    verbose=1
)

early_stop = EarlyStopping(
    patience=10,
    verbose=1,
    monitor="val_accuracy",
    restore_best_weights=True
)

resnet_pt.compile(optimizer=keras.optimizers.Adam(1e-4),loss = "sparse_categorical_crossentropy",metrics=["accuracy"])

resnet_pt.fit(x=train_set,validation_data=test_set,callbacks=[early_stop,tracker],epochs=50)

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

