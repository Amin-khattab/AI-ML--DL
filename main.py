import keras
import numpy as np
import os
from keras.preprocessing import image


model_path = "cifar10_best_model_B.h5"
image_path="test_images"

images_that_i_know = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

model = keras.models.load_model(model_path)

files_found = [f for f in os.listdir(image_path)]

for file in files_found:
    img_path = os.path.join(image_path,file)

    try:
        img = image.load_img(img_path,target_size=(32,32))

        img_array = image.img_to_array(img)
        img_array = img_array/255.0

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
