# Model testing script

import pathlib
import numpy as np
import tensorflow as tf

CATEGORIES = ['class_names[0]', 'class_names[1]', '...']


IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

test_image_path = pathlib.Path('') # insert the path to the test image

test_image = tf.keras.preprocessing.image.load_img(
    test_image_path,
    grayscale=False,
    target_size=IMG_SIZE,
    interpolation='nearest'
)

input_array = tf.keras.preprocessing.image.img_to_array(test_image)
input_array = np.array([input_array])

model = tf.keras.models.load_model('model_name', compile = True)

prediction = model.predict(input_array)
score = tf.nn.softmax(prediction[0])


print(
    "This fish most likely belongs to {} with a {:.2f} percent confidence."
    .format(CATEGORIES[np.argmax(score)], 100 * np.max(score))
)
