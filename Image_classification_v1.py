# Simple convolutional neural network (CNN)

import pathlib
import tensorflow as tf

from keras.utils import image_dataset_from_directory
from keras import layers
from keras.models import Sequential

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
IMG_SHAPE = IMG_SIZE + (3,)
BATCH_SIZE = 32
EPOCHS = 15

train_images = pathlib.Path('') # insert the path to the folder with images for training

image_count = len(list(train_images.glob('*/*.jpg')))
print(image_count)

train_ds = image_dataset_from_directory(
  train_images,
  batch_size=BATCH_SIZE,
  image_size=IMG_SIZE,
  seed=123,
  validation_split=0.2,
  subset="training"
)

validation_ds = image_dataset_from_directory(
  train_images,
  batch_size=BATCH_SIZE,
  image_size=IMG_SIZE,
  seed=123,
  validation_split=0.2,
  subset="validation"
)

class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

data_augmentation = Sequential(
  [
    layers.RandomFlip("horizontal", input_shape=(IMG_SHAPE)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

history = model.fit(
  train_ds,
  validation_data=validation_ds,
  epochs=EPOCHS
)

model.save('model_name') # enter the model name