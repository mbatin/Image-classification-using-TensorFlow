# Image classification using pre-trained VGG16 model

import pathlib
import tensorflow as tf

from keras.utils import image_dataset_from_directory
from keras import layers
from keras.models import Sequential
from keras import Model

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
IMG_SHAPE = IMG_SIZE + (3,)
BATCH_SIZE = 32
EPOCHS = 5
FINE_TUNE_EPOCHS = 5
TOTAL_EPOCHS =  EPOCHS + FINE_TUNE_EPOCHS

train_images = pathlib.Path('') # insert the path to the folder with images for training

image_count = len(list(train_images.glob('*/*.jpg')))

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

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

data_augmentation = Sequential(
  [
    layers.RandomFlip("horizontal", input_shape=IMG_SHAPE),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

preprocess_input = tf.keras.applications.vgg16.preprocess_input

base_model = tf.keras.applications.VGG16(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

base_model.trainable = False

base_model.summary()

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(1, activation='softmax')(x)
model = Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()

history = model.fit(
  train_ds,
  validation_data=validation_ds,
  epochs=EPOCHS
)

base_model.trainable = True

print(len(base_model.layers))

for layer in base_model.layers[:15]:
    layer.trainable = False
    
for i, layer in enumerate(base_model.layers):
    print(i, layer.name, layer.trainable)

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
              metrics=['accuracy'])

model.summary()

history_fine_tuning = model.fit(train_ds,
                         epochs=TOTAL_EPOCHS,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_ds)

model.save('model_name') # enter the model name
