# NVIDIA Jetson Nano Banknote Counter Project
# Utillizing the NVIDIA Jetson Nano for banknote classification 
# and to control LEGO power functions' motors and servomotors 
# to feed single notes to realize a money counting LEGO MOC.
#
# TRAINING A DEEP NEURAL NETWORK
#
# to classify banknotes
# - Tensorflow 2.x model
# - transfer learning (VGG16 with ImageNet weights as base network)
#   or simple 5 layer CNN from scratch
# - optional: continue training of a model
#
# Tf2 transfer learning as described in the Tf2 tutorial:
# [1] https://www.tensorflow.org/tutorials/images/transfer_learning
# and in this blog post by Robert Thas John: 
# [2] https://towardsdatascience.com/transfer-learning-with-tf-2-0-ff960901046d
# 
#
# https://www.youtube.com/channel/UCqL-arxKMK15cO7SVVBH2lA
# sh2021

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pathlib
print(tf.__version__)

# flags instead of args (Fortran < 2003 style...)
continue_training = True # continue training of a previously trained model
scratch_model = False    # start a new simple CNN model from scratch

num_classes = 6          # number of classes 

img_height =  80         # input image size (reduced to increase possible batch size given hardware limitations)
img_width =   80
NUM_CHANNELS = 3 

batch_size = 5          # max batch size possible for training on Jetson 4GB
nr_epochs = 10          # number of epochs 

if continue_training:   # do we have a model trained already -> load this one
 model = load_model('model_vgg16_augmented_no_dropout_closer_bg_further_new_model7.h5')
else:                   # else transfer learning using VGG16 with ImageNet weights
 base_model = tf.keras.applications.vgg16.VGG16(input_shape=(img_height, img_width, NUM_CHANNELS), include_top=False, weights='imagenet', pooling='avg')
 base_model.trainable = False # freeze base layers
 print(base_model.summary())

 x = base_model.output
 x = layers.Dense(25, activation='relu')(x)  # add 3 dense layers (25,10,#classes) on top of VGG16 base
 x = layers.Dense(10, activation='relu')(x)
 #x = tf.keras.layers.Dropout(0.1)(x)        # dropout might be used to increase robustness
 x = layers.Dense(num_classes, activation='sigmoid')(x)
 model_3 = models.Model(inputs=base_model.input, outputs=x)
 print(model_3.summary())

data_dir = '/home/steinheilig/jetson-inference/python/training/classification/data/money_close/train/' # all data stored in this PyTorch style data directory
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print('Toral number of images:',image_count)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.05,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.05,
  subset="validation",
  seed=123, # note that here the same seed is used, to avoid information leakage into the validation set.
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)  # class names (and labels) extracted from the directory structure 

# save some training example overview to disc
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(3):
    ax = plt.subplot(3, 1, i + 1)
    if batch_size > 3:
         plt.imshow(images[i].numpy().astype("uint8"))
         plt.title(class_names[labels[i]])
    plt.axis("off")
plt.savefig('demo_smaller.png')

# increase performance of dataset handling
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255) # rescale input values to [0:1] for better ANN performance

# data augmentation
data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomRotation(0.2),  #  parameters can be tweeked, e.g. 0.1 first run n epochs & 0.2 later
  layers.experimental.preprocessing.RandomContrast(factor=0.3, seed=42),  
  layers.experimental.preprocessing.RandomZoom(height_factor=(-0.01, -0.01),  width_factor=None, fill_mode='reflect',
    interpolation='bilinear', seed=42, fill_value=0.0),
  layers.experimental.preprocessing.RandomTranslation(height_factor=0.05, width_factor=0.05, fill_mode='reflect',  
    interpolation='bilinear', seed=42, fill_value=0.0)
])

if not(continue_training): # if we do not contiue training a model...
 if scratch_model:    # define model // train simple CNN from scratch (3 convolution, 2 dense layers)
    model = tf.keras.Sequential([
      layers.experimental.preprocessing.Rescaling(1./255),
      data_augmentation,
      #layers.BatchNormalization(),
      layers.Conv2D(32, 3, activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
    ])
 else:              # add preprocessing steps (Rescaling and data augmentation) to the pretrained model 
     model = tf.keras.Sequential([
      layers.experimental.preprocessing.Rescaling(1./255),
      data_augmentation,
      model_3
    ])

# compile the model
model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), # from_logits because no final softmax layer is used [1]
  metrics=['accuracy'])

# fit the model
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=nr_epochs
)

# save model to file
model.save('model_vgg16_augmented_no_dropout_closer_bg_further_new_model8.h5')
