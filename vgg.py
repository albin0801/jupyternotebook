import numpy as np
import tensorflow as tf

mnist=np.load("/content/mnist.npz")
x_train=mnist["x_train"]
y_train=mnist["y_train"]
x_test=mnist["x_test"]
y_test=mnist["y_test"]
y_test = tf.keras.utils.to_categorical(y_test)
y_train = tf.keras.utils.to_categorical(y_train)
npad = ((0, 0), (10, 10), (10, 10))
x_train=np.pad(x_train, npad, 'constant', constant_values=(255))
x_train=np.array([np.stack((img,img,img), axis=-1) for img in x_train])
from tensorflow.keras import models,layers
from tensorflow.keras.applications import VGG16
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Add custom classification layers
model = models.Sequential()
model.add(vgg_model)
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
