import tensorflow as tf
from tensorflow.keras import layers, models, initializers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
# Step 1: Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels, test_labels = to_categorical(train_labels), to_categorical(test_labels)
# Step 2: Preprocess the Data
# ...
# Step 3: Define the Neural Network Architecture
def create_model(initializer, dropout_rate=0.0, l2_regularizer=None):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(32, 32, 3)))
    model.add(layers.Dense(512, kernel_initializer=initializer, kernel_regularizer=l2_regularizer, activation='relu'))
    model.add(layers.Dense(256, kernel_initializer=initializer, kernel_regularizer=l2_regularizer, activation='relu'))
    model.add(layers.Dense(128, kernel_initializer=initializer, kernel_regularizer=l2_regularizer, activation='relu'))
    model.add(layers.Dense(64, kernel_initializer=initializer, kernel_regularizer=l2_regularizer, activation='relu'))  # Additional dense layer
    model.add(layers.Dense(32, kernel_initializer=initializer, kernel_regularizer=l2_regularizer, activation='relu'))  # Additional dense layer
    model.add(layers.Dense(10, activation='softmax'))
    return model
# Step 4: Choose Weight Initialization Techniques
xavier_initializer = initializers.glorot_normal()
kaiming_initializer = initializers.he_normal()

# Step 5: Compile the Model
# ...
# Step 6: Train the Model with Different Configurations
xavier_model = create_model(xavier_initializer, dropout_rate=0.3, l2_regularizer=tf.keras.regularizers.l2(0.001))
kaiming_model = create_model(kaiming_initializer, dropout_rate=0.3, l2_regularizer=tf.keras.regularizers.l2(0.001))
xavier_model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
kaiming_model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
xavier_history = xavier_model.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))
kaiming_history = kaiming_model.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))
# Step 7: Evaluate and Visualize Performance
# ...
# Step 8: Display Output
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(xavier_history.history['accuracy'], label='Xavier (train)')
plt.plot(xavier_history.history['val_accuracy'], label='Xavier (val)')
plt.title('Xavier Initialization')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(kaiming_history.history['accuracy'], label='Kaiming (train)')
plt.plot(kaiming_history.history['val_accuracy'], label='Kaiming (val)')
plt.title('Kaiming Initialization')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()
