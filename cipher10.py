import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test) 
def create_model(hidden_units=None, activation=None):
    model = models.Sequential([
        layers.Flatten(input_shape=(32, 32, 3)),
        layers.Dense(hidden_units[0], activation=activation),
        layers.Dense(hidden_units[1], activation=activation), 
        layers.Dense(hidden_units[2], activation=activation), 
        layers.Dense(10, activation='softmax')
    ])
    return model
hidden_units = [512, 256, 128]
activation = 'relu'
results_dict = {}
counter = 1
model = create_model(hidden_units=hidden_units, activation=activation)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
_, test_acc = model.evaluate(x_test, y_test)
model_info = {
    "Hidden units": hidden_units,
    "Activation": activation,
    "Test accuracy": round(test_acc * 100, 4)
}
results_dict[counter] = model_info  
counter += 1
for key, value in results_dict.items():
    print(f"Run {key}:")
    for info_key, info_value in value.items():
        print(f"{info_key}: {info_value}")
    print("- -" * 15) 
print("\n")
max_accuracy_run = max(results_dict, key=lambda k: results_dict[k]["Test accuracy"])
max_accuracy_info = results_dict[max_accuracy_run]
print("Run with the highest test accuracy:")
print(f"Run {max_accuracy_run}:")
for info_key, info_value in max_accuracy_info.items():
    print(f"{info_key}: {info_value}")
num_images = 3
sample_images = x_train[:num_images]
predictions = model.predict(sample_images)
def plot_probability_meter(predictions, image):
    class_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 2))
    axs[0].imshow(image)
    axs[0].axis('off')
    axs[1].barh(class_labels, predictions[0], color='blue')
    axs[1].set_xlim([0, 1])
    plt.tight_layout()
    plt.show()
for i in range(num_images):
    plot_probability_meter(predictions[i:i+1], sample_images[i])
