import numpy as np
import os
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Paths to your dataset
data_paths = {
    'angry': 'Input_Data/angry/',
    'disgust': 'Input_Data/disgust/',
    'fear': 'Input_Data/fear/',
    'happy': 'Input_Data/happy/',
    'neutral': 'Input_Data/neutral/',
    'sad': 'Input_Data/sad/',
    'surprise': 'Input_Data/surprise/'
}

# Load and preprocess dataset
def load_data(data_paths):
    images = []
    labels = []
    label_map = {label: idx for idx, label in enumerate(data_paths.keys())}
    for label, path in data_paths.items():
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (50, 50))
                images.append(img)
                labels.append(label_map[label])
    return np.array(images), np.array(labels)

images, labels = load_data(data_paths)
labels = to_categorical(labels, num_classes=7)

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the model
def create_vgg19_model():
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=(50, 50, 3))
    for layer in vgg.layers:
        layer.trainable = False

    model = Sequential()
    model.add(vgg)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(7, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model

model = create_vgg19_model()
model.summary()

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Save the weights
model.save_weights('vgg19.weights.h5')
print("Model trained and weights saved.")
