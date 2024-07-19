import numpy as np
import cv2
import pygame
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Flatten, Dense
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

#====================== MODEL DEFINITION ==========================

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

weights_file = 'vgg19.weights.h5'

if os.path.exists(weights_file):
    model.build(input_shape=(None, 50, 50, 3))
    model.load_weights(weights_file)
else:
    print(f"Weights file '{weights_file}' not found. Please train the model first or provide the correct path to the weights file.")
    sys.exit()

# Initialize pygame for sound playback
pygame.mixer.init()

# Load MP3 files
sounds = {
    "angry": pygame.mixer.Sound(r"C:\Users\sadan\Downloads\2. Emotion Recognition\1.Emotion Recognition\angry.mp3"),
    "disgust": pygame.mixer.Sound(r"C:\Users\sadan\Downloads\2. Emotion Recognition\1.Emotion Recognition\disgust.mp3"),
    "fear": pygame.mixer.Sound(r"C:\Users\sadan\Downloads\2. Emotion Recognition\1.Emotion Recognition\fear.mp3"),
    "happy": pygame.mixer.Sound(r"C:\Users\sadan\Downloads\2. Emotion Recognition\1.Emotion Recognition\happy.mp3"),
    "sad": pygame.mixer.Sound(r"C:\Users\sadan\Downloads\2. Emotion Recognition\1.Emotion Recognition\sad.mp3"),
    "surprise": pygame.mixer.Sound(r"C:\Users\sadan\Downloads\2. Emotion Recognition\1.Emotion Recognition\suprise.mp3"),
    "neutral": pygame.mixer.Sound(r"C:\Users\sadan\Downloads\2. Emotion Recognition\1.Emotion Recognition\neutral.mp3"),
}

song_dict = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def preprocess_image(image):
    resized_image = cv2.resize(image, (50, 50))
    return resized_image

#====================== CAPTURE LIVE VIDEO ==========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    sys.exit()

print("Press 'c' to capture an image and 'q' to quit.")

# Initialize the tkinter GUI
root = tk.Tk()
root.title("Emotion Music Player")
root.geometry("600x400")

# Function to control music playback
def play_music(emotion):
    if pygame.mixer.get_busy():
        pygame.mixer.stop()
    sounds[emotion].play()
    status_label.config(text=f"Playing: {emotion_labels[song_dict_inv[emotion]]}")

def pause_music():
    pygame.mixer.pause()
    status_label.config(text="Paused")

def unpause_music():
    pygame.mixer.unpause()
    status_label.config(text=f"Playing: {emotion_labels[song_dict_inv[current_emotion]]}")

def stop_music():
    pygame.mixer.stop()
    status_label.config(text="Stopped")

def capture_image():
    global current_emotion
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("captured_image.jpg", frame)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = img.resize((200, 150), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        
        image_label.config(image=img)
        image_label.image = img
        
        resized_image = preprocess_image(frame)
        prediction = model.predict(np.expand_dims(resized_image, axis=0))
        emotion_index = np.argmax(prediction)
        current_emotion = song_dict[emotion_index]
        emotion_label.config(text=f'Mood: {emotion_labels[emotion_index]}')
        
        play_music(current_emotion)

song_dict_inv = {v: k for k, v in song_dict.items()}
current_emotion = ""

# Create and place widgets
capture_button = ttk.Button(root, text="Capture Image", command=capture_image)
capture_button.pack(pady=10)

pause_button = ttk.Button(root, text="Pause Music", command=pause_music)
pause_button.pack(pady=5)

unpause_button = ttk.Button(root, text="Unpause Music", command=unpause_music)
unpause_button.pack(pady=5)

stop_button = ttk.Button(root, text="Stop Music", command=stop_music)
stop_button.pack(pady=5)

status_label = tk.Label(root, text="Status: Ready", font=('Helvetica', 12))
status_label.pack(pady=10)

emotion_label = tk.Label(root, text="Mood: None", font=('Helvetica', 16))
emotion_label.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

# Start the tkinter GUI loop
root.mainloop()

cap.release()
cv2.destroyAllWindows()
