# Music recommender system using facial expressions

## Overview

This project is an application for real-time emotion recognition using a live webcam feed. It captures images from the webcam, processes them to determine the user's emotional state, and displays the detected emotion. The application leverages deep learning models to classify emotions.

## Features

- Real-time emotion recognition using a live webcam feed.
- Displays the captured image and the detected emotion.
- Utilizes pre-trained deep learning models for emotion classification.
- Simple graphical user interface (GUI) using `tkinter`.

## Requirements

- Python 3.x
- Required Python libraries:
  - `numpy`
  - `opencv-python`
  - `Pillow`
  - `tensorflow` (with Keras)
  - `tkinter` (usually included with Python)
  - `matplotlib` (for plotting images)

## Setup

1. **Install Required Libraries**

   You can install the required libraries using pip. Open a terminal or command prompt and run:

   ```sh
   pip install numpy opencv-python Pillow tensorflow matplotlib

## Download or Train the Model

If you don’t have a pre-trained model, you can train one using the provided code or download a pre-trained model that fits your needs. Ensure the model file (e.g., vgg19.h5.keras) is in the correct directory.

**Update Model Path**

If using a pre-trained model, ensure the path to the weights file in the code is correct. Update the path in your code to point to the correct weights file.

**Run the Application**

Run the main Python script to start the application. Make sure your webcam is connected.

  ```sh
  python mainfile.py
