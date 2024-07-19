# Music recommender system using emotions

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
  - `pygame`
  - `Pillow`
  - `tensorflow` (with Keras)
  - `tkinter` (usually included with Python)
  - `matplotlib` (for plotting images)

## Setup

1. **Install Required Libraries**
You can install the required libraries using pip. Open a terminal or command prompt and run:
   
   ```s
   pip install numpy opencv-python Pillow tensorflow matplotlib

2. **Download or Train the Model**

      If you don’t have a pre-trained model, you can train one using the provided code or download a pre-trained model that fits your needs. Ensure the model file (e.g., vgg19.h5.keras) is in the correct directory.

3. **Update Model Path**

      If using a pre-trained model, ensure the path to the weights file in the code is correct. Update the path in your code to point to the correct weights file.

4. **Run the Application**

      Run the main Python script files one by one to start the application. Make sure your webcam is connected.

```s
  python Training.py
```
```s
  python Prediction.py 
```
5. **Capture and Analyze Images**

      The application will display the live feed from the webcam. Click the "Capture Image" button to take a snapshot from the webcam. The application will process the captured image to determine the emotion and display it.

## Usage

• Launch the application. The live webcam feed will appear in the tkinter window.

• Click the "Capture Image" button to take a snapshot from the webcam.

• The application will process the captured image to determine the emotion and display it.

## Troubleshooting

• Webcam Not Opening: Ensure your webcam is properly connected and not in use by another application.

• Model Loading Issues: Ensure the path to the model weights file is correct and that the file exists.


• Errors During Execution: Check the error messages for details. Common issues include missing dependencies or incorrect model paths.

## Acknowledgments
• [OpenCV](https://opencv.org/) for computer vision capabilities.

• [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for deep learning models.

• [tkinter](https://docs.python.org/3/library/tkinter.html) for GUI creation.

## License
This project is licensed under the [MIT License](https://opensource.org/license/mit)
### Key Sections Explained

- **Overview**: A brief description of the project's purpose and functionality.
- **Features**: Highlights the core features of the application.
- **Requirements**: Lists the software and libraries needed to run the project.
- **Setup**: Instructions on how to prepare the environment and run the application.
- **Usage**: How to use the application, including capturing and analyzing images.
- **Troubleshooting**: Common problems and their solutions.
- **Acknowledgments**: Credits to the tools and libraries used.

This README file should help users understand how to set up and use your emotion recognition application effectively. Adjust the content as needed to fit any additional details or specific instructions relevant to your project.




