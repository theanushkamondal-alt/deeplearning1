> MNIST Interactive Digit Recognizer

A Convolutional Neural Network (CNN) built using TensorFlow/Keras that
learns handwritten digits from the MNIST dataset and predicts digits
from your own image files interactively.

Features

1 Trains a CNN on MNIST handwritten digits

2 Accepts custom image files from your computer 3 Automatic image
preprocessing

4 Smart color inversion

5 Displays prediction confidence

6 Shows the processed image visually

How It Works

Phase 1 — Training: The model learns digit patterns using 60,000 MNIST
training images.

Phase 2 — Interactive Prediction: User enters an image filename, the
program preprocesses it, and the model predicts the digit.

Model Architecture

1 Conv2D (32 filters) — Detects edges and strokes 2 MaxPooling — Reduces
image size

3 Conv2D (64 filters) — Detects complex shapes 4 MaxPooling — Further
compression

5 Flatten — Converts 2D features to 1D 6 Dense (128) — Learns digit
patterns

7 Dense (10 Softmax) — Outputs digit probabilities

Installation

pip install tensorflow numpy matplotlib pillow

How to Run

Run the script using: python your_script_name.py

Image Guidelines

1 Use a clear image of one handwritten digit 2 Keep the digit centered

3 Use a simple background

4 Supported formats: PNG, JPG, JPEG, BMP

Learning Outcomes

This project demonstrates CNNs, computer vision basics, image
preprocessing, and real-world AI inference.
