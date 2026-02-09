import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.datasets import mnist
from PIL import Image

# ==========================================
# 1. DATA PREPARATION & TRAINING
# ==========================================
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape for CNN
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# Build Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

print("--- Step 1: Training Model on MNIST ---")
model.fit(x_train, y_train, epochs=3, validation_split=0.1, verbose=1)

# ==========================================
# 2. INTERACTIVE IMAGE PREDICTION
# ==========================================

def run_interactive_prediction():
    print("\n" + "="*40)
    print("MNIST INTERACTIVE PREDICTOR")
    print("="*40)
    print("Instructions: Type the full path or filename of your image.")
    print("Type 'exit' or 'q' to quit.")

    while True:
        file_input = input("\nEnter image filename: ").strip()

        if file_input.lower() in ['q', 'exit']:
            print("Goodbye!")
            break

        if not os.path.isfile(file_input):
            print(f"Error: File '{file_input}' not found. Please try again.")
            continue

        try:
            # Load and Preprocess
            img = Image.open(file_input).convert('L') # Grayscale
            img = img.resize((28, 28))                # MNIST size
            
            img_array = np.array(img) / 255.0         # Normalize
            
            # Smart Inversion: 
            # MNIST is white-on-black. If image average is bright, invert it.
            if np.mean(img_array) > 0.5:
                img_array = 1.0 - img_array
            
            # Reshape for model (1, 28, 28, 1)
            input_tensor = img_array.reshape(1, 28, 28, 1)
            
            # Predict
            prediction = model.predict(input_tensor, verbose=0)
            digit = np.argmax(prediction)
            confidence = np.max(prediction)

            # Output to terminal
            print(f">> Result: {digit} ({confidence:.2%} confidence)")

            # Visual Feedback
            plt.figure(figsize=(4, 4))
            plt.imshow(img_array, cmap='gray')
            plt.title(f"Predicted: {digit}")
            plt.axis('off')
            plt.show()

        except Exception as e:
            print(f"An error occurred while processing the image: {e}")

if __name__ == "__main__":
    run_interactive_prediction()
