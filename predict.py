import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("digit_model.keras")

# Load the test dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize images
x_test = x_test / 255.0

# Ask user for an image index
index = int(input("Enter a test image number (0-9999): "))

# Predict
prediction = model.predict(x_test[index].reshape(1, 28, 28))

predicted_digit = prediction.argmax()

print("AI Prediction:", predicted_digit)
print("Actual Digit:", y_test[index])

# Show image
plt.imshow(x_test[index], cmap="gray")
plt.title(f"Prediction: {predicted_digit} | Actual: {y_test[index]}")
plt.axis("off")
plt.show()