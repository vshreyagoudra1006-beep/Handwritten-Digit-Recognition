import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("digit_model.keras")

# Create window
root = tk.Tk()
root.title("Handwritten Digit Recognition")
root.geometry("400x450")

# Create canvas
canvas = tk.Canvas(root, width=280, height=280, bg="white")
canvas.pack(pady=10)

# Create blank image
image = Image.new("L", (280, 280), "white")
draw = ImageDraw.Draw(image)

# Draw on canvas
def paint(event):
    x, y = event.x, event.y
    r = 8
    canvas.create_oval(x-r, y-r, x+r, y+r,
                       fill="black", outline="black")
    draw.ellipse((x-r, y-r, x+r, y+r), fill="black")

canvas.bind("<B1-Motion>", paint)

# Predict digit
def predict():

    # Resize to MNIST size
    img = image.resize((28, 28))

    # Invert colors
    img = ImageOps.invert(img)

    # Convert to array
    img = np.array(img)

    # Normalize
    img = img / 255.0

    # Reshape
    img = img.reshape(1, 28, 28)

    # Predict
    prediction = model.predict(img, verbose=0)

    digit = np.argmax(prediction)

    result.config(text=f"AI Prediction: {digit}")

# Clear canvas
def clear():
    canvas.delete("all")
    draw.rectangle((0, 0, 280, 280), fill="white")
    result.config(text="AI Prediction:")

# Buttons
tk.Button(root,
          text="Predict",
          command=predict,
          width=15).pack(pady=5)

tk.Button(root,
          text="Clear",
          command=clear,
          width=15).pack()

# Result label
result = tk.Label(root,
                  text="AI Prediction:",
                  font=("Arial", 22))

result.pack(pady=20)

root.mainloop()