import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("model.h5")  # Ensure model.h5 is in the same directory

# Define class labels (ensure all classes are listed)
monument_classes = [
    "Aga Khan Palace", "Badrinath Temple", "Bekal", "Bhudha Temple", "Brihadeshwara Temple", 
    "Cathederal", "Champaner", "Chandi Devi mandir hariwar", "Cheese", "Chhatrapati Shivaji terminus",
    "Chittorgarh Padmini Lake Palace", "Daman", "Diu Museum", "Fatehpur Sikri Fort", "Hampi", 
    "Hoshang Shah Tomb", "India Gate", "Isarlat Sargasooli", "ajanta caves", "ajmeri gate delhi", 
    "albert hall museum", "bara imambara", "barsi gate hansi old", "basilica of bom jesus", 
    "bharat mata mandir haridwar", "bhoramdev mandir", "bidar fort", "buland darwaza", 
    "byzantine architecture", "chandigarh college of architecture", "chapora fort", "charminar", 
    "chhatisgarh ke saat ajube", "chhatrapati shivaji statue", "chittorgarh", "city palace", 
    "dhamek stupa", "diu", "dome", "dubdi monastery yuksom sikkim", "falaknuma palace", 
    "fatehpur sikri", "ford Auguda", "fortification", "gol ghar", "golden temple", "hawa mahal", 
    "hidimbi devi temple", "hindu temple"
]


# Function to predict the monument
def predict_from_file(file_path):
    try:
        # Load and preprocess the image
        image = Image.open(file_path).resize((64, 64))
        image_array = np.expand_dims(np.array(image) / 255.0, axis=0)

        # Debugging info
        print(f"Image shape: {image_array.shape}")

        # Make prediction
        prediction = model.predict(image_array)
        print(f"Raw Prediction Output: {prediction}")

        # Get the highest confidence prediction
        predicted_index = np.argmax(prediction)
        confidence_score = np.max(prediction)

        # Handle out-of-range index error
        if predicted_index >= len(monument_classes):
            predicted_monument = "Unknown Monument"
        else:
            predicted_monument = monument_classes[predicted_index]

        # Print and display result
        print(f"Predicted: {predicted_monument}, Confidence: {confidence_score:.2f}")
        result_label.config(text=f"Prediction: {predicted_monument}\nConfidence: {confidence_score:.2f}")
        root.update()
    except Exception as e:
        result_label.config(text=f"Error: {str(e)}")

# Function to select an image
def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        img = Image.open(file_path).resize((200, 200))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
        predict_from_file(file_path)

# Create GUI
root = tk.Tk()
root.title("Monument Classifier")

frame = tk.Frame(root)
frame.pack(pady=20)

btn = tk.Button(frame, text="Select Image", command=select_image)
btn.pack()

image_label = tk.Label(root)
image_label.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()