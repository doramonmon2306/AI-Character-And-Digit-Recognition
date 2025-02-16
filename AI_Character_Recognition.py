import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import torchvision as tv
import torch
from torch.utils.data import DataLoader
import string

##########################################
# PART I: Model_Charact_AI

def Model_Charact_AI():
    
    transform = tv.transforms.Compose([
    tv.transforms.Grayscale(num_output_channels=1), # Ensure grayscale
    tv.transforms.Resize((28, 28)),
    tv.transforms.Lambda(lambda img: tv.transforms.functional.vflip(img)),
    tv.transforms.Lambda(lambda img: tv.transforms.functional.rotate(img, -90)),  
    tv.transforms.ToTensor()
    ])

    # Load dataset
    train_dataset = tv.datasets.EMNIST(root='./data', split='byclass', train=True, download=True, transform=transform)
    test_dataset = tv.datasets.EMNIST(root='./data', split='byclass', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=1, padding=1),
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Flatten(),
        torch.nn.Linear(64*7*7, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(128, 62)
    )

    loss_f = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    num_epochs = 2
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} started")
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"Processing batch {batch_idx+1}/{len(train_loader)}", end="\r")
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_f(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "character_recognition_model.pth")
    print("Model saved successfully!")

##########################################
# PART II: Writing_Test

def Writing_Test():
    global model_loaded, image, draw_pil

    # Function to load the trained model
    def load_model():
        model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(64*7*7, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(128, 62)
        )

        model.load_state_dict(torch.load("character_recognition_model.pth", map_location=torch.device('cpu')))
        model.eval()  # Set to evaluation mode
        return model

    # Load the trained model
    model_loaded = load_model()

    window = tk.Tk()
    window.geometry("600x600")
    window.title("Writing Test")

    Canvas = tk.Canvas(window, bg="white", height=300, width=300) 
    Canvas.pack()

    # Create a PIL image (grayscale)
    image = Image.new("L", (300, 300), color=0)
    draw_pil = ImageDraw.Draw(image)

    # Drawing function
    def draw_digit(event):
        x, y = event.x, event.y
        r = 10
        Canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", width=20)
        draw_pil.ellipse([x - r, y - r, x + r, y + r], fill=255)

    # Clear function
    def clear():
        global image, draw_pil
        Canvas.delete("all")
        image = Image.new("L", (300, 300), color=0) 
        draw_pil = ImageDraw.Draw(image)

    def emnist_label_to_char(label):
        if 0 <= label <= 9:  
            return str(label)
        elif 10 <= label <= 35: 
            return chr(label + 55)
        elif 36 <= label <= 61:
            return chr(label + 61) 
        else:
            return "?"  # Unknown label
    
    # Prediction function
    def predict_the_charact():
        img_resized = image.resize((28, 28))  # Resize to match model input
        img_array = np.array(img_resized, dtype=np.float32) / 255.0  # Normalize
        img_array = img_array.reshape(1, 1, 28, 28)  # (batch, channel, height, width)
        img_tensor = torch.tensor(img_array, dtype=torch.float32)

        with torch.no_grad():
            prediction = model_loaded(img_tensor)
            predicted_label = torch.argmax(prediction).item()
            predicted_char = emnist_label_to_char(predicted_label)

        result = tk.Toplevel(window)
        result.title("Result")
        label = tk.Label(result, text=f"Predicted character: {predicted_char}", font=("Arial", 24))
        label.pack()

    Canvas.bind("<B1-Motion>", draw_digit)

    clear_button = tk.Button(window, text="Clear", font=('Arial', 18), command=clear)
    clear_button.pack()

    predict_button = tk.Button(window, text="Predict", font=('Arial', 18), command=predict_the_charact)
    predict_button.pack()

    window.mainloop()

##########################################
# PART III: Run  

if __name__ == "__main__":
    Model_Charact_AI()
    Writing_Test()

