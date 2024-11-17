import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import gdown

# Define the CustomCNN class (should match the model architecture used during training)
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 26)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.batch_norm = nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 256 * 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

file_id = "1xjKdLtxC-GcItFJs0GwCeFy94c65cs0o"
destination = "custom_cnn_model.pth"

# Construct the Google Drive URL
gdrive_url = f"https://drive.google.com/uc?id={file_id}"

# Download the file
gdown.download(gdrive_url, destination, quiet=False)

# Load the model
@st.cache_resource(allow_output_mutation=True)
def load_model():
    model = CustomCNN()
    model.load_state_dict(torch.load(destination, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Define the transforms for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Streamlit UI
st.title("Image Classification with Custom CNN")
st.write("Upload an image to classify")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    # Map class index to labels (replace with your dataset's labels)
    class_labels = [chr(i + 65) for i in range(26)]  # Example: A-Z for 26 classes
    predicted_label = class_labels[predicted.item()]

    st.write(f"Prediction: **{predicted_label}**")
