import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import cv2

# -----------------------------
# Set page configuration
st.set_page_config(page_title="Cataract Detection App", layout="wide")

# -----------------------------
# App Title and Description
st.title("🩺 Cataract Detection Dashboard")
st.markdown("""
Upload an eye image to check if it is **Cataract** or **Normal**.
The dashboard displays the model prediction, confidence score, and a Grad-CAM heatmap for explainability.
""")

# -----------------------------
# Progress bar placeholder
progress_bar = st.progress(0)

# -----------------------------
# Load Model
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)  # 2 classes: Cataract, Normal
    model.load_state_dict(torch.load("cataract_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model()
progress_bar.progress(20)

# -----------------------------
# Define class names and image transforms
class_names = ['Cataract', 'Normal']
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
progress_bar.progress(40)

# -----------------------------
# Prediction function
def predict(image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    outputs = model(image_tensor)
    probabilities = F.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)
    return class_names[predicted.item()], confidence.item(), image_tensor

# -----------------------------
# Grad-CAM function
def generate_gradcam(image_tensor, model):
    model.eval()
    features = []
    gradients = []

    # Define hooks to capture features and gradients from the last conv layer
    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_fw = model.layer4.register_forward_hook(forward_hook)
    handle_bw = model.layer4.register_backward_hook(backward_hook)

    output = model(image_tensor)
    pred_class = output.argmax(dim=1).item()

    model.zero_grad()
    class_loss = output[0, pred_class]
    class_loss.backward()

    grads = gradients[0].squeeze(0)  # [C, H, W]
    fmap = features[0].squeeze(0)   # [C, H, W]

    weights = torch.mean(grads, dim=(1, 2))  # [C]

    cam = torch.zeros(fmap.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * fmap[i, :, :]

    cam = F.relu(cam)
    cam = cam - cam.min()
    if cam.max() != 0:
        cam = cam / cam.max()

    cam = cam.detach().cpu().numpy()
    cam = cv2.resize(cam, (224, 224))

    handle_fw.remove()
    handle_bw.remove()

    return cam

progress_bar.progress(60)

# -----------------------------
# Streamlit App: Sidebar for file upload
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
progress_bar.progress(80)

# -----------------------------
# Main App Logic
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    # Layout: Two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, caption="Original Image", use_column_width=True)

    # Run prediction and Grad-CAM
    pred_class, confidence, image_tensor = predict(image)
    gradcam = generate_gradcam(image_tensor, model)

    # Convert Grad-CAM to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    image_np = np.array(image.resize((224, 224))) / 255.0
    overlayed_image = heatmap + image_np
    overlayed_image = overlayed_image / np.max(overlayed_image)

    with col2:
        st.subheader("Model Prediction")
        st.markdown(f"**Class:** {pred_class}")
        st.markdown(f"**Confidence:** {confidence * 100:.2f}%")

        st.subheader("Grad-CAM Heatmap")
        st.image(overlayed_image, caption="Grad-CAM Heatmap", use_column_width=True)

    progress_bar.progress(100)

else:
    st.warning("👈 Please upload an image from the sidebar to get started!")

# -----------------------------
# Footer
st.markdown("---")
st.markdown("Made with ❤️ for Cataract Detection")
