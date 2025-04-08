import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import time

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
    # Load pre-trained ResNet18 using the updated weights argument
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)  # 2 classes: Cataract, Normal
    # Load model weights from file
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
    # This function expects a preprocessed image tensor with shape [1, 3, H, W]
    model.eval()
    features = []
    gradients = []

    # Define hooks to capture features and gradients from the last convolutional layer (layer4)
    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    # Register hooks on model.layer4
    handle_fw = model.layer4.register_forward_hook(forward_hook)
    handle_bw = model.layer4.register_backward_hook(backward_hook)
    
    # Forward pass
    output = model(image_tensor)
    pred_class = output.argmax(dim=1).item()
    
    # Backward pass
    model.zero_grad()
    class_loss = output[0, pred_class]
    class_loss.backward()
    
    # Now, get the gradients and features from the hooks
    grads = gradients[0].squeeze(0)        # shape: [C, H, W]
    fmap = features[0].squeeze(0)            # shape: [C, H, W]
    
    # Global average pooling of gradients: compute weights
    weights = torch.mean(grads, dim=(1, 2))   # shape: [C]
    
    # Compute weighted combination of feature maps
    cam = torch.zeros(fmap.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * fmap[i, :, :]
    
    # Apply ReLU and normalize the CAM
    cam = F.relu(cam)
    cam = cam - cam.min()
    if cam.max() != 0:
        cam = cam / cam.max()
    
    # Detach and convert to numpy array
    cam = cam.detach().cpu().numpy()
    
    # Resize the CAM to the input image size (224,224)
    import cv2
    cam = cv2.resize(cam, (224, 224))
    
    # Clean up hooks
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
        st.image(image, caption="Your uploaded image", use_container_width=True)
    
    with st.spinner("Analyzing image..."):
        time.sleep(1)  # Optional delay for spinner effect
        label, confidence, image_tensor = predict(image)
    
    with col2:
        st.subheader("Prediction Results")
        st.write(f"**Predicted Class:** {label}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")
        
        # Progress Bar Animation for Confidence
        st.subheader("Confidence Progress")
        progress = st.progress(0)
        conf_percent = int(confidence * 100)
        for i in range(conf_percent + 1):
            progress.progress(i)
            time.sleep(0.005)
        
        # Grad-CAM Visualization
        st.subheader("Grad-CAM Heatmap")
        cam = generate_gradcam(image_tensor, model)
        # Create a heatmap overlay on the uploaded image
        image_np = np.array(image.resize((224, 224)))
        heatmap = plt.get_cmap('jet')(cam / 255.0)
        # Remove alpha channel and scale heatmap
        heatmap = np.delete(heatmap, 3, axis=2)
        overlay = heatmap * 0.4 + image_np / 255.0

        # Display the overlay
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(overlay)
        ax.axis('off')
        st.pyplot(fig)
else:
    st.info("Please upload an image to begin.")

# Finish progress bar
progress_bar.progress(100)
st.success("Analysis complete!")

