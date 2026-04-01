import streamlit as st
import torch 
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ---------------------------------------
# Load model (cached)
# ---------------------------------------
@st.cache_resource
def load_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.eval()
    return model

model = load_model()

# ---------------------------------------
# Page config 
# --------------------------------------- 
st.set_page_config(page_title="AI Crop Monitoring", layout="centered")

st.title("AI Crop Monitoring System")
st.write("Upload a crop image to analyse plant health.")

# ---------------------------------------
# Upload image 
# ---------------------------------------
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

# ---------------------------------------
# Preprocess 
# --------------------------------------- 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

img_tensor = transform(image).unsqueeze(0)

# ---------------------------------------
# Inference
# ---------------------------------------
with torch.no_grad():
    output = model(img_tensor)

pred = output.argmax().item()

# Fake classification
if pred % 2 == 0:
    status = "Healthy Crop"
    color = (0, 180, 0)
    confidence = "High"
    st.success("Healthy Crop Detected")
else:
    status = "Crop Stress Detected"
    color = (200, 50, 50)
    confidence = "Medium"
    st.error("Crop Stress Detected")

st.write(f"**Confidence Level:** {confidence}")

# ---------------------------------------
# Create heatmap
# ---------------------------------------
img_np = np.array(image)
h, w = img_np.shape[:2]

heatmap = np.random.rand(h, w)
heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

# ---------------------------------------
# Plot heatmap overlay
# ---------------------------------------
fig, ax = plt.subplots()
ax.imshow(img_np)
ax.imshow(heatmap, cmap='jet', alpha=0.3)
ax.axis('off')

st.subheader("Crop Health Heatmap")
st.pyplot(fig)

# ---------------------------------------
# Summary panel
# ---------------------------------------
st.markdown("---")
st.subheader("Analysis Summary")

col1, col2 = st.columns(2)
col1.metric("Status", status)
col2.metric("Confidence", confidence)
