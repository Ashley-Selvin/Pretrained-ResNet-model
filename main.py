import torch 
from torchvision import models, transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
# from colorama import Fore, Style, init 

# initialise colorama
# init()

# Load pretrained model
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load image 
image = Image.open("leaf.jpg").convert("RGB")
image_resized = image.resize((224,224))
img_tensor = transform(image).unsqueeze(0)

# Run inference 
with torch.no_grad():
    output = model(img_tensor)

# Get predicted class index
pred = output.argmax().item()

# Simple rule - based "health classification" for demo purposes 
if pred % 2 == 0:
    status = "Healthy Crop"
    color = (0, 200, 0)
    confidence = "High"
else:
    status = "Crop Stress Detected"
    color = (200, 0, 0)
    confidence = "Medium"

# Draw text overlay
draw = ImageDraw.Draw(image_resized)

try:
    font = ImageFont.truetype("arial.ttf", 22)
except:
    font = None # fallback
# background box
draw.rectangle([(10, 10), (420, 80)], fill=(0, 0, 0))

# Text lines
draw.text((15, 20), f"Status: {status}", fill=color, font=font)
draw.text((15, 45), f"Confidence: {confidence}", fill=(255, 255, 255), font=font)

# Create heatmap
heatmap = np.random.rand(224, 224)

# Convert image to numpy
img_np = np.array(image_resized)

# Plot final result 
plt.figure(figsize=(8, 8), dpi=150)
plt.imshow(img_np)
plt.imshow(heatmap, cmap='jet', alpha=0.4)

plt.title("AI Crop Monitoring Output", fontsize=16)
plt.axis('off')

plt.tight_layout()
# Save high-quality image 
plt.savefig("output.png", dpi=300)
plt.show()

# Terminal Output 
print("\n----- AI Crop Monitoring System -----\n")
print(f"Prediction Index: {pred}")
print(f"Status: {status}")
print(f"Confidence Level: {confidence}")
print("\n-----------------------------------\n")
