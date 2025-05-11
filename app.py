import streamlit as st
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import time
import os  # Added for path manipulation

# Import your model architectures here
from models import ANN, CNN, YOLOv7Classifier, create_vit_model

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names (replace with your actual class list)
class_names = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___late_blight',
    'Tomato_Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato__Spider_mites',
    'Tomato_Target_spot',
    'Tomato_Curl_Virus',
    'Tomato_Mosaic_virus',
    'Tomato_Healthy'
]

# Enhanced image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Initial resize
    transforms.CenterCrop(224),  # Final crop size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model(model_name):
    model_map = {
        "ANN": {
            "class": ANN,
            "args": {"input_size": 224 * 224 * 3, "num_classes": len(class_names)},
            "file": "best_ann_model.pth"
        },
        "CNN": {
            "class": CNN,
            "args": {"num_classes": len(class_names)},
            "file": "best_cnn_model.pth"
        },
        "YOLOv7": {
            "class": YOLOv7Classifier,
            "args": {"num_classes": len(class_names)},
            "file": "best_yolov7_model.pth"
        },
        "ViT": {
            "class": create_vit_model,
            "args": {"num_classes": len(class_names)},
            "file": "best_vit_model.pth"
        }
    }

    try:
        config = model_map.get(model_name)
        if not config:
            raise ValueError(f"Model {model_name} not found in configuration")

        # Instantiate model
        model = config["class"](**config["args"]) if model_name != "ViT" else config["class"](**config["args"])

        # Load weights with progress
        with st.spinner(f'Loading {model_name} weights...'):
            model.load_state_dict(torch.load(config["file"], map_location=device))
            model = model.to(device)
            model.eval()

        return model

    except Exception as e:
        st.error(f"❌ Failed to load {model_name}: {str(e)}")
        st.stop()


def predict(image, model):
    start_time = time.time()
    try:
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        inference_time = time.time() - start_time
        return outputs.cpu().numpy(), probabilities.cpu().numpy(), inference_time
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None, None, 0


# Streamlit UI
st.set_page_config(
    page_title="Plant Disease Classifier",
    layout="wide",
    page_icon="🌿"
)

st.title("🌿 Plant Disease Classifier")
st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Model selection
model_name = st.selectbox(
    "Select Model Architecture",
    ["ANN", "CNN", "YOLOv7", "ViT"],
    help="Choose which deep learning model to use for classification"
)

# Sidebar
with st.sidebar:
    st.header("🧠 Model Information")

    with st.expander("ℹ️ Architecture Details", expanded=True):
        model_details = {
            "ANN": "• 3 Fully Connected Layers\n• Input size: 224×224×3\n• Best for simple classification tasks",
            "CNN": "• 4 Convolutional Layers\n• Batch normalization\n• Optimized for image features",
            "YOLOv7": "• YOLO-inspired architecture\n• Specialized for object detection\n• Adapted for classification",
            "ViT": "• Vision Transformer\n• Pretrained on ImageNet\n• State-of-the-art performance"
        }
        st.markdown(model_details[model_name])

    with st.expander("⚙️ Performance Metrics"):
        st.write("Coming soon: Actual performance metrics")
        st.progress(75 if model_name == "ViT" else
                    60 if model_name == "YOLOv7" else
                    40 if model_name == "CNN" else 30)

# Load model
try:
    model = load_model(model_name)
    st.success(f"✅ {model_name} model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# File uploader with case handling
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file:
    try:
        # Convert extension to lowercase while preserving filename
        filename = uploaded_file.name
        base, ext = os.path.splitext(filename)
        corrected_filename = base + ext.lower()

        # Process image
        image = Image.open(uploaded_file).convert('RGB')

        # Display in columns
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Uploaded Image")
            st.image(image, caption=corrected_filename, use_column_width=True)
            st.caption(f"Dimensions: {image.size[0]}×{image.size[1]} pixels")

        with col2:
            st.subheader("Analysis Results")

            with st.spinner('Analyzing image...'):
                outputs, probs, inference_time = predict(image, model)

                if probs is not None:
                    top5_prob, top5_class = torch.topk(torch.from_numpy(probs), 5, dim=1)

                    # Display predictions
                    with st.expander("🔍 Detailed Predictions", expanded=True):
                        st.write(f"**Model:** {model_name}")
                        st.write(f"**Inference Time:** {inference_time:.2f} seconds")

                        for i in range(5):
                            class_name = class_names[top5_class[0][i]]
                            prob = top5_prob[0][i].item()

                            st.write(f"**{i + 1}. {class_name}**")
                            st.progress(prob)
                            st.write(f"{prob:.2%} confidence")
                            st.write("---")

                    # Visualization
                    st.subheader("Confidence Distribution")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.barh(
                        [class_names[i] for i in top5_class[0].numpy()[::-1]],
                        top5_prob[0].numpy()[::-1],
                        color='#4CAF50'
                    )
                    ax.set_xlim(0, 1)
                    ax.set_xlabel("Confidence Score")
                    ax.set_title("Top 5 Predictions")
                    st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray;">
    Plant Disease Classifier • Built with Streamlit and PyTorch
    </div>
""", unsafe_allow_html=True)