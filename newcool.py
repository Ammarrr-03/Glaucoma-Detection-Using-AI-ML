import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import streamlit as st

# Define the model architecture for EfficientNet V2
class GlaucomaDetectionModel(nn.Module):
    def __init__(self):
        super(GlaucomaDetectionModel, self).__init__()
        self.features = models.efficientnet_v2_s(pretrained=False).features  # Use the features directly
        self.classifier = None

    def forward(self, x):
        x = self.features(x)  # Pass through feature extractor
        x = torch.flatten(x, 1)  # Flatten the output
        if self.classifier is None:
            num_ftrs = x.shape[1]
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(num_ftrs, 2)  # Binary classification (2 classes)
            )
        x = self.classifier(x)
        return x

# Function to load the trained model
def load_model(model_path):
    model = GlaucomaDetectionModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    model.eval()
    return model

# Preprocess the image before passing it to the model
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)
    return image

# Perform prediction with the model
def predict_glaucoma(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)

    return predicted.item()

# Streamlit app to interact with the user
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Glaucoma Detector",
        page_icon="👁️",
        layout="wide"
    )

    # Custom CSS for styling
    st.markdown("""
    <style>
    .big-font {
        font-size:25px !important;
        font-weight:bold;
        color:#2C3E50;
    }
    .result-good {
        color:#2ECC71;
        font-weight:bold;
    }
    .result-warning {
        color:#E74C3C;
        font-weight:bold;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create columns for main content and sidebar
    col1, col2 = st.columns([3, 1])

    with col1:
        # Title and description
        st.title("👁️ Glaucoma Detection AI")
        st.markdown("<p class='big-font'>Upload an eye image to detect potential glaucoma</p>", unsafe_allow_html=True)
        
        # Upload the image
        uploaded_file = st.file_uploader("📸 Choose an eye image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Open the image and display it
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Eye Image", use_container_width=True)

            # Load model (ensure this is the path where your model is saved)
            model_path = r"D:\LY SEM 1\StreamLit App\best.pth"  # Replace with actual path
            model = load_model(model_path)

            # Preprocess the image and get prediction
            image_tensor = preprocess_image(image)
            predicted = predict_glaucoma(model, image_tensor)

            # Display the result
            if predicted == 1:
                st.markdown("<h2 class='result-warning'>🚨 Glaucoma Detected</h2>", unsafe_allow_html=True)
                st.markdown("<h2 class='result-warning'>Please consult an ophthalmologist</h2>", unsafe_allow_html=True)
            else:
                st.markdown("<h2 class='result-good'>✅ No Glaucoma Detected</h2>", unsafe_allow_html=True)

    with col2:
        st.markdown("### 🤖 Model Information")
        
        st.markdown("#### Model Used")
        st.write("- EfficientNet V2 Small")
        st.write("- Transfer Learning Architecture")
        
        st.markdown("### 🩺 Important Medical Disclaimer")
        st.warning("""
        ⚠️ This AI model is NOT a substitute for professional medical advice:
        
        - Always consult a qualified eye specialist
        - This is a screening tool, not a definitive diagnosis
        - Professional medical examination is crucial
        - Individual medical conditions vary
        
        🏥 Schedule regular eye check-ups with a doctor
        """)

if __name__ == "__main__":
    main()