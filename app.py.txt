import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# 1. Define Architecture matching your trained model
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

class AttentionMelanomaModel(nn.Module):
    def __init__(self):
        super(AttentionMelanomaModel, self).__init__()
        base_model = models.efficientnet_b0(weights=None)
        self.features = base_model.features
        self.attention = ChannelAttention(in_channels=1280)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 2. Load Model Weights safely
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionMelanomaModel()
    try:
        model.load_state_dict(torch.load('attention_melanoma_best.pth', map_location=device))
        st.success("Loaded trained model weights successfully!")
    except Exception as e:
        st.warning("⚠️ Model weights file 'attention_melanoma_best.pth' not found in folder. Running in UI preview mode with randomized weights.")
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# 3. Streamlit UI Design
st.title("🔬 Melanoma Skin Lesion Detection")
st.write("Upload a dermoscopic skin lesion image below to analyze malignancy risk using our custom attention-enhanced deep learning model.")

uploaded_file = st.file_uploader("Choose an image (.jpg, .png, .jpeg)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Lesion")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Analysis Result")
        with st.spinner("Analyzing clinical features..."):
            # Preprocess image
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                output = model(img_tensor).squeeze(1)
                prob = torch.sigmoid(output).item()
            
            # Display metrics
            st.metric(label="Melanoma Risk Probability", value=f"{prob * 100:.2f}%")
            
            if prob > 0.5:
                st.error("⚠️ **High Risk / Suspicious for Melanoma**: Recommended to consult a dermatologist for clinical evaluation.")
            else:
                st.success("✅ **Low Risk / Benign Appearance**: No acute malignant indicators strongly detected.")