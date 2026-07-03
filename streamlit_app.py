import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Building Damage Classification",
    page_icon="🏚️",
    layout="wide"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# MODEL
# --------------------------------------------------

class DualEfficientNet(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        self.feature_extractor = backbone.features

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(

            nn.Linear(2560,512),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(512,4)

        )

    def forward(self, pre_img, post_img):

        pre = self.feature_extractor(pre_img)
        pre = self.pool(pre)
        pre = torch.flatten(pre,1)

        post = self.feature_extractor(post_img)
        post = self.pool(post)
        post = torch.flatten(post,1)

        x = torch.cat([pre,post],dim=1)

        return self.classifier(x)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "dual_efficientnet_final.pth"

@st.cache_resource
def load_model():

    model = DualEfficientNet()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.to(device)

    model.eval()

    return model


model = load_model()

# --------------------------------------------------
# IMAGE TRANSFORM
# --------------------------------------------------

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor()

])

classes = [

    "No Damage",

    "Minor Damage",

    "Major Damage",

    "Destroyed"

]

# --------------------------------------------------
# MAIN TITLE
# --------------------------------------------------

st.markdown("""
# 🛰️ AI Building Damage Classification

### Dual EfficientNet-B0 Deep Learning Model

Upload **Pre-Disaster** and **Post-Disaster** building images to classify the structural damage level.
""")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("📌 Project Information")

st.sidebar.markdown("""
### Model
Dual EfficientNet-B0

---

### Dataset

- xBD Dataset
- 16,956 Building Samples
- 4 Damage Classes

---

### Model Performance

✅ Accuracy : **78.1%**

✅ Precision : **79%**

✅ Recall : **78%**

---

### Damage Classes

🟢 No Damage

🟡 Minor Damage

🟠 Major Damage

🔴 Destroyed

---

Developed by

**Somendar Kumar Das**

MSc Data Science
""")

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

left, right = st.columns(2)

with left:

    st.subheader("📷 Pre-Disaster Image")

    pre_file = st.file_uploader(
        "Upload PRE Image",
        type=["png","jpg","jpeg"]
    )

with right:

    st.subheader("🔥 Post-Disaster Image")

    post_file = st.file_uploader(
        "Upload POST Image",
        type=["png","jpg","jpeg"]
    )

# --------------------------------------------------
# SHOW IMAGES
# --------------------------------------------------

if pre_file is not None and post_file is not None:

    pre_image = Image.open(pre_file).convert("RGB")
    post_image = Image.open(post_file).convert("RGB")

    img_left, img_right = st.columns(2)

    with img_left:

        st.image(
            pre_image,
            caption="Pre Disaster Image",
            use_container_width=True
        )

    with img_right:

        st.image(
            post_image,
            caption="Post Disaster Image",
            use_container_width=True
        )

    st.markdown("---")

    predict = st.button(
        "🔍 Predict Damage",
        use_container_width=True
    )
else:

    predict = False


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if predict:

    pre_tensor = transform(pre_image).unsqueeze(0).to(device)
    post_tensor = transform(post_image).unsqueeze(0).to(device)

    with st.spinner("Analyzing Building Damage..."):

        with torch.no_grad():

            outputs = model(pre_tensor, post_tensor)

            probabilities = torch.softmax(outputs, dim=1)[0]

            prediction = torch.argmax(probabilities).item()

            confidence = probabilities[prediction].item() * 100

    st.markdown("---")

    colors = {
        "No Damage":"🟢",
        "Minor Damage":"🟡",
        "Major Damage":"🟠",
        "Destroyed":"🔴"
    }

    st.success(
        f"{colors[classes[prediction]]} Prediction: {classes[prediction]}"
    )

    st.metric(
        label="Model Confidence",
        value=f"{confidence:.2f}%"
    )

    st.markdown("---")

    st.subheader("Prediction Probabilities")

    for i in range(len(classes)):

        st.write(f"### {classes[i]}")

        st.progress(float(probabilities[i].cpu()))

        st.write(
            f"{probabilities[i].item()*100:.2f}%"
        )

    st.markdown("---")

    st.subheader("Model Summary")

    c1,c2,c3 = st.columns(3)

    with c1:

        st.metric(
            "Architecture",
            "Dual EfficientNet-B0"
        )

    with c2:

        st.metric(
            "Validation Accuracy",
            "78.1%"
        )

    with c3:

        st.metric(
            "Classes",
            "4"
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
"""
AI Building Damage Classification using Dual EfficientNet-B0

University of Europe for Applied Sciences

MSc Data Science

Developed by Somendar , Prudhvi , Sunil
"""
) 