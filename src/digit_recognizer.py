import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.25),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=3136, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x

@st.cache_resource
def load_model():
    model = MNISTModel()
    model_path = Path("best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_digit(model, image_array):
    tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        digit = torch.argmax(probabilities, dim=1).item()
    
    return digit, probabilities[0].cpu().numpy()

def preprocess_uploaded_image(uploaded_file):
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    if np.mean(img_array) < 0.5:
        img_array = 1.0 - img_array
    
    return img_array

def preprocess_canvas_image(canvas_result):
    if canvas_result is None or canvas_result.image_data is None:
        return None
    img_array = canvas_result.image_data.astype(np.uint8)
    img_gray = np.mean(img_array, axis=2)
    img_gray = 255 - img_gray
    from skimage.transform import resize
    img_resized = resize(img_gray, (28, 28), anti_aliasing=True)
    img_resized = img_resized / 255.0    
    return img_resized.astype(np.float32)

def main():
    st.set_page_config(page_title="Распознавание цифр", layout="centered")
    st.title("Распознавание рукописных цифр")
    st.markdown("Нарисуйте цифру мышкой или загрузите картинку")
    with st.spinner("Загрузка модели..."):
        model = load_model()
    tab1, tab2 = st.tabs(["🎨 Нарисовать цифру", "📁 Загрузить картинку"])
    with tab1:
        st.subheader("Напишите цифру мышкой")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=20,
            stroke_color="black",
            background_color="white",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        if st.button("🔍 Распознать", key="predict_canvas"):
            img_array = preprocess_canvas_image(canvas_result)
            if img_array is not None:
                digit, probs = predict_digit(model, img_array)
                st.success(f"### Результат: **{digit}**")
                st.write("#### Уверенность модели:")
                top3_indices = np.argsort(probs)[-3:][::-1]
                for idx in top3_indices:
                    st.write(f"- Цифра **{idx}**: {probs[idx]*100:.1f}%")
            else:
                st.warning("Сначала нарисуйте цифру!")
    with tab2:
        st.subheader("Загрузите картинку с цифрой")
        uploaded_file = st.file_uploader("Выберите изображение", 
                                          type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption="Загруженное изображение", width=200)
            if st.button("🔍 Распознать", key="predict_upload"):
                img_array = preprocess_uploaded_image(uploaded_file)
                digit, probs = predict_digit(model, img_array)
                st.success(f"### Результат: **{digit}**")
                st.write("#### Уверенность модели:")
                top3_indices = np.argsort(probs)[-3:][::-1]
                for idx in top3_indices:
                    st.write(f"- Цифра **{idx}**: {probs[idx]*100:.1f}%")
if __name__ == "__main__":
    main()