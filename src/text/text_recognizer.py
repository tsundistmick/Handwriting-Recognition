import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
import torch.nn.functional as F
import math
from collections import defaultdict

ALPHABET = (
    " !\"#&'()*+,-./0123456789:;?"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)

BLANK_IDX = 0
CHAR2IDX  = {ch: i + 1 for i, ch in enumerate(ALPHABET)}
IDX2CHAR  = {i + 1: ch for i, ch in enumerate(ALPHABET)}
NUM_CLASSES = len(ALPHABET) + 1


class Codec:
    @staticmethod
    def decode_greedy(log_probs: torch.Tensor) -> str:
        indices = log_probs.argmax(dim=-1).tolist()
        chars, prev = [], None
        for idx in indices:
            if idx != prev:
                if idx != BLANK_IDX:
                    chars.append(IDX2CHAR.get(idx, ""))
            prev = idx
        return "".join(chars)


class CRNNModel(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, (4, 1)), nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(
            input_size=512, hidden_size=256,
            num_layers=2, bidirectional=True,
            batch_first=True, dropout=0.3,
        )
        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = x.permute(1, 0, 2)
        return x.log_softmax(dim=2)

@st.cache_resource
def load_text_model():
    model = CRNNModel(NUM_CLASSES)
    model_path = Path("best_text_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False))
    model.eval()
    return model

def predict_text(model, image_array):
    if image_array.ndim == 2:
        image_array = image_array[np.newaxis, ...]
    if image_array.ndim == 3 and image_array.shape[0] != 1:
        image_array = image_array[np.newaxis, ...]
    
    tensor = torch.tensor(image_array, dtype=torch.float32)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    
    with torch.no_grad():
        log_probs = model(tensor)
        pred_text = Codec.decode_greedy(log_probs.squeeze(1).cpu())
    
    return pred_text

def preprocess_uploaded_image(uploaded_file, img_height=64):
    img = Image.open(uploaded_file).convert('L')
    w, h = img.size
    new_w = int(w * img_height / h)
    img = img.resize((new_w, img_height), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = img_array[np.newaxis, ...]
    return img_array


def preprocess_canvas_image(canvas_result, img_height=64):
    if canvas_result is None or canvas_result.image_data is None:
        return None
    img_array = canvas_result.image_data.astype(np.uint8)
    img_gray = np.mean(img_array, axis=2)
    from skimage.transform import resize
    h, w = img_gray.shape
    new_w = int(w * img_height / h)
    img_resized = resize(img_gray, (img_height, new_w), anti_aliasing=True, preserve_range=True).astype(np.float32)
    img_resized = img_resized / 255.0
    img_resized = img_resized[np.newaxis, ...]
    return img_resized

def main():
    st.set_page_config(page_title="Распознавание рукописного текста", layout="centered")
    st.title("Распознавание рукописного текста")
    st.markdown("Напишите слово или фразу, или загрузите картинку (строка текста)")

    with st.spinner(""):
        model = load_text_model()
    st.success("")

    tab1, tab2 = st.tabs(["✍️ Написать текст", "📁 Загрузить картинку"])

    with tab1:
        st.subheader("Напишите текст")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=5,
            stroke_color="black",
            background_color="white",
            height=200,
            width=1000,
            drawing_mode="freedraw",
            key="canvas_text",
        )
        if st.button("🔍 Распознать", key="predict_canvas_text"):
            img_array = preprocess_canvas_image(canvas_result, img_height=64)
            if img_array is not None:
                text = predict_text(model, img_array)
                st.success(f"### Распознано: **{text}**")
            else:
                st.warning("Сначала напишите текст!")

    with tab2:
        st.subheader("Загрузите картинку")
        uploaded_file = st.file_uploader("Выберите изображение", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption="Загруженное изображение", width=300)
            if st.button("🔍 Распознать", key="predict_upload_text"):
                img_array = preprocess_uploaded_image(uploaded_file, img_height=64)
                text = predict_text(model, img_array)
                st.success(f"### Распознано: **{text}**")


if __name__ == "__main__":
    main()