import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os

# Dicionário dos modelos disponíveis
MODEL_PATHS = {
    "Model S (loss: Categorical crossentropy, optimizer: RMSprop)": "../modelS_CatCross_RMS.keras",
    "Model S (loss: Categorical crossentropy, optimizer: RMSprop, regularization: dropout)": "../modelS_CatCross_RMS_dropout.keras",
    "Model S (loss: KLDivergence, optimizer: SGD)" : '../modelS_KLD_SGD.keras',
    "Model S (loss: KLDivergence, optimizer: RMSprop)" : '../modelS_KLD_RMS.keras',
    "Model S (loss: Categorical crossentropy, optimizer: RMSprop, dropout)" : '../modelS_CatCross_RMS_dropout.keras',
    "Model S (loss: Categorical crossentropy, optimizer: RMSprop, dropout and L2)": '../modelS_CatCross_RMS_dropout_L2.keras',
    "Model S (loss: Categorical crossentropy, optimizer: RMSprop, L2)": '../modelS_CatCross_RMS_L2.keras',
    "Model T (loss: Categorical crossentropy, optimizer: RMSprop, feature extraction)": '../modelT_featureExtraction.keras',
    "Model T (loss: Categorical crossentropy, optimizer: RMSprop, feature extraction, data augmentation)": '../modelT_featureExtraction_DataAugmentation.keras',
    "Model T (loss: Categorical crossentropy, optimizer: RMSprop, fine tuning)": '../modelT_fineTuning.keras',
    "Model T (loss: Categorical crossentropy, optimizer: RMSprop, fine tuning, data augmentation)": '../modelT_fineTuning_DataAugmentation.keras',
}

CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Função para carregar modelo
@st.cache_resource
def load_selected_model(model_path):
    return load_model(model_path)

# Função para obter o tamanho esperado do input do modelo
def get_target_size(model):
    input_shape = model.input_shape  # e.g., (None, 150, 150, 3)
    return input_shape[1:3]  # (height, width)

# Função para preparar imagem
def preprocess_image(image, target_size):
    image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Interface Streamlit
st.set_page_config(page_title="Classificação de Imagens", layout="centered")
st.title("Classificação de paisagens com CNNs")

# Escolher modelo
selected_model_name = st.selectbox("Modelo a utilizar:", list(MODEL_PATHS.keys()))
model = load_selected_model(MODEL_PATHS[selected_model_name])

# Carregamento da imagem
uploaded_file = st.file_uploader("Imagem (JPG ou PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem carregada", width=300)

    if st.button("Classificar imagem"):
        with st.spinner("A processar e classificar..."):
            target_size = get_target_size(model)
            processed = preprocess_image(image, target_size=target_size)
            preds = model.predict(processed)[0]
            pred_class = CLASS_NAMES[np.argmax(preds)]
            confidence = np.max(preds) * 100

        st.success(f"Classe prevista: **{pred_class}** com confiança de {confidence:.2f}%")

        st.subheader("📊 Distribuição de probabilidades:")
        for i, prob in enumerate(preds):
            st.write(f"- {CLASS_NAMES[i]}: {prob:.4f}")
