import streamlit as st
from src.preprocessing import preprocess_image
from src.model import build_autoencoder
from src.detect import detect_anomaly, show_anomaly
import matplotlib.pyplot as plt

st.title("Détection d'anomalies thermiques")

uploaded_file = st.file_uploader("Choisissez une image thermique", type=["png", "jpg"])
if uploaded_file:
    img = preprocess_image(uploaded_file)
    model = build_autoencoder(img.shape)
    model.load_weights("model_weights.h5")  # poids préentraînés

    anomaly_map, recon = detect_anomaly(model, img)

    st.subheader("Résultat")
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img, cmap="gray")
    ax[1].imshow(anomaly_map, cmap="hot")
    st.pyplot(fig)
