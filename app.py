import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(page_title="MNIST App", page_icon="🔢")

st.title("🔢 Clasificador de Dígitos")

# Carga del modelo en caché
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('digits_mnist_model.h5')
    except:
        return None

model = load_model()

if model is None:
    st.error("No se encontró 'digits_mnist_model.h5'")
    st.stop()

# Interfaz
uploaded_file = st.file_uploader("Sube una imagen (0-9)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen original", width=150)

    # Preprocesamiento: Escala de grises y resize
    img_gray = ImageOps.grayscale(image)
    img_resized = img_gray.resize((28, 28))

    # Inversión de colores (MNIST usa fondo negro, letras blancas)
    if st.checkbox("Invertir colores (marcar si es fondo blanco)", value=True):
        img_final = ImageOps.invert(img_resized)
    else:
        img_final = img_resized

    # Normalizar y reshape
    img_array = np.array(img_final).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28)

    # Predicción
    if st.button("Predecir"):
        pred = model.predict(img_array)
        clase = np.argmax(pred)
        conf = np.max(pred)

        st.success(f"Predicción: {clase}")
        st.info(f"Confianza: {conf:.2%}")
        st.bar_chart(pred[0])
