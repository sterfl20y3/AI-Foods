import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

from tensorflow.keras.applications.efficientnet import preprocess_input

MODEL_FILE = 'vegetable_classifier_EfficientNetB0.h5'
IMG_SIZE = (224, 224)
EXAMPLES_DIR = 'examples'

CLASS_NAMES = [
    'Bean', 'Bitter Gourd', 'Bottle Gourd', 'Brinjal', 'Broccoli', 'Cabbage',
    'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato',
    'Pumpkin', 'Radish', 'Tomato'
]

TRANSLATION_DICT = {
    'Bean': 'Квасоля',
    'Bitter Gourd': 'Гіркий гарбуз',
    'Bottle Gourd': 'Гарбуз-пляшка',
    'Brinjal': 'Баклажан',
    'Broccoli': 'Броколі',
    'Cabbage': 'Капуста',
    'Capsicum': 'Капсікум',
    'Carrot': 'Морква',
    'Cauliflower': 'Цвітна капуста',
    'Cucumber': 'Огірок',
    'Papaya': 'Папая',
    'Potato': 'Картопля',
    'Pumpkin': 'Гарбуз',
    'Radish': 'Редиска',
    'Tomato': 'Помідор'
}

if not os.path.exists(MODEL_FILE):
    st.error(f"ПОМИЛКА: Файл моделі '{MODEL_FILE}' не знайдено.")
    st.stop()
if not os.path.exists(EXAMPLES_DIR):
    st.warning(f"Папку '{EXAMPLES_DIR}' не знайдено. Приклади не працюватимуть.")

@st.cache_resource
def load_our_model():
    try:
        model = tf.keras.models.load_model(MODEL_FILE)
        return model
    except Exception as e:
        st.error(f"Помилка завантаження моделі: {e}")
        st.stop()

model = load_our_model()

def predict(image_to_predict):
    try:
        image = image_to_predict.resize(IMG_SIZE)
        img_array = np.asarray(image)
        if img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
        img_array = np.expand_dims(img_array, axis=0)
        processed_img = preprocess_input(img_array)
        prediction = model.predict(processed_img)
        return prediction[0]
    except Exception as e:
        st.error(f"Помилка під час прогнозування: {e}")
        return None

def get_top_predictions(prediction, top_k=3):
    top_indices = np.argsort(prediction)[-top_k:][::-1]
    top_preds = [
        (CLASS_NAMES[i], prediction[i] * 100) for i in top_indices
    ]
    return top_preds

st.set_page_config(page_title="Лабораторія AI", layout="wide")

if 'image_to_show' not in st.session_state:
    st.session_state.image_to_show = None

col1, col2 = st.columns(2)

with col1:
    st.title("Лабораторія AI")
    st.subheader("Завантажте дані для аналізу")

    uploaded_file = st.file_uploader(
        "Перетягніть фото сюди або натисніть для завантаження",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.session_state.image_to_show = Image.open(uploaded_file)

    st.write("---")

    st.subheader("Або спробуйте приклад")

    example_buttons_col = st.columns(3)

    with example_buttons_col[0]:
        try:
            if st.button("Морква"):
                st.session_state.image_to_show = Image.open(os.path.join(EXAMPLES_DIR, 'carrot.jpg'))
        except FileNotFoundError:
            st.error("Файл 'carrot.jpg' не знайдено")

    with example_buttons_col[1]:
        try:
            if st.button("Броколі"):
                st.session_state.image_to_show = Image.open(os.path.join(EXAMPLES_DIR, 'broccoli.jpg'))
        except FileNotFoundError:
            st.error("Файл 'broccoli.jpg' не знайдено")

    with example_buttons_col[2]:
        try:
            if st.button("Помідор"):
                st.session_state.image_to_show = Image.open(os.path.join(EXAMPLES_DIR, 'tomato.jpg'))
        except FileNotFoundError:
            st.error("Файл 'tomato.jpg' не знайдено")

with col2:
    st.title("Результат Аналізу")

    if st.session_state.image_to_show is not None:
        st.image(st.session_state.image_to_show, caption="Зображення для аналізу", width=400)
        st.write("---")

        with st.spinner("Аналізую зображення..."):
            prediction = predict(st.session_state.image_to_show)

        if prediction is not None:
            top_predictions = get_top_predictions(prediction, top_k=3)
            main_class, main_confidence = top_predictions[0]
            translated_class = TRANSLATION_DICT.get(main_class, main_class)
            st.subheader(f"Це... {translated_class}!")

            st.progress(int(main_confidence))
            st.caption(f"Впевненість: {main_confidence:.2f} %")

            st.subheader("Топ-3 припущення:")
            for class_name, confidence in top_predictions:
                translated_name = TRANSLATION_DICT.get(class_name, class_name)
                st.text(f"{translated_name:<15} ({confidence:>6.2f} %)")

    else:
        st.info("Результат з'явиться тут...")
        st.image("https://static.streamlit.io/examples/analytics.jpg", caption="Очікування даних")