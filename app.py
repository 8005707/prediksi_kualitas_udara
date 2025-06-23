import streamlit as st
import numpy as np
import pickle

# Load model LightGBM
@st.cache_resource
def load_model():
    with open("model_lgbm.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Judul Aplikasi
st.title("Prediksi Kualitas Udara")
st.write("Masukkan data sensor untuk prediksi kualitas udara.")

# Form input manual
with st.form("form_input"):
    CO = st.number_input("CO (ppm)")
NO2 = st.number_input("NO2")
SO2 = st.number_input("SO2")
O3 = st.number_input("O3")
PM10 = st.number_input("PM10")
PM25 = st.number_input("PM2.5")
Temperature = st.number_input("Temperature (°C)")
Humidity = st.number_input("Humidity (%)")

submit = st.form_submit_button("Prediksi")

if submit:
    input_data = np.array([[CO, NO2, SO2, O3, PM10, PM25, Temperature, Humidity]])
pred = model.predict(input_data)[0]


label_dict = {
        0: "Baik",
        1: "Sedang",
        2: "Tidak Sehat",
        3: "Sangat Tidak Sehat",
        4: "Berbahaya"
    }

st.success(f"Hasil Prediksi: {label_dict.get(pred, 'Tidak diketahui')} (Kode: {pred})")
