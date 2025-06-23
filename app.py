import streamlit as st
import numpy as np
import pickle

# Load model
with open("model_lgbm.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Prediksi Kualitas Udara")

with st.form("form_input"):
    Temperature = st.number_input("Suhu Udara (°C)", min_value=-10.0, step=0.1)
    Humidity = st.number_input("Kelembaban Udara (%)", min_value=0.0, max_value=100.0, step=1.0)
    PM25 = st.number_input("Partikulat Halus (PM2.5 µg/m³)", min_value=0.0, step=0.1)
    PM10 = st.number_input("Partikulat Kasar (PM10 µg/m³)", min_value=0.0, step=0.1)
    NO2 = st.number_input("Nitrogen Dioksida (NO₂ µg/m³)", min_value=0.0, step=0.1)
    SO2 = st.number_input("Sulfur Dioksida (SO₂ µg/m³)", min_value=0.0, step=0.1)
    CO = st.number_input("Karbon Monoksida (CO ppm)", min_value=0.0, step=0.1)
    Proximity = st.number_input("Jarak ke Kawasan Industri (km)", min_value=0.0, step=0.1)
    Population = st.number_input("Kepadatan Penduduk (jiwa per km²)", min_value=0.0, step=1.0)

    submit = st.form_submit_button("Prediksi Kualitas Udara")


if submit:
    input_data = np.array([[Temperature, Humidity, PM25, PM10, NO2, SO2, CO, Proximity, Population]])
    pred = model.predict(input_data)[0]

    # Definisikan label_dict sesuai label model kamu
    label_dict = {
        0: "Baik",
        1: "Sedang",
        2: "Tidak Sehat",
        3: "Sangat Tidak Sehat"
    }

    st.success(f"Hasil Prediksi: {label_dict.get(pred, 'Tidak diketahui')})")
