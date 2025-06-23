import streamlit as st
import numpy as np
import pickle

# ==== Load model ====
with open("model_lgbm.pkl", "rb") as f:
    model = pickle.load(f)

# ==== Judul aplikasi ====
st.title("Prediksi Kualitas Udara")

# ==== Form input ====
with st.form("form_input"):
    Temperature = st.number_input("Suhu Udara (°C)", value=30.0)
    Humidity = st.number_input("Kelembaban Udara (%)", value=70.0)
    PM25 = st.number_input("PM2.5 (µg/m³)", value=150.0)
    PM10 = st.number_input("PM10 (µg/m³)", value=200.0)
    NO2 = st.number_input("Nitrogen Dioksida (NO2 µg/m³)", value=100.0)
    SO2 = st.number_input("Sulfur Dioksida (SO2 µg/m³)", value=80.0)
    CO = st.number_input("Karbon Monoksida (CO ppm)", value=2.0)
    Proximity = st.number_input("Jarak ke Kawasan Industri (km)", value=1.0)
    Population = st.number_input("Kepadatan Penduduk (jiwa/km²)", value=10000.0)

    submit = st.form_submit_button("Prediksi Kualitas Udara")

# ==== Proses prediksi ====
if submit:
    input_data = np.array([[Temperature, Humidity, PM25, PM10, NO2, SO2, CO, Proximity, Population]])
    
    # Debug: tampilkan input dan output
    st.write("Input ke model:", input_data)
    
    pred = model.predict(input_data)[0]
    st.write("Kode prediksi:", pred)
    st.write("Kelas yang dikenali model:", model.classes_)

    label_dict = {
        0: "Baik",
        1: "Sedang",
        2: "Tidak Sehat",
        3: "Sangat Tidak Sehat"
    }

    hasil = label_dict.get(pred, "Tidak Diketahui")
    st.success(f"Hasil Prediksi: {hasil} (Kode: {pred})")
