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
# Form input manual
with st.form("form_input"):
    Temperature = st.number_input("Temperature (°C)", min_value=-10.0, step=0.1)
    Humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=1.0)
    PM25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, step=0.1)
    PM10 = st.number_input("PM10 (µg/m³)", min_value=0.0, step=0.1)
    NO2 = st.number_input("NO2 (µg/m³)", min_value=0.0, step=0.1)
    SO2 = st.number_input("SO2 (µg/m³)", min_value=0.0, step=0.1)
    CO = st.number_input("CO (ppm)", min_value=0.0, step=0.1)
    Proximity = st.number_input("Jarak ke Kawasan Industri (km)", min_value=0.0, step=0.1)
    Population = st.number_input("Kepadatan Penduduk (jiwa/km²)", min_value=0.0, step=1.0)

    submit = st.form_submit_button("Prediksi")



if submit:
    input_data = np.array([[Temperature, Humidity, PM25, PM10, NO2, SO2, CO, Proximity, Population]])
    pred = model.predict(input_data)[0]

    st.success(f"Hasil Prediksi Kualitas Udara: {pred}")



label_dict = {
        0: "Baik",
        1: "Sedang",
        2: "Tidak Sehat",
        3: "Sangat Tidak Sehat",
        4: "Berbahaya"
    }

st.success(f"Hasil Prediksi: {label_dict.get(pred, 'Tidak diketahui')} (Kode: {pred})")
