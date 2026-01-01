import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Rotterdam Traffic Predictor",
    page_icon="🚦",
    layout="wide"
)

# Judul dan Header
st.title("🚦 Rotterdam Traffic Predictor")
st.markdown("""
Aplikasi ini memprediksi **Arus Lalu Lintas (Flow)** dan **Okupansi Jalan** di kota Rotterdam 
menggunakan teknologi *Machine Learning*.
""")
st.divider()

# ==========================================
# 2. FUNGSI LOADER (CACHE AGAR CEPAT)
# ==========================================
@st.cache_resource
def load_resources():
    # Load Label Encoder untuk daftar detektor
    encoder = joblib.load('models/label_encoder_detid.pkl')
    return encoder

@st.cache_resource
def load_model(model_name):
    # Bersihkan nama agar sesuai nama file (misal "Extra Trees" -> "ExtraTrees")
    clean_name = model_name.replace(" ", "")
    
    # Load model Flow dan Occupancy sesuai pilihan user
    model_flow = joblib.load(f'models/{clean_name}_flow.pkl')
    model_occ = joblib.load(f'models/{clean_name}_occ.pkl')
    return model_flow, model_occ

try:
    le_det = load_resources()
    # Daftar ID Detektor asli untuk dropdown
    detector_list = list(le_det.classes_)
except FileNotFoundError:
    st.error("File model tidak ditemukan. Pastikan Anda sudah menjalankan TAHAP 1 dan folder 'models' tersedia.")
    st.stop()

# ==========================================
# 3. FUZZY LOGIC (STATUS JALAN)
# ==========================================
def get_traffic_status(flow, occupancy):
    """
    Menentukan status jalan berdasarkan Flow dan Occupancy.
    Referensi: Laporan Munich Traffic Prediction (Bab 3.3)
    """
    status = "Unknown"
    color = "grey"
    
    # Aturan Sederhana (Sesuaikan threshold dengan EDA Anda)
    if occupancy < 5: 
        status = "LANCAR 🟢"
        color = "green"
    elif 5 <= occupancy <= 15:
        status = "PADAT MERAYAP 🟠"
        color = "orange"
    else:
        status = "MACET 🔴"
        color = "red"
        
    # Override jika Flow sangat rendah tapi Occupancy tinggi (Macet total)
    if flow < 100 and occupancy > 20:
        status = "MACET TOTAL (GRIDLOCK) ⚫"
        color = "black"
        
    return status, color

# ==========================================
# 4. SIDEBAR - INPUT USER
# ==========================================
st.sidebar.header("⚙️ Panel Kontrol")

# Pilihan Model
model_choice = st.sidebar.selectbox(
    "Pilih Model AI / Logika:",
    ["Extra Trees", "XGBoost", "LightGBM", "Polynomial Reg"]
)

st.sidebar.divider()

# Input Parameter Waktu
st.sidebar.subheader("1. Waktu Prediksi")
day_input = st.sidebar.selectbox("Hari:", 
    ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu'])
hour_input = st.sidebar.slider("Jam (0-23):", 0, 23, 12)

# Konversi Hari ke Angka (0=Senin, 6=Minggu)
day_mapping = {
    'Senin': 0, 'Selasa': 1, 'Rabu': 2, 'Kamis': 3, 
    'Jumat': 4, 'Sabtu': 5, 'Minggu': 6
}
day_code = day_mapping[day_input]

# Input Lokasi
st.sidebar.subheader("2. Lokasi Sensor")
det_input = st.sidebar.selectbox("ID Detektor:", detector_list)

# Konversi Detektor ID ke Kode Angka
det_code = le_det.transform([det_input])[0]

# Tombol Prediksi
predict_btn = st.sidebar.button("🚀 Prediksi Sekarang", type="primary")

# ==========================================
# 5. HALAMAN UTAMA - HASIL
# ==========================================
col1, col2 = st.columns([2, 1])

with col1:
    if predict_btn:
        with st.spinner(f'Sedang memproses menggunakan logika {model_choice}...'):
            # Load Model yang dipilih
            model_flow, model_occ = load_model(model_choice)
            
            # Siapkan Data Input
            input_data = pd.DataFrame({
                'hour': [hour_input],
                'day_of_week': [day_code],
                'detid_code': [det_code]
            })
            
            # Lakukan Prediksi
            pred_flow = model_flow.predict(input_data)[0]
            pred_occ = model_occ.predict(input_data)[0]
            
            # Pastikan tidak ada nilai negatif (Clipping)
            pred_flow = max(0, pred_flow)
            pred_occ = max(0, pred_occ)
            
            # Tentukan Status
            status_text, status_color = get_traffic_status(pred_flow, pred_occ)
            
            # Tampilkan Hasil
            st.success("Analisis Selesai!")
            
            # Scorecard Layout
            m1, m2, m3 = st.columns(3)
            m1.metric("Prediksi Arus (Flow)", f"{int(pred_flow)} kendaraan")
            m2.metric("Okupansi Jalan", f"{pred_occ:.2f} %")
            m3.metric("Model Digunakan", model_choice)
            
            # Kotak Status Besar
            st.markdown(f"""
            <div style="
                background-color: {status_color};
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                color: white;
                font-size: 24px;
                font-weight: bold;
                margin-top: 20px;">
                STATUS: {status_text}
            </div>
            """, unsafe_allow_html=True)
            
            # Penjelasan Tambahan
            st.info(f"""
            💡 **Interpretasi:**
            Pada hari **{day_input}** jam **{hour_input}:00** di lokasi **{det_input}**, 
            model **{model_choice}** memprediksi kepadatan jalan sebesar **{pred_occ:.1f}%**.
            """)
            
    else:
        st.info("👈 Silakan atur parameter di panel kiri dan klik tombol 'Prediksi Sekarang'.")
        # Tampilkan gambar ilustrasi atau grafik dummy jika belum ada prediksi
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Rotterdam_Skyline_Erasmusbrug.jpg/800px-Rotterdam_Skyline_Erasmusbrug.jpg", 
                 caption="Kota Rotterdam", use_container_width=True)

with col2:
    st.subheader("Tentang Model")
    st.write(f"""
    Anda memilih logika **{model_choice}**.
    
    Setiap model memiliki karakteristik berbeda:
    - **Extra Trees/Random Forest**: Stabil & Akurat.
    - **XGBoost/LightGBM**: Cepat & Cerdas mempelajari pola error.
    - **Polynomial Reg**: Bagus untuk pola gelombang sederhana.
    """)
    
    with st.expander("Lihat Aturan Fuzzy Logic"):
        st.write("""
        - **Occupancy < 5%**: Lancar 🟢
        - **Occupancy 5-15%**: Padat Merayap 🟠
        - **Occupancy > 15%**: Macet 🔴
        """)