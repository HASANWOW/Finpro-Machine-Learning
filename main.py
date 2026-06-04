import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="FoodVibe App", page_icon="🍔", layout="centered")

# ==========================================
# INJECT CUSTOM CSS & HTML
# ==========================================
custom_css = """
<style>
    /* Mengambil font Poppins dari Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    /* Menerapkan font ke seluruh teks di aplikasi */
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    /* Menyembunyikan menu bawaan Streamlit agar terlihat seperti aplikasi mandiri */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Styling Judul Utama */
    h1 {
        color: #FF4B4B;
        text-align: center;
        font-weight: 700;
        margin-bottom: -10px;
    }

    /* Styling Subjudul */
    .subjudul {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 30px;
    }

    /* Styling Tombol Utama (Gradient & Shadow) */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8F8F 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 18px;
        font-weight: 600;
        box-shadow: 0px 4px 10px rgba(255, 75, 75, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    /* Efek melayang (Hover) pada tombol */
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0px 6px 15px rgba(255, 75, 75, 0.5);
    }

    /* Styling Custom Card untuk Hasil Prediksi */
    .result-card-sehat {
        background: linear-gradient(135deg, #D4EDDA 0%, #A3E4D7 100%);
        border-left: 8px solid #28A745;
        padding: 20px;
        border-radius: 10px;
        color: #155724;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .result-card-kurang {
        background: linear-gradient(135deg, #FFF3CD 0%, #FDEBD0 100%);
        border-left: 8px solid #FFC107;
        padding: 20px;
        border-radius: 10px;
        color: #856404;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ==========================================
# 1. LOAD MODEL (Latency < 100ms)
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load('models/best_model.pkl')

model = load_model()

# ==========================================
# 2. TAMPILAN ANTARMUKA (UI)
# ==========================================
# Menggunakan HTML untuk memusatkan teks judul
st.markdown("<h1>🍔 FoodVibe</h1>", unsafe_allow_html=True)
st.markdown("<p class='subjudul'>Classifying Food Choice Behavior in College Students</p>", unsafe_allow_html=True)

st.write("---")
st.markdown("### ✨ Ceritakan Kondisimu Hari Ini")

# Membagi form menjadi 2 kolom agar layout lebih modern dan padat
col1, col2 = st.columns(2)

with col1:
    mood_options = {
        "Stres / Sedih / Bosan (Negatif)": 1,
        "Bahagia / Merayakan (Positif)": 2,
        "Lapar biasa": 3,
        "Lainnya": 4
    }
    mood_input = st.selectbox("🧠 Kondisi Mood Utama Hari Ini?", options=list(mood_options.keys()))

    fries_options = {
        "Ya, sangat suka": 1,
        "Biasa saja / Tidak suka": 2
    }
    fries_input = st.selectbox("🍟 Kebiasaan ngemil makanan cepat saji (misal: kentang goreng)?", options=list(fries_options.keys()))

with col2:
    eating_out_input = st.slider("🛵 Frekuensi makan di luar (per minggu)?", min_value=1, max_value=5, value=3)
    nutri_input = st.slider("📋 Sering cek label nutrisi makanan?", min_value=1, max_value=5, value=2)

st.write("<br>", unsafe_allow_html=True) # Memberi jarak kosong

# ==========================================
# 3. PROSES PREDIKSI
# ==========================================
if st.button("🔮 Analisis Pola Makan Saya"):
    input_data = pd.DataFrame({
        'comfort_food_reasons_coded': [mood_options[mood_input]],
        'eating_out': [eating_out_input],
        'fries': [fries_options[fries_input]],
        'nutritional_check': [nutri_input]
    })
    
    with st.spinner("Mengolah data dengan AI..."):
        prediction = model.predict(input_data)[0]
    
    st.write("---")
    st.markdown("### 📊 Hasil Analisis:")
    
    if prediction == 1:
        # Menampilkan kartu hasil Sehat dengan HTML
        html_result = """
        <div class="result-card-sehat">
            <h3 style="margin-top: 0; color: #155724;">🥗 Pola Makan Cenderung SEHAT</h3>
            <p style="margin-bottom: 0;">Luar biasa! Berdasarkan kondisi mood dan gaya hidupmu saat ini, kamu memiliki kecenderungan memilih makanan yang teratur dan bernutrisi. Pertahankan kebiasaan baik ini!</p>
        </div>
        """
        st.markdown(html_result, unsafe_allow_html=True)
        st.balloons() # Animasi balon dari Streamlit
    else:
        # Menampilkan kartu hasil Kurang Sehat dengan HTML
        html_result = """
        <div class="result-card-kurang">
            <h3 style="margin-top: 0; color: #856404;">🍔 Pola Makan Cenderung KURANG SEHAT</h3>
            <p style="margin-bottom: 0;">Wah, sepertinya situasi emosional atau kesibukanmu memicu <i>decision fatigue</i> sehingga kamu memilih makanan yang kurang seimbang. Yuk, mulai lebih sadar dengan asupan makananmu!</p>
        </div>
        """
        st.markdown(html_result, unsafe_allow_html=True)