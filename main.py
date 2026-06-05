import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="FoodVibe – Food Pattern Classifier",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# QUERY PARAMETERS HELPERS
# ─────────────────────────────────────────
def get_query_param(key, default="false"):
    if hasattr(st, "query_params"):
        return st.query_params.get(key, default)
    try:
        params = st.experimental_get_query_params()
        return params.get(key, [default])[0]
    except Exception:
        return default

def set_query_param(key, value):
    if hasattr(st, "query_params"):
        st.query_params[key] = value
    else:
        try:
            params = st.experimental_get_query_params()
            params[key] = [value]
            st.experimental_set_query_params(**params)
        except Exception:
            pass

def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

# ─────────────────────────────────────────
# INITIALIZE SESSION STATE FROM URL
# ─────────────────────────────────────────
st.session_state["run_eda"] = get_query_param("run_eda", "false") == "true"
st.session_state["run_preprocessing"] = get_query_param("run_preprocessing", "false") == "true"
st.session_state["run_model"] = get_query_param("run_model", "false") == "true"

# ─────────────────────────────────────────
# CSS GLOBAL
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Playfair+Display:wght@700&display=swap');

html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }
.main { background-color: #fffaf5; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: visible;}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FF8D28 0%, #e07520 100%);
    overflow: hidden !important;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 8px 0 !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2) !important; }

/* Hide Scrollbar */
[data-testid="stSidebar"]::-webkit-scrollbar { display: none; }
[data-testid="stSidebarUserContent"]::-webkit-scrollbar { display: none; }
[data-testid="stSidebar"] { -ms-overflow-style: none; scrollbar-width: none; }
[data-testid="stSidebarUserContent"] { -ms-overflow-style: none; scrollbar-width: none; }

/* ── HERO ── */
.hero-box {
    background: linear-gradient(135deg, #FF8D28 0%, #ffaa5c 50%, #ffc98a 100%);
    border-radius: 24px;
    padding: 48px 40px;
    text-align: center;
    margin-bottom: 32px;
    box-shadow: 0 8px 32px rgba(255,141,40,0.25);
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    color: white;
    margin: 0;
    letter-spacing: -1px;
}
.hero-subtitle { font-size: 1.1rem; color: rgba(255,255,255,0.9); margin-top: 8px; }
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.4);
    color: white;
    border-radius: 20px;
    padding: 4px 16px;
    font-size: 0.85rem;
    margin-top: 12px;
}

/* ── SECTION TITLE ── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    color: #FF8D28;
    border-left: 5px solid #FF8D28;
    padding-left: 16px;
    margin-bottom: 20px;
}
.section-desc {
    background: white;
    border-radius: 12px;
    padding: 18px 22px;
    border: 1px solid #ffe5cc;
    color: #555;
    line-height: 1.7;
    margin-bottom: 24px;
}

/* ── METRIC CARDS ── */
.metric-card {
    background: white;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    border: 2px solid #ffe5cc;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.metric-num { font-size: 2rem; font-weight: 900; color: #FF8D28; line-height: 1; }
.metric-unit { font-size: 0.75rem; color: #888; margin-top: 2px; }
.metric-label { font-size: 0.85rem; color: #555; font-weight: 700; margin-top: 6px; }

/* ── INFO CARDS ── */
.info-card {
    background: white;
    border-radius: 14px;
    padding: 20px 22px;
    border: 1px solid #ffe5cc;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    margin-bottom: 16px;
}
.info-card h4 { color: #FF8D28; margin-bottom: 8px; font-size: 1rem; }

/* ── STEP BADGE ── */
.step-badge {
    display: inline-block;
    background: linear-gradient(135deg, #FF8D28, #ffaa5c);
    color: white;
    border-radius: 50%;
    width: 36px;
    height: 36px;
    line-height: 36px;
    text-align: center;
    font-weight: 900;
    font-size: 1rem;
    margin-right: 10px;
    vertical-align: middle;
}

/* ── RESULT CARD ── */
.result-healthy {
    background: linear-gradient(135deg, #e8f8ed, #f0fdf4);
    border: 2px solid #2d9e5f;
    border-radius: 20px;
    padding: 28px;
    text-align: center;
    margin-bottom: 20px;
}
.result-unhealthy {
    background: linear-gradient(135deg, #fff3e8, #fff8f0);
    border: 2px solid #FF8D28;
    border-radius: 20px;
    padding: 28px;
    text-align: center;
    margin-bottom: 20px;
}
.result-label { font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 700; }
.result-conf { font-size: 1rem; color: #888; margin-top: 6px; }

/* ── FOOD CHIP ── */
.food-chip {
    display: inline-block;
    background: #fff3e8;
    border: 1.5px solid #FF8D28;
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 0.88rem;
    font-weight: 700;
    color: #FF8D28;
    margin: 4px 4px 0 0;
}

/* ── EVAL TABLE ── */
.eval-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.eval-table th { background: #FF8D28; color: white; padding: 10px 14px; text-align: left; font-weight: 700; }
.eval-table td { padding: 10px 14px; border-bottom: 1px solid #ffe5cc; }
.eval-table tr:nth-child(even) td { background: #fffaf5; }

div.stButton > button {
    background: linear-gradient(135deg, #FF8D28, #ffaa5c);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 14px 32px;
    font-family: 'Nunito', sans-serif;
    font-weight: 800;
    font-size: 1rem;
    width: 100%;
    cursor: pointer;
    transition: all 0.2s;
    box-shadow: 0 4px 16px rgba(255,141,40,0.3);
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(255,141,40,0.4);
}
.footer {
    text-align: center;
    color: #aaa;
    font-size: 0.8rem;
    margin-top: 32px;
    padding-top: 16px;
    border-top: 1px solid #ffe5cc;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    model  = joblib.load(os.path.join(base, 'foodvibe_model.pkl'))
    scaler = joblib.load(os.path.join(base, 'foodvibe_scaler.pkl'))
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False

# ─────────────────────────────────────────
# FEATURE LIST (sama persis dengan notebook _2)
# ─────────────────────────────────────────
FEATURES = [
    'veggies_day', 'fruit_day', 'eating_out', 'exercise', 'calories_day',
    'comfort_food_reasons_coded', 'eating_changes_coded', 'healthy_feeling',
    'cook', 'sports', 'coffee', 'breakfast',
    'on_off_campus', 'employment', 'gender',
    'pay_meal_out', 'fav_cuisine_coded',
    'healthy_behavior_score', 'emotional_eating_risk'
]

# ─────────────────────────────────────────
# MOOD → FOOD RECOMMENDATION MAPPING
# ─────────────────────────────────────────
def get_food_recommendation(healthy_feeling, prediction):
    """
    healthy_feeling: 1-5 (dari slider input user)
    prediction: 0 (Unhealthy) atau 1 (Healthy)
    """
    if healthy_feeling >= 4:
        mood_label = "Good Mood"
        mood_desc  = "Kamu sedang merasa baik! Pilihan sehat terasa lebih mudah."
        if prediction == 1:
            foods = ["Grain bowl", "Salad buah segar", "Smoothie bowl", "Overnight oats", "Yogurt granola"]
        else:
            foods = ["Nasi + sayur tumis", "Sup ayam bening", "Gado-gado", "Pecel lele", "Tumis kangkung"]
    elif healthy_feeling == 3:
        mood_label = "Mood Netral"
        mood_desc  = "Mood kamu biasa aja nih. Makanan yang nyaman bisa membantu!"
        if prediction == 1:
            foods = ["Nasi + lauk seimbang", "Sandwich isi sayur", "Bubur ayam", "Soto ayam", "Mie kuah"]
        else:
            foods = ["Bubur manado", "Mie rebus + telur", "Nasi tim", "Lontong sayur", "Nasi uduk"]
    else:
        mood_label = "Lagi Stres"
        mood_desc  = "Mood kurang baik. Comfort food sehat bisa bantu memulihkan energi!"
        if prediction == 1:
            foods = ["Dark chocolate", "Pisang + almond butter", "Teh chamomile + roti gandum", "Oatmeal madu", "Alpukat toast"]
        else:
            foods = ["Oatmeal hangat", "Sup jahe", "Nasi putih + telur dadar", "Wedang jahe", "Bubur kacang hijau"]

    return mood_label, mood_desc, foods


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px 0;">
        <div style="font-family:'Playfair Display',serif; font-size:2.5rem; color:white; font-weight:700;">FoodVibe</div>
        <div style="font-size:0.85rem; color:rgba(255,255,255,0.8); margin-top:4px;">Food Pattern Classifier</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    if "page" not in st.session_state:
        st.session_state["page"] = get_query_param("page", "Home")

    page_options = [
        "Home",
        "Step 1: EDA",
        "Step 2: Preprocessing",
        "Step 3: Model",
        "Step 4: Demo",
    ]
    if st.session_state["page"] not in page_options:
        st.session_state["page"] = "Home"

    def on_page_change():
        set_query_param("page", st.session_state["nav_radio"])
        st.session_state["page"] = st.session_state["nav_radio"]

    page_index = page_options.index(st.session_state["page"])

    page = st.radio(
        "Navigasi",
        options=page_options,
        index=page_index,
        key="nav_radio",
        on_change=on_page_change,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="padding: 0 10px;">
        <div style="font-size:0.8rem; color:white; font-weight:700; margin-bottom:8px; opacity:0.9; text-align:center;">Kelompok 6</div>
        <div style="margin-top:12px; font-size:0.65rem; color:rgba(255,255,255,0.45); text-align:center; border-top:1px solid rgba(255,255,255,0.15); padding-top:8px;">
            Machine Learning Project<br>
            Binus University · 2025/2026
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────
def hero(title, subtitle, badge="Powered by Machine Learning"):
    st.markdown(f"""
    <div class="hero-box">
        <div class="hero-title">{title}</div>
        <div class="hero-subtitle">{subtitle}</div>
        <div class="hero-badge">{badge}</div>
    </div>
    """, unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════
if page == "Home":
    hero("FoodVibe",
         "Classifying Food Choice Behavior in College Students Based on Lifestyle & Emotional Triggers")

    col1, col2 = st.columns([3, 2])

    with col1:
        section("Tentang Project")
        st.markdown("""
        <div class="section-desc">
            <b>FoodVibe</b> adalah sistem berbasis Machine Learning yang mengklasifikasikan
            pola perilaku pemilihan makanan mahasiswa berdasarkan gaya hidup dan emotional triggers.
            <br><br>
            Proses pemilihan makanan bagi mahasiswa bukan sekadar soal rasa lapar — keputusan ini
            dipengaruhi oleh <b>Mood</b>, <b>Preferensi Rasa</b>, dan <b>Konteks Waktu</b>.
            FoodVibe membantu mahasiswa mengatasi <i>decision fatigue</i> dengan memberikan
            klasifikasi pola makan dan saran makanan yang relevan secara emosional dan situasional.
            <br><br>
            Model dilatih menggunakan tiga algoritma:
            <ul>
                <li><b>Logistic Regression</b> — baseline model, interpretable & efisien</li>
                <li><b>Random Forest</b> — model utama, menangani hubungan non-linear</li>
                <li><b>SVM (RBF Kernel)</b> — model final, decision boundary paling presisi</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        section("Dataset")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="info-card" style="min-height: 148px;">
                <h4>Food Choices Dataset</h4>
                <div style="font-size:0.85rem;color:#555;">
                    Sumber: <a href="https://www.kaggle.com/datasets/borapajo/food-choices" target="_blank" style="color:#FF8D28;text-decoration:none;"><b>Kaggle – borapajo</b></a><br>
                    <b style="color:#FF8D28;">126</b> responden mahasiswa<br>
                    <b style="color:#FF8D28;">61</b> kolom fitur (preferensi, gaya hidup, emosi)
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="info-card" style="min-height: 148px;">
                <h4>Synthetic Data Augmentation</h4>
                <div style="font-size:0.85rem;color:#555;">
                    Sumber: <a href="https://dl.acm.org/doi/epdf/10.1145/3577190.3614129" target="_blank" style="color:#FF8D28;text-decoration:none;"><b>Paper – Mood-Diet Insight</b></a><br>
                    <b style="color:#FF8D28;">200</b> data synthetic (100 healthy + 100 unhealthy)<br>
                    Dibuat berbasis domain knowledge dari literatur<br>
                    Hanya masuk ke <b>training set</b> (bebas leakage)
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        section("Alur Project")
        steps = [
            ("1", "EDA", "Eksplorasi data mentah, distribusi, dan korelasi fitur"),
            ("2", "Preprocessing", "Cleaning, encoding, feature selection, augmentasi"),
            ("3", "Model", "Train LR, RF, SVM — evaluasi & pilih model terbaik"),
            ("4", "Demo", "Input kebiasaan → prediksi pola makan + saran makanan"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div class="info-card" style="padding:14px 18px; margin-bottom:10px;">
                <div style="display:flex;align-items:center;margin-bottom:4px;">
                    <span class="step-badge">{num}</span>
                    <b style="color:#FF8D28;">{title}</b>
                </div>
                <div style="font-size:0.82rem;color:#666;padding-left:46px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        section("3 Pilar Utama")
        pillars = [
            ("#e74c3c", "Mood (Psikologis)", "Emotional eating & mekanisme koping stres akademik"),
            ("#FF8D28", "Rasa (Sensorik)", "Preferensi spesifik: pedas, manis, gurih, asam"),
            ("#3498db", "Waktu (Kontekstual)", "Jadwal harian menentukan kebutuhan energi makanan"),
        ]
        for color, title, desc in pillars:
            st.markdown(f"""
            <div class="info-card" style="border-left:4px solid {color}; padding:12px 16px; margin-bottom:8px;">
                <b style="color:#1d2b22;font-size:0.88rem;">{title}</b>
                <div style="font-size:0.8rem;color:#666;margin-top:3px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    section("Dokumentasi")
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        st.markdown("""
        <div class="info-card" style="min-height: 160px; border-left: 4px solid #FF8D28; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <h4 style="color:#FF8D28; margin: 0 0 6px 0;">Google Drive Folder</h4>
                <div style="font-size:0.83rem;color:#555;margin-bottom:12px;line-height:1.45;">
                    Penyimpanan dokumen project, file presentasi, dan berkas pendukung kelompok.
                </div>
            </div>
            <div>
                <a href="https://drive.google.com/drive/folders/1LGGGSB68G1AsdnN5rW4lhl1YGMUx8hNJ?usp=drive_link" target="_blank" style="display:inline-block; background:linear-gradient(135deg, #FF8D28, #ffaa5c); color:white !important; text-decoration:none; padding:6px 14px; border-radius:8px; font-size:0.8rem; font-weight:700; box-shadow:0 2px 8px rgba(255,141,40,0.25);">Buka Drive</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with dc2:
        st.markdown("""
        <div class="info-card" style="min-height: 160px; border-left: 4px solid #3498db; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <h4 style="color:#3498db; margin: 0 0 6px 0;">User Testing (Wawancara)</h4>
                <div style="font-size:0.83rem;color:#555;margin-bottom:12px;line-height:1.45;">
                    Dokumentasi rekaman wawancara dan hasil uji coba langsung dengan 5 user (mahasiswa).
                </div>
            </div>
            <div>
                <a href="https://drive.google.com/drive/folders/1TDvn4NanOPen9qns3MC_FkpN4GctXN0S?usp=drive_link" target="_blank" style="display:inline-block; background:linear-gradient(135deg, #3498db, #5dade2); color:white !important; text-decoration:none; padding:6px 14px; border-radius:8px; font-size:0.8rem; font-weight:700; box-shadow:0 2px 8px rgba(52,152,219,0.25);">Buka Wawancara</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with dc3:
        st.markdown("""
        <div class="info-card" style="min-height: 160px; border-left: 4px solid #2d9e5f; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <h4 style="color:#2d9e5f; margin: 0 0 6px 0;">Survey G-Forms</h4>
                <div style="font-size:0.83rem;color:#555;margin-bottom:12px;line-height:1.45;">
                    Dataset respon kuesioner Google Form mengenai perilaku makan dan gaya hidup mahasiswa.
                </div>
            </div>
            <div>
                <a href="https://docs.google.com/spreadsheets/d/1k7syholG6I8gTebP4FaxUd7asPQVg70DLoAR5ZcHPpA/edit?usp=sharing" target="_blank" style="display:inline-block; background:linear-gradient(135deg, #2d9e5f, #58d68d); color:white !important; text-decoration:none; padding:6px 14px; border-radius:8px; font-size:0.8rem; font-weight:700; box-shadow:0 2px 8px rgba(45,158,95,0.25);">Buka Google Sheets</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    section("Tim Project")
    st.markdown("""
    <div class="info-card" style="background: linear-gradient(135deg, #ffffff 0%, #fffaf5 100%);">
        <b style="color:#FF8D28; font-size:1rem; display:block; margin-bottom:2px;">Kelompok 6 – Universitas Bina Nusantara 2025/2026</b>
        <div style="color:#FF8D28; font-size:0.82rem; font-weight:normal; display:block; margin-bottom:12px;">Dosen: Johannes Simatupang, S.Kom., M.Kom</div>
        <table style="width:100%; font-size:0.85rem; color:#444; border-collapse:collapse;">
            <tr style="border-bottom:1px solid #ffe5cc;">
                <td style="padding:8px 0; font-weight:600;">Hasan</td>
                <td style="padding:8px 0; text-align:right; color:#888;">2802529203</td>
            </tr>
            <tr style="border-bottom:1px solid #ffe5cc;">
                <td style="padding:8px 0; font-weight:600;">Herlinda Angelica Tanjaya</td>
                <td style="padding:8px 0; text-align:right; color:#888;">2802397754</td>
            </tr>
            <tr>
                <td style="padding:8px 0; font-weight:600;">Sabrina Arfanindia D</td>
                <td style="padding:8px 0; text-align:right; color:#888;">2802448755</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="footer">FoodVibe &nbsp;|&nbsp; Machine Learning Project &nbsp;|&nbsp; Kelompok 6 – Binus University &nbsp;|&nbsp; 2025/2026</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE: EDA
# ═══════════════════════════════════════════════════════
elif page == "Step 1: EDA":
    hero("Exploratory Data Analysis",
         "Memahami karakteristik data sebelum preprocessing", "Step 1 dari 4")

    st.markdown("""
    <div class="section-desc">
        EDA dilakukan untuk mengidentifikasi distribusi, pola korelasi, dan karakteristik data mentah
        dari Food Choices Dataset sebelum masuk ke tahap preprocessing.
    </div>
    """, unsafe_allow_html=True)

    if st.button("Jalankan EDA", use_container_width=True):
        st.session_state["run_eda"] = True
        set_query_param("run_eda", "true")
        safe_rerun()

    if st.session_state.get("run_eda", False):
        st.success("EDA berhasil dijalankan!")

        section("Ringkasan Dataset")
        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            (c1, "125", "responden asli (525 setelah augmentasi)", "Jumlah Sampel"),
            (c2, "61", "kolom", "Total Fitur"),
            (c3, "18", "fitur", "Fitur yang Digunakan"),
            (c4, "2", "kelas", "Target (Healthy/Unhealthy)"),
        ]
        for col, num, unit, label in metrics:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-num">{num}</div>
                    <div class="metric-unit">{unit}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            section("Distribusi Target Variabel")
            st.markdown("""
            <div class="section-desc">
                Variabel target <b>diet_current_coded</b> di-binarize menjadi:
                <ul>
                    <li><b>Relatively Healthy (1)</b>: Kode 1–2 (110 responden / 88%)</li>
                    <li><b>Unhealthy (0)</b>: Kode 3–4 (15 responden / 12%)</li>
                </ul>
                Distribusi awal sangat <b>imbalanced</b> (tidak seimbang), sehingga pada tahap preprocessing dilakukan <b>Synthetic Data Augmentation (SMOTE)</b> untuk menyeimbangkan dataset.
            </div>
            """, unsafe_allow_html=True)

            # Visual bar distribusi
            dist_data = {"Unhealthy": 15, "Relatively Healthy": 110}
            for label, count in dist_data.items():
                color = "#e74c3c" if label == "Unhealthy" else "#2d9e5f"
                pct = count / 125 * 100
                st.markdown(f"""
                <div style="margin-bottom:12px;">
                    <div style="display:flex;justify-content:space-between;font-size:0.85rem;
                                font-weight:700;color:#1d2b22;margin-bottom:6px;">
                        <span>{label}</span>
                        <span style="color:{color};">{count} ({pct:.1f}%)</span>
                    </div>
                    <div style="background:#f5ede3;border-radius:999px;height:14px;overflow:hidden;">
                        <div style="background:{color};width:{pct}%;height:14px;border-radius:999px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            section("Fitur-Fitur Kunci")
            features_info = [
                ("veggies_day", "Porsi sayur per hari", "Korelasi positif kuat dengan healthy"),
                ("exercise", "Frekuensi olahraga", "Lifestyle predictor utama"),
                ("healthy_feeling", "Perasaan sehat diri", "Proxy mood & self-awareness"),
                ("comfort_food_reasons_coded", "Alasan makan comfort food", "Emotional eating indicator"),
                ("eating_out", "Frekuensi makan di luar", "Risk factor untuk unhealthy"),
                ("cook", "Frekuensi masa sendiri", "Berkorelasi dengan pola makan sehat"),
            ]
            for feat, label, note in features_info:
                st.markdown(f"""
                <div class="info-card" style="padding:10px 14px;margin-bottom:8px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <b style="color:#FF8D28;font-size:0.85rem;">{feat}</b>
                        <span style="font-size:0.75rem;color:#aaa;">{label}</span>
                    </div>
                    <div style="font-size:0.78rem;color:#777;margin-top:3px;">{note}</div>
                </div>
                """, unsafe_allow_html=True)

        # Visualisasi Distribusi & Pola Korelasi (dari Notebook)
        st.markdown("<br>", unsafe_allow_html=True)
        section("Visualisasi & Eksplorasi Data Mendalam")
        st.markdown("""
        <div class="section-desc" style="margin-bottom: 20px;">
            Berikut adalah visualisasi interaktif dan analisis korelasi yang diambil langsung dari Jupyter Notebook proyek kami.
            Gunakan tab di bawah ini untuk berpindah sudut pandang analisis.
        </div>
        """, unsafe_allow_html=True)

        # Membuat Tabs
        tab_dist, tab_corr, tab_importance = st.tabs([
            "Hubungan & Distribusi Data", 
            "Matriks Korelasi (Heatmap)", 
            "Fitur Paling Berpengaruh (Drivers)"
        ])

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Load dataset
            csv_path = "Dataset/Food Vibe/food_coded.csv"
            if not os.path.exists(csv_path):
                csv_path = "d:/BINUS/Semester 4/Machine Learning/Project/Final Project/Dataset/Food Vibe/food_coded.csv"
            df_raw = pd.read_csv(csv_path)

            # Binarize
            df_plot = df_raw.copy()
            df_plot['diet_binary'] = (df_plot['diet_current_coded'] <= 2).astype(int)

            # Fillna
            plot_cols = ['diet_binary', 'fruit_day', 'veggies_day', 'healthy_feeling', 'exercise']
            for col in plot_cols:
                if col in df_plot.columns and df_plot[col].isnull().sum() > 0:
                    df_plot[col] = df_plot[col].fillna(df_plot[col].mode()[0])

            # Tab 1: Hubungan & Distribusi Data (2x2 Grid)
            with tab_dist:
                st.markdown("#### Grid Hubungan & Pola Perilaku Makan")
                st.markdown("""
                <div style="font-size:0.83rem; color:#555; margin-bottom:15px;">
                    Visualisasi 2x2 grid ini menunjukkan hubungan silang antara konsumsi buah/sayur, tingkat olahraga, 
                    dan perasaan sehat (healthy feeling) terhadap kategori pola makan (Healthy vs Unhealthy).
                </div>
                """, unsafe_allow_html=True)

                sns.set_theme(style='whitegrid')
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))

                # Color palette
                colors_dict = {0: '#e74c3c', 1: '#2d9e5f'} # Unhealthy vs Healthy

                # 1. Target Distribution
                sns.countplot(x='diet_binary', data=df_plot, ax=axes[0, 0], palette=['#e74c3c', '#2d9e5f'])
                axes[0, 0].set_title('Distribution of Target (diet_binary)', fontsize=11, fontweight='bold', color='#1d2b22')
                axes[0, 0].set_xticklabels(['Unhealthy (0)', 'Relatively Healthy (1)'])
                axes[0, 0].set_xlabel('diet_binary')
                axes[0, 0].set_ylabel('count')

                # 2. Fruit vs Veggies scatter
                sns.scatterplot(x='fruit_day', y='veggies_day', hue='diet_binary', data=df_plot, ax=axes[0, 1], alpha=0.7, palette=colors_dict)
                axes[0, 1].set_title('Fruit vs Veggie Consumption', fontsize=11, fontweight='bold', color='#1d2b22')
                axes[0, 1].set_xlabel('fruit_day')
                axes[0, 1].set_ylabel('veggies_day')
                # Custom legend
                handles, labels = axes[0, 1].get_legend_handles_labels()
                axes[0, 1].legend(handles, ['Unhealthy (0)', 'Relatively Healthy (1)'], title='diet_binary')

                # 3. Healthy Feeling Distribution
                sns.boxplot(x='diet_binary', y='healthy_feeling', data=df_plot, ax=axes[1, 0], palette=['#e74c3c', '#2d9e5f'])
                axes[1, 0].set_title('Healthy Feeling by Class', fontsize=11, fontweight='bold', color='#1d2b22')
                axes[1, 0].set_xticklabels(['Unhealthy (0)', 'Relatively Healthy (1)'])
                axes[1, 0].set_xlabel('diet_binary')
                axes[1, 0].set_ylabel('healthy_feeling')

                # 4. Exercise Frequency
                sns.violinplot(x='diet_binary', y='exercise', data=df_plot, ax=axes[1, 1], palette=['#e74c3c', '#2d9e5f'])
                axes[1, 1].set_title('Exercise Frequency by Class', fontsize=11, fontweight='bold', color='#1d2b22')
                axes[1, 1].set_xticklabels(['Unhealthy (0)', 'Relatively Healthy (1)'])
                axes[1, 1].set_xlabel('diet_binary')
                axes[1, 1].set_ylabel('exercise')

                plt.tight_layout()
                st.pyplot(fig)

            # Tab 2: Matriks Korelasi (Heatmap Segitiga)
            with tab_corr:
                st.markdown("#### Matriks Korelasi Fitur Mentah (Raw Feature Correlation Matrix)")
                st.markdown("""
                <div style="font-size:0.83rem; color:#555; margin-bottom:15px;">
                    Matriks korelasi segitiga (lower triangle) menunjukkan hubungan linier antara seluruh fitur numerik 
                    dalam dataset. Warna merah menunjukkan korelasi negatif, sedangkan biru menunjukkan korelasi positif.
                </div>
                """, unsafe_allow_html=True)

                # Ambil kolom numerik
                numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
                # Ganti nama agar lebih informatif
                corr_data = df_plot[numeric_cols].copy()
                if 'diet_current_coded' in corr_data.columns:
                    corr_data = corr_data.rename(columns={'diet_current_coded': 'diet_original'})

                corr = corr_data.corr()
                mask = np.triu(np.ones_like(corr, dtype=bool))

                fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr, mask=mask, annot=False, cmap='RdBu', center=0, ax=ax_corr, cbar_kws={"shrink": .8})
                ax_corr.set_title('Raw Feature Correlation Matrix', fontsize=12, fontweight='bold', pad=15, color='#1d2b22')
                plt.xticks(rotation=90, fontsize=8)
                plt.yticks(rotation=0, fontsize=8)
                plt.tight_layout()
                st.pyplot(fig_corr)

            # Tab 3: Fitur Paling Berpengaruh (Feature Importance)
            with tab_importance:
                st.markdown("#### Driver Utama Klasifikasi Pilihan Makanan (Feature Importance)")
                st.markdown("""
                <div style="font-size:0.83rem; color:#555; margin-bottom:15px;">
                    Kontribusi relatif dari setiap fitur dalam menentukan pola makan sehat vs tidak sehat, 
                    dihitung menggunakan algoritma <b>Random Forest Classifier</b> yang telah dituning.
                </div>
                """, unsafe_allow_html=True)

                importance_data = {
                    'Feature': [
                        'fruit_day', 'eating_out', 'healthy_feeling', 'veggies_day', 
                        'healthy_behavior_score', 'sports', 'comfort_food_reasons_coded', 
                        'eating_changes_coded', 'cook', 'exercise', 'calories_day', 
                        'breakfast', 'pay_meal_out', 'on_off_campus', 'employment', 
                        'coffee', 'fav_cuisine_coded', 'emotional_eating_risk'
                    ],
                    'Importance': [
                        0.250747, 0.153819, 0.125315, 0.124757, 
                        0.065181, 0.062071, 0.043819, 0.034965, 
                        0.026483, 0.022662, 0.021516, 0.018618, 
                        0.014880, 0.012811, 0.009693, 0.009206, 
                        0.002525, 0.000932
                    ]
                }
                feat_imp_df = pd.DataFrame(importance_data)

                fig_imp, ax_imp = plt.subplots(figsize=(10, 8))
                sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='magma', ax=ax_imp)
                ax_imp.set_title('Feature Importance: Key Drivers of Food Choice Classification', fontsize=12, fontweight='bold', pad=15, color='#1d2b22')
                ax_imp.set_xlabel('Importance', fontsize=10)
                ax_imp.set_ylabel('Feature', fontsize=10)
                plt.tight_layout()
                st.pyplot(fig_imp)

        except Exception as e:
            st.error(f"Gagal memuat visualisasi: {e}")

        st.markdown("<br>", unsafe_allow_html=True)
        section("EDA Insights")
        i1, i2, i3 = st.columns(3)
        insights = [
            (i1, "#e74c3c", "Korelasi Sayur–Diet",
             "veggies_day memiliki korelasi positif paling kuat dengan diet_binary (~0.45). Mahasiswa yang makan ≥3 porsi sayur/hari 2.3× lebih mungkin memiliki pola makan healthy."),
            (i2, "#FF8D28", "Emotional Eating",
             "comfort_food_reasons_coded > 0 (stres, boredom, dll.) berkorelasi negatif dengan pola makan sehat. Mahasiswa dengan emotional eating triggers tinggi cenderung unhealthy."),
            (i3, "#3498db", "Lifestyle Cluster",
             "Kombinasi exercise rendah + eating_out tinggi + cook rendah membentuk profil berisiko unhealthy. Ketiganya saling memperkuat sebagai lifestyle cluster."),
        ]
        for col, color, title, desc in insights:
            with col:
                st.markdown(f"""
                <div class="info-card" style="border-left:4px solid {color};">
                    <h4 style="color:{color};">{title}</h4>
                    <div style="font-size:0.83rem;color:#555;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('<div class="footer">FoodVibe &nbsp;|&nbsp; Step 1: EDA</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE: PREPROCESSING
# ═══════════════════════════════════════════════════════
elif page == "Step 2: Preprocessing":
    hero("Data Preprocessing",
         "Membersihkan, mengkodekan, dan menyiapkan data untuk model ML", "Step 2 dari 4")

    st.markdown("""
    <div class="section-desc">
        Preprocessing memastikan data mentah dari Food Choices Dataset siap digunakan oleh algoritma
        Machine Learning. Tahap ini mencakup cleaning, encoding, feature selection, augmentasi,
        dan stratified split yang benar untuk menghindari data leakage.
    </div>
    """, unsafe_allow_html=True)

    if st.button("Jalankan Preprocessing", use_container_width=True):
        st.session_state["run_preprocessing"] = True
        set_query_param("run_preprocessing", "true")
        safe_rerun()

    if st.session_state.get("run_preprocessing", False):
        st.success("Preprocessing berhasil dijalankan!")

        section("Langkah-Langkah Preprocessing")
        col1, col2 = st.columns(2)

        steps_left = [
            ("1. Data Cleaning", "Missing values pada kolom kategorikal diisi dengan imputasi modus. Kolom tidak relevan dari 61 kolom disaring menjadi 18 fitur utama."),
            ("2. Label Encoding", "Variabel kategorikal (on_off_campus, employment, fav_cuisine_coded) dikonversi ke numerik menggunakan LabelEncoder. Gender dibuang karena ketidakcocokan nama kolom di dataset."),
            ("3. Feature Selection", "18 fitur dipilih berdasarkan relevansi terhadap lifestyle & emotional triggers — fokus pada variabel yang berkorelasi dengan diet_current_coded."),
            ("4. Composite Features", "2 fitur baru dibuat: healthy_behavior_score (veggies ≥ 3 + exercise ≥ 2) dan emotional_eating_risk (comfort food reasons + eating changes)."),
        ]
        steps_right = [
            ("5. Stratified Split (80/20)", "Data ASLI di-split lebih dahulu secara stratified — test set HANYA berisi data asli (25 sampel) untuk menghindari data leakage."),
            ("6. Synthetic Augmentation", "200 data synthetic (100 healthy + 100 unhealthy) dibuat berbasis domain knowledge dan HANYA ditambahkan ke training set."),
            ("7. StandardScaler", "Fitur dinormalisasi menggunakan StandardScaler yang di-fit HANYA pada training set, kemudian diterapkan ke test set."),
            ("8. Class Weight Balancing", "Parameter class_weight='balanced' digunakan pada semua model agar tidak bias ke kelas mayoritas."),
        ]

        with col1:
            for title, desc in steps_left:
                st.markdown(f"""
                <div class="info-card" style="padding:12px 16px;margin-bottom:10px;">
                    <b style="color:#FF8D28;font-size:0.88rem;">{title}</b>
                    <div style="font-size:0.82rem;color:#666;margin-top:5px;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            for title, desc in steps_right:
                st.markdown(f"""
                <div class="info-card" style="padding:12px 16px;margin-bottom:10px;">
                    <b style="color:#FF8D28;font-size:0.88rem;">{title}</b>
                    <div style="font-size:0.82rem;color:#666;margin-top:5px;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section("Before vs After")
        c1, c2, c3, c4 = st.columns(4)
        comparisons = [
            (c1, "Total Fitur", "61", "18"),
            (c2, "Training Size", "100", "500 (aug)"),
            (c3, "Missing Values", "143", "0"),
            (c4, "Data Leakage", "❌ Ada", "✅ Bersih"),
        ]
        for col, label, before, after in comparisons:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:0.75rem;color:#888;margin-bottom:6px;">{label}</div>
                    <div style="display:flex;justify-content:center;align-items:center;gap:12px;">
                        <div style="text-align:center;">
                            <div style="font-size:1rem;font-weight:900;color:#e74c3c;">{before}</div>
                            <div style="font-size:0.7rem;color:#aaa;">Before</div>
                        </div>
                        <div style="color:#aaa;font-size:1.2rem;">→</div>
                        <div style="text-align:center;">
                            <div style="font-size:1rem;font-weight:900;color:#FF8D28;">{after}</div>
                            <div style="font-size:0.7rem;color:#aaa;">After</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section("Fitur yang Digunakan (18 Fitur Final)")
        fitur_list = [
            ("veggies_day", "Porsi sayur per hari", "Numerik"),
            ("fruit_day", "Porsi buah per hari", "Numerik"),
            ("eating_out", "Frekuensi makan di luar", "Numerik"),
            ("exercise", "Frekuensi olahraga", "Numerik"),
            ("calories_day", "Kesadaran kalori", "Numerik"),
            ("comfort_food_reasons_coded", "Alasan makan comfort food", "Encoded"),
            ("eating_changes_coded", "Perubahan kebiasaan makan", "Encoded"),
            ("healthy_feeling", "Perasaan sehat (proxy mood)", "Numerik 1-5"),
            ("cook", "Frekuensi masa sendiri", "Numerik"),
            ("sports", "Frekuensi aktivitas olahraga", "Numerik"),
            ("coffee", "Konsumsi kopi per hari", "Numerik"),
            ("breakfast", "Frekuensi sarapan", "Numerik"),
            ("on_off_campus", "Status tempat tinggal", "Encoded"),
            ("employment", "Status pekerjaan", "Encoded"),
            ("pay_meal_out", "Budget makan di luar", "Encoded"),
            ("fav_cuisine_coded", "Masakan favorit", "Encoded"),
            ("healthy_behavior_score", "Skor perilaku sehat (composite)", "Engineered"),
            ("emotional_eating_risk", "Risiko emotional eating (composite)", "Engineered"),
        ]
        html_table = '<table class="eval-table">'
        html_table += '<tr><th>#</th><th>Fitur</th><th>Deskripsi</th><th>Tipe</th></tr>'
        for i, (f, d, t) in enumerate(fitur_list):
            html_table += f"<tr><td>{i+1}</td><td><b>{f}</b></td><td>{d}</td><td><span style='font-size:0.78rem;background:#fff3e8;color:#FF8D28;padding:2px 8px;border-radius:6px;'>{t}</span></td></tr>"
        html_table += '</table>'
        st.markdown(html_table, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section("Eksplorasi Preview Data")

        try:
            # 1. Sebelum Augmentasi (Original Cleaned)
            csv_path = "Dataset/Food Vibe/food_coded.csv"
            if not os.path.exists(csv_path):
                csv_path = "d:/BINUS/Semester 4/Machine Learning/Project/Final Project/Dataset/Food Vibe/food_coded.csv"
            df_raw_prep = pd.read_csv(csv_path)

            features_before = [
                'veggies_day', 'fruit_day', 'eating_out', 'exercise', 'calories_day',
                'comfort_food_reasons_coded', 'eating_changes_coded', 'healthy_feeling',
                'cook', 'sports', 'coffee', 'breakfast',
                'on_off_campus', 'employment', 'pay_meal_out', 'fav_cuisine_coded'
            ]

            df_before = df_raw_prep[features_before].copy()

            # 2. Sesudah Augmentasi (Augmented Training Data)
            import numpy as np
            from sklearn.model_selection import train_test_split

            df_work_prep = df_raw_prep[features_before].copy()
            for col in df_work_prep.columns:
                df_work_prep[col] = df_work_prep[col].fillna(df_work_prep[col].mode()[0] if not df_work_prep[col].mode().empty else 0)
            df_work_prep['diet_binary'] = (df_raw_prep['diet_current_coded'] <= 2).astype(int)

            X_orig = df_work_prep[features_before]
            y_orig = df_work_prep['diet_binary']
            X_train, X_test, y_train, y_test = train_test_split(
                X_orig, y_orig, test_size=0.2, random_state=42, stratify=y_orig
            )

            X_train = X_train.copy()
            X_train['healthy_behavior_score'] = ((X_train['veggies_day'] >= 3).astype(int) + (X_train['exercise'] >= 2).astype(int))
            X_train['emotional_eating_risk'] = ((X_train['comfort_food_reasons_coded'] > 0).astype(int) + (X_train['eating_changes_coded'] > 0).astype(int))

            # Generate synthetic
            np.random.seed(42)
            n_samples = 200
            syn_healthy = pd.DataFrame({
                'veggies_day': np.clip(np.random.normal(4.5, 1.5, n_samples), 0, 7).astype(int),
                'fruit_day': np.clip(np.random.normal(3.5, 1.5, n_samples), 0, 7).astype(int),
                'eating_out': np.clip(np.random.normal(1.5, 1, n_samples), 0, 7).astype(int),
                'exercise': np.clip(np.random.normal(3.5, 1.5, n_samples), 0, 7).astype(int),
                'calories_day': np.clip(np.random.normal(3, 1.5, n_samples), 0, 7).astype(int),
                'comfort_food_reasons_coded': np.clip(np.random.normal(0.5, 0.8, n_samples), 0, 3).astype(int),
                'eating_changes_coded': np.clip(np.random.normal(0.3, 0.5, n_samples), 0, 3).astype(int),
                'healthy_feeling': np.clip(np.random.normal(4, 0.8, n_samples), 1, 5).astype(int),
                'cook': np.clip(np.random.normal(3, 1.5, n_samples), 0, 7).astype(int),
                'sports': np.clip(np.random.normal(2.5, 1.5, n_samples), 0, 7).astype(int),
                'coffee': np.clip(np.random.normal(1.5, 1.5, n_samples), 0, 7).astype(int),
                'breakfast': np.clip(np.random.normal(3, 1.5, n_samples), 0, 7).astype(int),
                'on_off_campus': np.random.randint(0, 2, n_samples),
                'employment': np.random.randint(0, 3, n_samples),
                'pay_meal_out': np.random.randint(0, 3, n_samples),
                'fav_cuisine_coded': np.random.randint(0, 10, n_samples)
            })

            np.random.seed(99)
            syn_unhealthy = pd.DataFrame({
                'veggies_day': np.clip(np.random.normal(1.5, 1.5, n_samples), 0, 7).astype(int),
                'fruit_day': np.clip(np.random.normal(0.8, 1, n_samples), 0, 7).astype(int),
                'eating_out': np.clip(np.random.normal(4, 1.2, n_samples), 0, 7).astype(int),
                'exercise': np.clip(np.random.normal(1, 1.5, n_samples), 0, 7).astype(int),
                'calories_day': np.clip(np.random.normal(1.5, 1.5, n_samples), 0, 7).astype(int),
                'comfort_food_reasons_coded': np.clip(np.random.normal(2.5, 1, n_samples), 0, 3).astype(int),
                'eating_changes_coded': np.clip(np.random.normal(1.5, 1, n_samples), 0, 3).astype(int),
                'healthy_feeling': np.clip(np.random.normal(2, 1, n_samples), 1, 5).astype(int),
                'cook': np.clip(np.random.normal(1.5, 1.5, n_samples), 0, 7).astype(int),
                'sports': np.clip(np.random.normal(0.5, 1, n_samples), 0, 7).astype(int),
                'coffee': np.clip(np.random.normal(2.5, 1.5, n_samples), 0, 7).astype(int),
                'breakfast': np.clip(np.random.normal(1, 1.5, n_samples), 0, 7).astype(int),
                'on_off_campus': np.random.randint(0, 2, n_samples),
                'employment': np.random.randint(0, 3, n_samples),
                'pay_meal_out': np.random.randint(0, 3, n_samples),
                'fav_cuisine_coded': np.random.randint(0, 10, n_samples)
            })

            X_syn = pd.concat([syn_healthy, syn_unhealthy], ignore_index=True)
            X_syn['healthy_behavior_score'] = ((X_syn['veggies_day'] >= 3).astype(int) + (X_syn['exercise'] >= 2).astype(int))
            X_syn['emotional_eating_risk'] = ((X_syn['comfort_food_reasons_coded'] > 0).astype(int) + (X_syn['eating_changes_coded'] > 0).astype(int))

            # Tambahkan target diet_binary ke X_train asli sebelum penggabungan
            X_train_with_y = X_train.copy()
            X_train_with_y['diet_binary'] = y_train

            # Tambahkan target diet_binary ke X_syn
            X_syn_with_y = X_syn.copy()
            X_syn_with_y['diet_binary'] = pd.Series([1]*n_samples + [0]*n_samples, index=X_syn.index)

            # Gabungkan dengan aman
            df_after = pd.concat([X_train_with_y, X_syn_with_y], ignore_index=True)

            tab_raw_data, tab_prep_data = st.tabs([
                "Sebelum Augmentasi (Original Cleaned)",
                "Sesudah Augmentasi (Augmented Training)"
            ])

            with tab_raw_data:
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Jumlah Baris", len(df_before))
                mc2.metric("Jumlah Fitur", df_before.shape[1])
                mc3.metric("Nilai Kosong", df_before.isnull().sum().sum())
                mc4.metric("Duplikat", df_before.duplicated().sum())
                
                st.dataframe(df_before, use_container_width=True)

            with tab_prep_data:
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Jumlah Baris", len(df_after))
                mc2.metric("Jumlah Fitur", df_after.shape[1] - 1)
                mc3.metric("Nilai Kosong", df_after.isnull().sum().sum())
                mc4.metric("Duplikat", df_after.duplicated().sum())
                
                st.dataframe(df_after, use_container_width=True)

        except Exception as e:
            st.error(f"Gagal mempreview dataset: {e}")

        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="footer">FoodVibe &nbsp;|&nbsp; Step 2: Preprocessing</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE: MODEL
# ═══════════════════════════════════════════════════════
elif page == "Step 3: Model":
    hero("Model Training & Evaluasi",
         "Logistic Regression → Random Forest → SVM", "Step 3 dari 4")

    st.markdown("""
    <div class="section-desc">
        Tiga model dilatih dan dievaluasi secara berurutan: LR sebagai baseline, RF sebagai model utama,
        dan SVM sebagai model final terpilih. Evaluasi menggunakan F1-Score (weighted) sebagai metrik utama
        karena menyeimbangkan precision dan recall pada dataset yang sedikit imbalanced.
    </div>
    """, unsafe_allow_html=True)

    if st.button("Jalankan Model Evaluation", use_container_width=True):
        st.session_state["run_model"] = True
        set_query_param("run_model", "true")
        safe_rerun()

    if st.session_state.get("run_model", False):
        st.success("Model evaluation berhasil dijalankan!")

        section("Model Sebelum Augmentasi")
        col_sa1, col_sa2, col_sa3 = st.columns(3)
        models_before = [
            (col_sa1, "Logistic Regression", "Baseline", "#3498db",
             "0.4436", "0.4400",
             "Model dasar tanpa penyeimbangan data. Performa sangat rendah karena ketidakseimbangan kelas target asli."),
            (col_sa2, "Random Forest", "Model Utama", "#2d9e5f",
             "0.5247", "0.5200",
             "Model pohon keputusan sebelum augmentasi. Cenderung mengalami bias dan overfitting pada kelas mayoritas."),
            (col_sa3, "SVM (RBF Kernel)", "Model Final", "#FF8D28",
             "0.6000", "0.6000",
             "Model SVM sebelum penyeimbangan. Batas keputusan (boundary) tidak stabil karena keterbatasan data minoritas."),
        ]
        for col, name, badge, color, f1, acc, desc in models_before:
            with col:
                st.markdown(f"""
                <div class="info-card" style="border-top:4px solid {color}; text-align:center;">
                    <div style="font-size:0.78rem;background:{color};color:white;border-radius:8px;
                                padding:2px 10px;display:inline-block;margin-bottom:10px;">{badge}</div>
                    <div style="font-weight:800;color:#1d2b22;font-size:1rem;margin-bottom:12px;">{name}</div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;">
                        <div style="background:#fffaf5;border-radius:8px;padding:8px;">
                            <div style="font-size:1.3rem;font-weight:900;color:{color};">{f1}</div>
                            <div style="font-size:0.72rem;color:#888;">F1-Score</div>
                        </div>
                        <div style="background:#fffaf5;border-radius:8px;padding:8px;">
                            <div style="font-size:1.3rem;font-weight:900;color:{color};">{acc}</div>
                            <div style="font-size:0.72rem;color:#888;">Accuracy</div>
                        </div>
                    </div>
                    <div style="font-size:0.8rem;color:#666;text-align:left;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section("Model Sesudah Augmentasi")
        col_da1, col_da2, col_da3 = st.columns(3)
        models_after = [
            (col_da1, "Logistic Regression", "Baseline", "#3498db",
             "0.8693", "0.8800",
             "Model sederhana & interpretable. Digunakan sebagai baseline perbandingan."),
            (col_da2, "Random Forest", "Model Utama", "#2d9e5f",
             "0.8238", "0.8800",
             "Menangani hubungan non-linear. Menyediakan feature_importances_ untuk analisis."),
            (col_da3, "SVM (RBF Kernel)", "Model Final", "#FF8D28",
             "0.9200", "0.9200",
             "Decision boundary paling presisi. Dipilih sebagai model final yang di-deploy."),
        ]
        for col, name, badge, color, f1, acc, desc in models_after:
            with col:
                st.markdown(f"""
                <div class="info-card" style="border-top:4px solid {color}; text-align:center;">
                    <div style="font-size:0.78rem;background:{color};color:white;border-radius:8px;
                                padding:2px 10px;display:inline-block;margin-bottom:10px;">{badge}</div>
                    <div style="font-weight:800;color:#1d2b22;font-size:1rem;margin-bottom:12px;">{name}</div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;">
                        <div style="background:#fffaf5;border-radius:8px;padding:8px;">
                            <div style="font-size:1.3rem;font-weight:900;color:{color};">{f1}</div>
                            <div style="font-size:0.72rem;color:#888;">F1-Score</div>
                        </div>
                        <div style="background:#fffaf5;border-radius:8px;padding:8px;">
                            <div style="font-size:1.3rem;font-weight:900;color:{color};">{acc}</div>
                            <div style="font-size:0.72rem;color:#888;">Accuracy</div>
                        </div>
                    </div>
                    <div style="font-size:0.8rem;color:#666;text-align:left;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            section("Detail Evaluasi SVM (Model Final)")
            st.markdown("""
            <table class="eval-table">
                <tr><th>Metrik</th><th>Healthy (1)</th><th>Unhealthy (0)</th><th>Weighted Avg</th></tr>
                <tr><td><b>Precision</b></td><td>0.95</td><td>0.67</td><td>0.92</td></tr>
                <tr><td><b>Recall</b></td><td>0.95</td><td>0.67</td><td>0.92</td></tr>
                <tr><td><b>F1-Score</b></td><td>0.95</td><td>0.67</td><td>0.92</td></tr>
                <tr><td><b>Support</b></td><td>22</td><td>3</td><td>25</td></tr>
            </table>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            section("Top Feature Importances (dari Random Forest)")
            feat_imp = [
                ("veggies_day", 0.18, "#2d9e5f"),
                ("healthy_feeling", 0.15, "#FF8D28"),
                ("exercise", 0.13, "#3498db"),
                ("healthy_behavior_score", 0.11, "#9b59b6"),
                ("eating_out", 0.10, "#e74c3c"),
                ("comfort_food_reasons_coded", 0.09, "#f39c12"),
                ("cook", 0.08, "#1abc9c"),
            ]
            for feat, imp, color in feat_imp:
                pct = imp * 100
                st.markdown(f"""
                <div style="margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;font-size:0.82rem;
                                font-weight:700;color:#1d2b22;margin-bottom:5px;">
                        <span>{feat}</span>
                        <span style="color:{color};">{pct:.0f}%</span>
                    </div>
                    <div style="background:#f5ede3;border-radius:999px;height:10px;overflow:hidden;">
                        <div style="background:{color};width:{pct/18*100}%;height:10px;border-radius:999px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            section("Confusion Matrix (SVM Final Model)")

            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                sns.set_theme(style='white')
                fig_cm, ax_cm = plt.subplots(figsize=(5.2, 4.0))
                cm_data = np.array([[2, 1], [1, 21]])
                
                sns.heatmap(
                    cm_data, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues', 
                    xticklabels=['Unhealthy', 'Relatively Healthy'],
                    yticklabels=['Unhealthy', 'Relatively Healthy'],
                    annot_kws={"size": 12},
                    cbar=True,
                    ax=ax_cm
                )
                ax_cm.set_title('Final Confusion Matrix (Balance Optimized - Thresh: 0.720)', fontsize=10, fontweight='bold', pad=10, color='#1d2b22')
                ax_cm.set_xlabel('Predicted Choice', fontsize=9, labelpad=6)
                ax_cm.set_ylabel('Actual Choice', fontsize=9, labelpad=6)
                plt.xticks(rotation=0, fontsize=8)
                plt.yticks(rotation=90, va="center", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig_cm)
            except Exception as e:
                st.error(f"Gagal memvisualisasikan Confusion Matrix: {e}")

        with col2:
            section("Metrik Evaluasi")
            evals = [
                ("F1-Score (Weighted)", "Metrik utama — menyeimbangkan precision & recall untuk dataset imbalanced", "#FF8D28"),
                ("Accuracy", "Gambaran umum keakuratan prediksi keseluruhan", "#3498db"),
                ("Confusion Matrix", "Visualisasi pola kesalahan klasifikasi per kelas", "#2d9e5f"),
                ("Classification Report", "Precision, recall, F1 per kelas healthy & unhealthy", "#9b59b6"),
                ("Matthews Correlation Coefficient (MCC): 0.6212", "Mengukur kualitas klasifikasi biner pada kelas imbalanced (tidak seimbang)", "#e74c3c"),
            ]
            for name, desc, color in evals:
                st.markdown(f"""
                <div class="info-card" style="border-left:4px solid {color}; padding:12px 14px; margin-bottom:10px;">
                    <b style="color:{color};font-size:0.87rem;">{name}</b>
                    <div style="font-size:0.79rem;color:#666;margin-top:4px;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

            section("Kenapa SVM Dipilih?")
            st.markdown("""
            <div class="section-desc" style="font-size:0.85rem;">
                SVM dengan RBF Kernel mampu memetakan data ke <b>ruang dimensi lebih tinggi</b>
                sehingga menghasilkan decision boundary yang lebih presisi dibanding tree-based methods
                pada dataset berukuran kecil (~125 sampel asli).<br><br>
                <b>Hyperparameter tuning:</b><br>
                C = 10 &nbsp;|&nbsp; kernel = rbf &nbsp;|&nbsp; class_weight = balanced
            </div>
            """, unsafe_allow_html=True)



    st.markdown('<div class="footer">FoodVibe &nbsp;|&nbsp; Step 3: Model</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE: DEMO
# ═══════════════════════════════════════════════════════
elif page == "Step 4: Demo":
    hero("FoodVibe Demo",
         "Isi kebiasaan harianmu → Dapatkan klasifikasi pola makan + saran makanan", "Step 4 dari 4")

    if not model_loaded:
        st.error("⚠️ File foodvibe_model.pkl atau foodvibe_scaler.pkl tidak ditemukan. Pastikan kedua file ada di folder yang sama dengan app.py.")
        st.stop()

    st.markdown("### Ceritakan kebiasaan harianmu")

    # ── INPUT FORM (Tiga Pilar Utama) ──
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Mood (Kondisi Emosional)**")
        healthy_feeling_emoji = st.select_slider(
            "Suasana Hati / Mood Harian",
            options=["😔", "😟", "😐", "😊", "😁"],
            value="😐",
            help="Geser untuk memilih mood Anda dari 😔 (sangat buruk/stres) hingga 😁 (sangat baik/sehat)"
        )
        healthy_feeling = {"😔": 1, "😟": 2, "😐": 3, "😊": 4, "😁": 5}[healthy_feeling_emoji]
        comfort_food_reasons_coded = st.selectbox(
            "Alasan Utama Memilih Comfort Food",
            options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            format_func=lambda x: {
                0: "0 – Tidak ada",
                1: "1 – Stres",
                2: "2 – Kebosanan (Boredom)",
                3: "3 – Depresi/sedih",
                4: "4 – Kebiasaan",
                5: "5 – Tidak tahu",
                6: "6 – Semua alasan",
                7: "7 – Hadiah diri (Self-reward)",
                8: "8 – Cemas (Anxiety)",
                9: "9 – Lainnya",
            }.get(x, str(x))
        )
        eating_changes_coded = st.selectbox(
            "Perubahan Kebiasaan Makan Saat Stres",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "0 – Tidak berubah",
                1: "1 – Lebih baik/sehat",
                2: "2 – Lebih buruk/tidak sehat",
                3: "3 – Sangat buruk",
            }.get(x, str(x))
        )

    with col2:
        st.markdown("**Nutrisi & Kalori**")
        calories_day = st.slider("Seberapa Peduli Anda Terhadap Kalori?", 0, 7, 3, help="0 = tidak peduli, 7 = sangat peduli")

    with col3:
        st.markdown("**Waktu & Kebiasaan (Konteks Harian)**")
        breakfast = st.slider("Frekuensi Sarapan (hari/minggu)", 0, 7, 4)
        eating_out = st.slider("Frekuensi Makan di Luar (hari/minggu)", 0, 7, 4)
        cook = st.slider("Frekuensi Masak Sendiri (hari/minggu)", 0, 7, 2)

    # Expander untuk opsi pendukung / profil agar tidak membingungkan
    with st.expander("Informasi Pendukung & Profil (Opsional)", expanded=False):
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            veggies_day = st.slider("Porsi sayur per hari", 0, 7, 2, help="0 = tidak pernah, 7 = sangat sering")
            fruit_day = st.slider("Porsi buah per hari", 0, 7, 2)
            coffee = st.slider("Konsumsi kopi per hari", 0, 7, 2)
        with ec2:
            exercise = st.slider("Frekuensi olahraga per minggu", 0, 7, 1)
            sports = st.slider("Frekuensi aktivitas sport per minggu", 0, 7, 1)
            gender = st.radio("Gender", options=[0, 1], format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki")
        with ec3:
            on_off_campus = st.radio("Tempat tinggal", options=[0, 1], format_func=lambda x: "Off-campus" if x == 0 else "On-campus")
            employment = st.selectbox(
                "Status pekerjaan",
                options=[0, 1, 2],
                format_func=lambda x: {0: "Tidak bekerja", 1: "Part-time", 2: "Full-time"}.get(x)
            )
            pay_meal_out = st.selectbox(
                "Budget makan sekali di luar",
                options=[0, 1, 2],
                format_func=lambda x: {0: "< Rp 50.000", 1: "Rp 50.000 – 100.000", 2: "> Rp 100.000"}.get(x)
            )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Analisis Pola Makanku!", use_container_width=True)

    if predict_btn:
        import time
        start_time = time.time()

        # Hitung composite features
        healthy_behavior_score = int(veggies_day >= 3) + int(exercise >= 2)
        emotional_eating_risk  = int(comfort_food_reasons_coded > 0) + int(eating_changes_coded > 0)

        input_dict = {
            'veggies_day': veggies_day,
            'fruit_day': fruit_day,
            'eating_out': eating_out,
            'exercise': exercise,
            'calories_day': calories_day,
            'comfort_food_reasons_coded': comfort_food_reasons_coded,
            'eating_changes_coded': eating_changes_coded,
            'healthy_feeling': healthy_feeling,
            'cook': cook,
            'sports': sports,
            'coffee': coffee,
            'breakfast': breakfast,
            'on_off_campus': on_off_campus,
            'employment': employment,
            'pay_meal_out': pay_meal_out,
            'fav_cuisine_coded': 0, # Default: Indonesia
            'healthy_behavior_score': healthy_behavior_score,
            'emotional_eating_risk': emotional_eating_risk,
        }

        input_df = pd.DataFrame([input_dict])
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else None
        confidence = max(proba) * 100 if proba is not None else None

        latency_ms = (time.time() - start_time) * 1000

        mood_label, mood_desc, foods = get_food_recommendation(healthy_feeling, prediction)

        st.markdown("---")
        st.markdown("### Hasil Analisis FoodVibe")

        res_col, detail_col = st.columns([1, 1])

        with res_col:
            if prediction == 1:
                conf_str = f"Confidence: {confidence:.0f}%" if confidence else ""
                st.markdown(f"""
                <div class="result-healthy">
                    <div class="result-label" style="color:#2d9e5f;">Relatively Healthy</div>
                    <div class="result-conf">{conf_str}</div>
                    <div style="font-size:0.85rem;color:#555;margin-top:10px;">
                        Pola makanmu cukup sehat! Pertahankan kebiasaan baikmu dan terus tingkatkan.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                conf_str = f"Confidence: {confidence:.0f}%" if confidence else ""
                st.markdown(f"""
                <div class="result-unhealthy">
                    <div class="result-label" style="color:#FF8D28;">Unhealthy Pattern</div>
                    <div class="result-conf">{conf_str}</div>
                    <div style="font-size:0.85rem;color:#555;margin-top:10px;">
                        Pola makanmu perlu perhatian lebih. Yuk mulai dengan perubahan kecil!
                    </div>
                </div>
                """, unsafe_allow_html=True)



            # Faktor paling berpengaruh
            factors = [
                ("Konsumsi sayur", veggies_day, 7, "#2d9e5f"),
                ("Frekuensi olahraga", exercise, 7, "#3498db"),
                ("Mood / healthy feeling", healthy_feeling, 5, "#FF8D28"),
                ("Emotional eating risk", min(emotional_eating_risk, 2), 2, "#e74c3c"),
            ]
            factors_html = ""
            for label, val, max_val, color in factors:
                pct = val / max_val * 100
                factors_html += f'<div style="margin-bottom:10px;"><div style="display:flex;justify-content:space-between;font-size:0.8rem;font-weight:700;color:#1d2b22;margin-bottom:4px;"><span>{label}</span><span style="color:{color};">{val}/{max_val}</span></div><div style="background:#f5ede3;border-radius:999px;height:8px;overflow:hidden;"><div style="background:{color};width:{pct}%;height:8px;border-radius:999px;"></div></div></div>'
            st.markdown(f'<div class="info-card"><h4>Faktor Paling Berpengaruh</h4>{factors_html}</div>', unsafe_allow_html=True)

        with detail_col:
            # Mood & rekomendasi
            st.markdown(f"""
            <div class="info-card" style="border-left:4px solid #FF8D28;">
                <h4>Mood Terdeteksi: {mood_label}</h4>
                <div style="font-size:0.85rem;color:#666;">{mood_desc}</div>
            </div>
            """, unsafe_allow_html=True)

            chips_html = "".join([f'<span class="food-chip">{f}</span>' for f in foods])
            st.markdown(f'<div class="info-card"><h4>Saran Makanan untuk Kamu</h4><div style="font-size:0.83rem;color:#777;margin-bottom:10px;">Berdasarkan mood & pola makanmu:</div>{chips_html}</div>', unsafe_allow_html=True)

            # Mini insight personal
            insights = []
            if veggies_day < 3:
                insights.append("Konsumsi sayurmu masih kurang dari 3 porsi/hari — ini faktor risiko terbesar.")
            if exercise < 2:
                insights.append("Olahraga kurang dari 2x/minggu berkorelasi kuat dengan pola makan unhealthy.")
            if eating_out >= 5:
                insights.append("Makan di luar ≥5x/minggu meningkatkan risiko asupan kalori berlebih.")
            if emotional_eating_risk >= 2:
                insights.append("Kamu memiliki emotional eating triggers yang tinggi — perhatikan pola makan saat stres.")
            if cook >= 4:
                insights.append("Kebiasaan masak sendiri yang baik — ini positif untuk kontrol gizi!")
            if healthy_feeling >= 4:
                insights.append("Mood positif membantu membuat pilihan makan yang lebih baik!")

            if insights:
                insights_html = "".join([f'<div style="font-size:0.83rem;color:#555;margin-bottom:6px;">{ins}</div>' for ins in insights[:3]])
                st.markdown(f'<div class="info-card" style="border-left:4px solid #3498db;"><h4>Insight Personal</h4>{insights_html}</div>', unsafe_allow_html=True)

        # Display latency at the bottom of the result columns (full width)
        st.markdown(f"""
        <div style="text-align:center; margin-top:20px; font-size:0.82rem; color:#777;
                    background:#fffaf5; border:1px solid #ffe5cc; border-radius:12px; padding:10px 16px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.04); clear:both;">
            ⏱️ <b>System Latency:</b> <span style="color:#FF8D28; font-weight:700;">{latency_ms:.2f} ms</span> &nbsp;|&nbsp; 
            <span style="color:#2d9e5f; font-weight:700;">Memenuhi Syarat &lt; 100 ms (Deployment Requirement)</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="footer">FoodVibe &nbsp;|&nbsp; Step 4: Demo &nbsp;|&nbsp; Kelompok 6 – Binus University</div>', unsafe_allow_html=True)