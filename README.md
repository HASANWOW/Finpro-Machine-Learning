# FoodVibe

## Deskripsi Singkat
FoodVibe adalah project Machine Learning untuk mengklasifikasikan perilaku pemilihan makanan mahasiswa berdasarkan faktor-faktor seperti mood, taste preference, time/context, lifestyle, dan emotional triggers. Project ini merupakan Final Project untuk mata kuliah Machine Learning BINUS.

## Tujuan Project
Membangun model Classical Machine Learning (Logistic Regression, Random Forest, SVM) untuk menganalisis dan memprediksi kecenderungan pemilihan makanan mahasiswa dengan alur data yang terstruktur. Pipeline mencakup preprocessing, feature engineering, training, evaluasi, hingga deployment berbasis Streamlit.

## Tech Stack
- **Bahasa Pemrograman:** Python
- **Library ML & Data:** pandas, numpy, scikit-learn, joblib
- **Visualisasi:** matplotlib, seaborn
- **Deployment:** Streamlit
- **Versioning:** Git / GitHub

## Struktur Project
Menggunakan clean architecture untuk memisahkan antara data, script proses (src), model (models), analisis (notebooks), dan tampilan (app).

## Cara Menjalankan Project
1. Clone repository ini.
2. Jalankan `setup.bat` (untuk pengguna Windows) untuk mengkonfigurasi virtual environment dan dependencies secara otomatis.
3. Atau jalankan secara manual:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Jalankan aplikasi web Streamlit:
   ```bash
   streamlit run app/streamlit_app.py
   ```
