"""
Main Streamlit application entry point.
"""
import streamlit as st

def main():
    st.set_page_config(page_title="FoodVibe", page_icon="🍔", layout="wide")
    
    st.title("FoodVibe 🍔")
    st.subheader("Klasifikasi Perilaku Pemilihan Makanan Mahasiswa")
    
    st.write("Selamat datang di aplikasi FoodVibe! Pengembangan sedang dalam proses.")
    st.info("Navigasi fitur akan tersedia di sidebar pada tahap pengembangan selanjutnya.")

if __name__ == "__main__":
    main()
