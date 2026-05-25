import pandas as pd
import os
import argparse

def clean_data(input_path, output_path):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Original shape: {df.shape}")
    
    # 1. Penanganan Missing Values Awal
    # Ini adalah contoh script baseline. Anda dapat memodifikasi logika ini 
    # setelah melihat hasil analisis dari notebooks/01_eda.ipynb
    
    # Mengisi missing value pada kolom numerik dengan nilai median
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    # Mengisi missing value pada kolom kategorikal (object) dengan modus
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    # Catatan: Kolom-kolom teks yang sangat panjang atau tidak terstruktur 
    # mungkin perlu di-drop atau diproses menggunakan teknik NLP, 
    # namun untuk tahap awal kita biarkan atau isi dengan modus.
            
    print(f"Cleaned shape: {df.shape}")
    
    # 2. Menyimpan Dataset yang sudah dibersihkan
    # Memastikan direktori output ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Simpan sebagai CSV baru
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved successfully to {output_path}")

if __name__ == '__main__':
    # Script ini dapat dijalankan melalui terminal
    parser = argparse.ArgumentParser(description='Membersihkan dataset FoodVibe.')
    parser.add_argument('--input', type=str, default='data/raw/food_coded.csv', help='Path ke raw data')
    parser.add_argument('--output', type=str, default='data/processed/food_coded_cleaned.csv', help='Path ke output data')
    
    args = parser.parse_args()
    clean_data(args.input, args.output)
