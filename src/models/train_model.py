import pandas as pd
import numpy as np
import os
import joblib
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

def train_model(input_path, model_output_path):
    print(f"Loading cleaned data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. Menentukan Target Variable
    target_col = 'comfort_food_reasons_coded'
    
    # Pastikan baris yang targetnya kosong dihapus
    df = df.dropna(subset=[target_col])
    
    # 2. Seleksi Fitur Awal (Baseline)
    # Untuk menghindari error karena format teks/string (seperti alasan panjang), 
    # kita membuang kolom bertipe teks 'object' sementara waktu untuk baseline model.
    # Nanti kalau performa kurang, kita bisa lakukan One-Hot Encoding di script ini.
    text_cols = df.select_dtypes(include=['object']).columns
    df = df.drop(columns=text_cols)
    
    # Pisahkan Fitur (X) dan Target (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Label Encoding untuk target (kalau nilainya belum berupa angka urut)
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # 3. Train-Test Split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Feature Scaling (Penting untuk algoritma seperti SVM & Logistic Regression)
    # Mengisi nilai yang mungkin masih terlewat kosong dengan angka 0
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Model Training (Melatih 3 Algoritma Classical ML)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='linear', random_state=42)
    }
    
    best_model = None
    best_acc = 0
    best_model_name = ""
    
    print("\n--- Training Models ---")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        print(f"{name} Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_model_name = name
            
    print(f"\nModel terbaik adalah {best_model_name} dengan Akurasi {best_acc:.4f}")
    
    # 6. Menyimpan Model Terbaik
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    
    # Simpan model, scaler, label_encoder, dan nama fitur agar bisa dipakai di Streamlit nanti
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'label_encoder': le,
        'features': X.columns.tolist()
    }
    
    joblib.dump(model_data, model_output_path)
    print(f"\nModel dan Scaler berhasil disimpan di {model_output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Melatih model Machine Learning FoodVibe.')
    parser.add_argument('--input', type=str, default='data/processed/food_coded_cleaned.csv', help='Path ke data bersih')
    parser.add_argument('--output', type=str, default='models/best_model.pkl', help='Path untuk menyimpan model')
    
    args = parser.parse_args()
    train_model(args.input, args.output)
