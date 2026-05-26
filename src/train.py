import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Mengimpor fungsi dari file preprocessing.py yang baru kita buat
from preprocessing import load_and_clean_data

def train_model():
    # 1. Atur lokasi file (karena file train.py ada di dalam folder src/)
    data_path = '../data/raw/food_coded.csv'
    model_save_path = '../models/best_model.pkl'
    
    # 2. Ambil data yang sudah bersih menggunakan fungsi tadi
    df = load_and_clean_data(data_path)
    
    # 3. Train-Test Split
    print("[INFO] Membagi data latih dan data uji...")
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    # 4. Melatih Model Utama (Random Forest)
    print("[INFO] Mulai melatih model Random Forest...")
    rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)
    
    # 5. Menyimpan Model
    joblib.dump(rf_model, model_save_path)
    print(f"[SUCCESS] Model berhasil dilatih dan disimpan di: {model_save_path} 🎉")

if __name__ == "__main__":
    train_model()