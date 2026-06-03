import pandas as pd

def load_and_clean_data(filepath):
    """Fungsi untuk memuat dan membersihkan dataset Food Choices"""
    print(f"[INFO] Membaca data dari {filepath}...")
    df = pd.read_csv(filepath)
    
    # 1. Feature Selection
    selected_features = [
        'comfort_food_reasons_coded',
        'eating_out',
        'fries',
        'nutritional_check',
        'diet_current_coded'
    ]
    df_selected = df[selected_features].copy()
    
    # 2. Data Cleaning (Imputasi Modus)
    print("[INFO] Melakukan imputasi nilai kosong...")
    for col in df_selected.columns:
        if df_selected[col].isnull().sum() > 0:
            mode_value = df_selected[col].mode()[0]
            df_selected[col] = df_selected[col].fillna(mode_value)
            
    # 3. Target Mapping (1 = Healthy, 0 = Unhealthy)
    print("[INFO] Melakukan binarisasi target ..")
    # MENGUBAH ATURAN MENJADI <= 2 AGAR DATA SEIMBANG
    df_selected['target'] = df_selected['diet_current_coded'].apply(lambda x: 1 if x <= 2 else 0)
    
    # Hapus target lama yang masih berupa banyak kategori
    df_clean = df_selected.drop(columns=['diet_current_coded'])
    
    return df_clean