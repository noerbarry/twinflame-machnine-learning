import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Fungsi untuk memuat data tentang MBTI dan kemungkinan twin flame
def load_data():
    # Data sementara, Anda perlu menggantinya dengan data sebenarnya
    data = {
        'MBTI': ['INFJ', 'INTP', 'ENFP', 'ENTJ', 'INFJ', 'INTP', 'ENFP', 'ENTJ'],
        'Usia': [25, 30, 28, 35, 22, 29, 27, 33],
        'Jenis_Kelamin': ['Laki-laki', 'Perempuan', 'Perempuan', 'Laki-laki', 'Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan'],
        'Minat': ['Buku', 'Film', 'Musik', 'Buku', 'Film', 'Musik', 'Buku', 'Film'],
        'Nilai_Pribadi': ['Integritas', 'Kreativitas', 'Kerjasama', 'Kerjasama', 'Integritas', 'Kreativitas', 'Kerjasama', 'Kerjasama'],
        'Twin_Flame': [1, 0, 1, 0, 1, 0, 1, 0],
        'Soulmate': [0, 1, 0, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    return df

# Fungsi untuk melatih model ensemble
def train_ensemble_model(df):
    X = df[['MBTI', 'Usia', 'Jenis_Kelamin', 'Minat', 'Nilai_Pribadi']]  # Fitur-fitur
    y_twin_flame = df['Twin_Flame']  # Target: kemungkinan twin flame (1: Ya, 0: Tidak)
    y_soulmate = df['Soulmate']  # Target: kemungkinan soulmate (1: Ya, 0: Tidak)

    # Lakukan label encoding untuk fitur kategorikal
    X_encoded = pd.get_dummies(X, columns=['MBTI', 'Jenis_Kelamin', 'Minat', 'Nilai_Pribadi'])

    # Bagi data menjadi data latih dan data uji
    X_train, X_test, y_train_tf, y_test_tf = train_test_split(X_encoded, y_twin_flame, test_size=0.2, random_state=42)
    
    # Inisialisasi model ensemble
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

    # Latih model
    rf_model.fit(X_train, y_train_tf)
    gb_model.fit(X_train, y_train_tf)

    return rf_model, gb_model

# Fungsi untuk memprediksi kemungkinan twin flame berdasarkan data pengguna
def predict_twin_flame(rf_model, gb_model, user_data):
    # Lakukan label encoding untuk data pengguna
    user_encoded = pd.get_dummies(user_data)

    # Lakukan prediksi menggunakan kedua model
    pred_rf = rf_model.predict_proba(user_encoded)[0][1]
    pred_gb = gb_model.predict_proba(user_encoded)[0][1]

    # Ambil rata-rata prediksi dari kedua model sebagai prediksi akhir
    avg_pred = (pred_rf + pred_gb) / 2

    return avg_pred

# Fungsi utama aplikasi Streamlit
def main():
    st.title("Deteksi Twin Flame dan Soulmate dengan MBTI menggunakan Ensemble Learning")

    # Muat data
    df = load_data()

    # Latih model
    rf_model, gb_model = train_ensemble_model(df)

    st.write("Masukkan informasi Anda untuk memprediksi kemungkinan Twin Flame.")
    
    # Menerima input pengguna
    mbti_type = st.text_input("Tipe MBTI (misalnya: INFJ, ENFP, dll.):")
    usia = st.number_input("Usia:", min_value=0, max_value=150, value=25)
    jenis_kelamin = st.radio("Jenis Kelamin:", ['Laki-laki', 'Perempuan'])
    minat = st.selectbox("Minat:", ['Buku', 'Film', 'Musik'])
    nilai_pribadi = st.selectbox("Nilai Pribadi:", ['Integritas', 'Kreativitas', 'Kerjasama'])

    # Ketika pengguna menekan tombol prediksi
    if st.button("Prediksi Kemungkinan Twin Flame"):
        # Buat dataframe dari input pengguna
        user_data = pd.DataFrame({
            'MBTI': [mbti_type],
            'Usia': [usia],
            'Jenis_Kelamin': [jenis_kelamin],
            'Minat': [minat],
            'Nilai_Pribadi': [nilai_pribadi]
        })

        # Prediksi kemungkinan Twin Flame
        pred_twin_flame = predict_twin_flame(rf_model, gb_model, user_data)
        
        st.write(f"Perkiraan kemungkinan Twin Flame untuk tipe MBTI {mbti_type}: {pred_twin_flame:.2%}")

if __name__ == "__main__":
    main()
