import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Fungsi untuk memuat data tentang pengguna dan kemungkinan twin flame
def load_data():
    data = {
        'Usia': [25, 30, 28, 35, 23, 40, 22, 27],
        'Jenis_Kelamin': ['Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan'],
        'Minat': ['Olahraga', 'Seni', 'Seni', 'Olahraga', 'Seni', 'Olahraga', 'Olahraga', 'Seni'],
        'Nilai_Pribadi': ['Baik', 'Baik', 'Baik', 'Baik', 'Buruk', 'Buruk', 'Buruk', 'Baik'],
        'Twin_Flame': [1, 0, 1, 0, 1, 0, 1, 0],
        'MBTI': ['INTJ', 'INFP', 'ENTJ', 'ENFP', 'INFJ', 'ISTP', 'ESTP', 'ISFJ']
    }
    df = pd.DataFrame(data)
    return df

# Fungsi untuk melatih model Random Forest
def train_model(df):
    X = df[['Usia', 'Jenis_Kelamin', 'Minat', 'Nilai_Pribadi', 'MBTI']]  # Fitur-fitur
    y_tf = df['Twin_Flame']  # Target Twin Flame

    # One-hot encode variabel kategorikal
    X_encoded = pd.get_dummies(X)

    # Inisialisasi dan latih model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_encoded, y_tf)

    return rf_model

# Fungsi untuk memprediksi kemungkinan twin flame berdasarkan input pengguna
def predict_twin_flame(rf_model, user_data):
    # Encode input pengguna
    user_encoded = pd.get_dummies(user_data)

    # Prediksi kemungkinan twin flame
    pred_tf = rf_model.predict_proba(user_encoded)[0][1]
    return pred_tf

# Fungsi utama aplikasi Streamlit
def main():
    st.title("Deteksi Twin Flame dengan Ensemble Learning")

    # Load data
    df = load_data()

    # Train model
    rf_model = train_model(df)

    # Input data pengguna
    st.subheader("Masukkan data pengguna:")
    usia = st.number_input("Usia", min_value=0, max_value=120, value=30)
    jenis_kelamin = st.selectbox("Jenis Kelamin", ['Laki-laki', 'Perempuan'])
    minat = st.selectbox("Minat", ['Olahraga', 'Seni'])
    nilai_pribadi = st.selectbox("Nilai Pribadi", ['Baik', 'Buruk'])
    mbti = st.selectbox("MBTI", ['INTJ', 'INFP', 'ENTJ', 'ENFP', 'INFJ', 'ISTP', 'ESTP', 'ISFJ'])

    # Tombol untuk melakukan prediksi
    if st.button("Prediksi Kemungkinan Twin Flame"):
        user_data = {'Usia': usia, 'Jenis_Kelamin': jenis_kelamin, 'Minat': minat, 'Nilai_Pribadi': nilai_pribadi, 'MBTI': mbti}
        pred_tf = predict_twin_flame(rf_model, user_data)
        st.write("Perkiraan kemungkinan Twin Flame:", pred_tf)

if __name__ == "__main__":
    main()
