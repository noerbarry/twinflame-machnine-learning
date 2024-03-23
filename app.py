import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Fungsi untuk memuat data tentang pengguna dan kemungkinan twin flame
def load_data():
    # Data sementara, Anda perlu menggantinya dengan data sebenarnya
    data = {
        'Usia': [25, 30, 28, 35, 23, 40, 22, 27],
        'Jenis_Kelamin': ['Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan'],
        'Minat': ['Olahraga', 'Seni', 'Seni', 'Olahraga', 'Seni', 'Olahraga', 'Olahraga', 'Seni'],
        'Nilai_Pribadi': ['Baik', 'Baik', 'Baik', 'Baik', 'Buruk', 'Buruk', 'Buruk', 'Baik'],
        'Twin_Flame': [1, 0, 1, 0, 1, 0, 1, 0],
        'Soulmate': [0, 1, 0, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    return df

# Fungsi untuk melatih model ensemble
def train_model(df):
    X = df[['Usia', 'Jenis_Kelamin', 'Minat', 'Nilai_Pribadi']]  # Fitur-fitur
    y_tf = df['Twin_Flame']  # Target Twin Flame
    y_sm = df['Soulmate']  # Target Soulmate

    # Encode variabel kategorikal
    X_encoded = pd.get_dummies(X)

    # Bagi data menjadi data latih dan data uji untuk twin flame
    X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(X_encoded, y_tf, test_size=0.2, random_state=42)

    # Inisialisasi model ensemble (Random Forest dan Gradient Boosting)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

    # Latih model twin flame
    rf_model.fit(X_train_tf, y_train_tf)
    gb_model.fit(X_train_tf, y_train_tf)

    # Evaluasi model twin flame
    accuracy_rf = accuracy_score(y_test_tf, rf_model.predict(X_test_tf))
    accuracy_gb = accuracy_score(y_test_tf, gb_model.predict(X_test_tf))
    st.write("Akurasi Model Twin Flame (Random Forest):", accuracy_rf)
    st.write("Akurasi Model Twin Flame (Gradient Boosting):", accuracy_gb)

    return rf_model, gb_model, X_encoded

# Fungsi untuk memprediksi kemungkinan twin flame berdasarkan input pengguna
def predict_twin_flame(rf_model, gb_model, user_data, X_encoded):
    # Encode input pengguna
    user_encoded = pd.get_dummies(user_data)

    # Periksa apakah fitur yang dimasukkan cocok dengan fitur yang digunakan saat melatih model
    if set(user_encoded.columns) != set(X_encoded.columns):
        missing_features = set(X_encoded.columns) - set(user_encoded.columns)
        st.error(f"Data yang dimasukkan tidak lengkap, fitur yang hilang: {missing_features}")
        return None

    # Lakukan prediksi menggunakan model
    pred_rf = rf_model.predict_proba(user_encoded)[:, 1]
    pred_gb = gb_model.predict_proba(user_encoded)[:, 1]

    # Ambil rata-rata probabilitas dari kedua model
    pred_tf = (pred_rf + pred_gb) / 2
    return pred_tf

# Fungsi utama aplikasi Streamlit
def main():
    st.title("Deteksi Twin Flame dengan Ensemble Learning")

    # Input data pengguna
    st.subheader("Masukkan data pengguna:")
    usia = st.text_input("Usia")
    jenis_kelamin = st.selectbox("Jenis Kelamin", ['Laki-laki', 'Perempuan'])
    minat = st.selectbox("Minat", ['Olahraga', 'Seni'])
    nilai_pribadi = st.selectbox("Nilai Pribadi", ['Baik', 'Buruk'])

    # Tombol untuk melakukan prediksi
    if st.button("Prediksi Kemungkinan Twin Flame"):
        user_data = {'Usia': usia, 'Jenis_Kelamin': jenis_kelamin, 'Minat': minat, 'Nilai_Pribadi': nilai_pribadi}
        rf_model, gb_model, X_encoded = train_model(df)
        pred_tf = predict_twin_flame(rf_model, gb_model, user_data, X_encoded)
        if pred_tf is not None:
            st.write("Perkiraan kemungkinan Twin Flame:", pred_tf)

if __name__ == "__main__":
    df = load_data()
    main()
