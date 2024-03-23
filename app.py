import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Fungsi untuk memuat data pengguna dan kemungkinan twin flame
def load_data():
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

# Fungsi untuk melatih model
def train_model(df):
    X = df[['Usia', 'Jenis_Kelamin', 'Minat', 'Nilai_Pribadi']]
    y_tf = df['Twin_Flame']

    # Encoding variabel kategorikal
    X_encoded = pd.get_dummies(X)

    # Split data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_tf, test_size=0.2, random_state=42)

    # Inisialisasi model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

    # Melatih model
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)

    # Evaluasi model
    accuracy_rf = accuracy_score(y_test, rf_model.predict(X_test))
    accuracy_gb = accuracy_score(y_test, gb_model.predict(X_test))
    st.write("Akurasi Model Twin Flame (Random Forest):", accuracy_rf)
    st.write("Akurasi Model Twin Flame (Gradient Boosting):", accuracy_gb)

    return rf_model, gb_model

# Fungsi untuk memprediksi kemungkinan twin flame
def predict_twin_flame(rf_model, gb_model, user_data):
    # Encode input pengguna
    user_encoded = pd.get_dummies(user_data)

    # Lakukan prediksi menggunakan model
    pred_rf = rf_model.predict_proba(user_encoded)[0][1]
    pred_gb = gb_model.predict_proba(user_encoded)[0][1]

    # Ambil rata-rata probabilitas dari kedua model
    pred_tf = (pred_rf + pred_gb) / 2
    return pred_tf

# Fungsi utama aplikasi Streamlit
def main():
    st.title("Deteksi Twin Flame dengan Ensemble Learning")

    # Memuat data dan melatih model saat aplikasi pertama kali dijalankan
    df = load_data()
    rf_model, gb_model = train_model(df)

    # Input data pengguna
    st.subheader("Masukkan data pengguna:")
    usia = st.text_input("Usia")
    jenis_kelamin = st.selectbox("Jenis Kelamin", ['Laki-laki', 'Perempuan'])
    minat = st.selectbox("Minat", ['Olahraga', 'Seni'])
    nilai_pribadi = st.selectbox("Nilai Pribadi", ['Baik', 'Buruk'])

    # Tombol untuk melakukan prediksi
    if st.button("Prediksi Kemungkinan Twin Flame"):
        # Membuat dataframe dari data pengguna
        user_data = pd.DataFrame({'Usia': [usia], 'Jenis_Kelamin': [jenis_kelamin], 'Minat': [minat], 'Nilai_Pribadi': [nilai_pribadi]})
        
        # Prediksi kemungkinan twin flame
        pred_tf = predict_twin_flame(rf_model, gb_model, user_data)
        st.write("Perkiraan kemungkinan Twin Flame:", pred_tf)

if __name__ == "__main__":
    main()
