import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

# Fungsi untuk melatih model ensemble
def train_model(df):
    X = df[['Usia', 'Jenis_Kelamin', 'Minat', 'Nilai_Pribadi', 'MBTI']]  # Fitur-fitur
    y_tf = df['Twin_Flame']  # Target Twin Flame

    # One-hot encode variabel kategorikal
    X_encoded = pd.get_dummies(X)

    # Bagi data menjadi data latih dan data uji untuk twin flame
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_tf, test_size=0.2, random_state=42)

    # Inisialisasi model ensemble (Random Forest)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Latih model twin flame
    rf_model.fit(X_train, y_train)

    # Evaluasi model twin flame
    accuracy_rf = accuracy_score(y_test, rf_model.predict(X_test))
    st.write("Akurasi Model Twin Flame (Random Forest):", accuracy_rf)

    return rf_model

# Fungsi untuk memprediksi kemungkinan twin flame berdasarkan input pengguna
def predict_twin_flame(rf_model, user_data):
    # One-hot encode input pengguna
    user_encoded = pd.get_dummies(user_data)

    # Lakukan prediksi menggunakan model
    pred_tf = rf_model.predict_proba(user_encoded)[0][1]
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
    mbti_types = ['INTJ', 'INTP', 'ENTJ', 'ENTP', 'INFJ', 'INFP', 'ENFJ', 'ENFP', 'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ', 'ISTP', 'ISFP', 'ESTP', 'ESFP']
    selected_mbti = st.selectbox("Pilih Jenis Kepribadian MBTI", mbti_types)

    # Tombol untuk melakukan prediksi
    if st.button("Prediksi Kemungkinan Twin Flame"):
        user_data = {'Usia': usia, 
                     'Jenis_Kelamin': jenis_kelamin, 
                     'Minat': minat, 
                     'Nilai_Pribadi': nilai_pribadi, 
                     'MBTI': selected_mbti}
        rf_model = train_model(df)
        pred_tf = predict_twin_flame(rf_model, user_data)
        st.write("Perkiraan kemungkinan Twin Flame:", pred_tf)

if __name__ == "__main__":
    df = load_data()
    main()
