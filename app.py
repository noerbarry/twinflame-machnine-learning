import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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

    # Preprocessing: one-hot encode variabel kategorikal
    categorical_features = ['Jenis_Kelamin', 'Minat', 'Nilai_Pribadi', 'MBTI']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder())
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ])

    # Inisialisasi dan latih model
    rf_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    rf_model.fit(X, y_tf)

    return rf_model

# Fungsi untuk memprediksi kemungkinan twin flame berdasarkan input pengguna
def predict_twin_flame(rf_model, user_data):
    # Prediksi kemungkinan twin flame
    pred_tf = rf_model.predict_proba(user_data)[:, 1]
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
    jenis_kelamin = st.radio("Jenis Kelamin", ['Laki-laki', 'Perempuan'])
    minat = st.radio("Minat", ['Olahraga', 'Seni'])
    nilai_pribadi = st.radio("Nilai Pribadi", ['Baik', 'Buruk'])
    mbti = st.radio("MBTI", ['INTJ', 'INFP', 'ENTJ', 'ENFP', 'INFJ', 'ISTP', 'ESTP', 'ISFJ'])

    # Transformasi input pengguna ke dalam format yang dapat diproses oleh model
    user_data = pd.DataFrame({
        'Usia': [usia],
        'Jenis_Kelamin': [jenis_kelamin],
        'Minat': [minat],
        'Nilai_Pribadi': [nilai_pribadi],
        'MBTI': [mbti]
    })

    # Tombol untuk melakukan prediksi
    if st.button("Prediksi Kemungkinan Twin Flame"):
        pred_tf = predict_twin_flame(rf_model, user_data)
        st.write("Perkiraan kemungkinan Twin Flame:", pred_tf[0])

if __name__ == "__main__":
    main()
