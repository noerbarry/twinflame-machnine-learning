import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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

    # Encode variabel kategorikal
    X_encoded = pd.get_dummies(X)

    # Bagi data menjadi data latih dan data uji untuk twin flame
    X_train_tf, _, y_train_tf, _ = train_test_split(X_encoded, y_tf, test_size=0.2, random_state=42)

    # Inisialisasi model ensemble (Random Forest dan Gradient Boosting)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

    # Latih model twin flame
    rf_model.fit(X_train_tf, y_train_tf)
    gb_model.fit(X_train_tf, y_train_tf)

    return rf_model, gb_model

# Fungsi untuk memprediksi kemungkinan twin flame berdasarkan input pengguna
def predict_twin_flame(rf_model, gb_model, user_data):
    # Encode input pengguna
    user_encoded = pd.get_dummies(pd.DataFrame(user_data, index=[0]))

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
    
    # Checkbox untuk Jenis Kelamin
    st.subheader("Jenis Kelamin:")
    jenis_kelamin_laki_laki = st.checkbox("Laki-laki")
    jenis_kelamin_perempuan = st.checkbox("Perempuan")
    
    # Checkbox untuk Minat
    st.subheader("Minat:")
    minat_olahraga = st.checkbox("Olahraga")
    minat_seni = st.checkbox("Seni")
    
    # Checkbox untuk Nilai Pribadi
    st.subheader("Nilai Pribadi:")
    nilai_pribadi_baik = st.checkbox("Baik")
    nilai_pribadi_buruk = st.checkbox("Buruk")

    # Tombol untuk melakukan prediksi
    if st.button("Prediksi Kemungkinan Twin Flame"):
        user_data = {'Usia': usia, 
                     'Jenis_Kelamin_Laki-laki': jenis_kelamin_laki_laki, 
                     'Jenis_Kelamin_Perempuan': jenis_kelamin_perempuan,
                     'Minat_Olahraga': minat_olahraga, 
                     'Minat_Seni': minat_seni, 
                     'Nilai_Pribadi_Baik': nilai_pribadi_baik, 
                     'Nilai_Pribadi_Buruk': nilai_pribadi_buruk}
        rf_model, gb_model = train_model(df)
        pred_tf = predict_twin_flame(rf_model, gb_model, user_data)
        st.write("Perkiraan kemungkinan Twin Flame:", pred_tf)

if __name__ == "__main__":
    df = load_data()
    main()
