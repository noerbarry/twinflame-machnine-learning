import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Fungsi untuk memuat data tentang pengguna dan kemungkinan twin flame
def load_data():
    data = {
        'Usia': [25, 30, 28, 35, 23, 40, 22, 27],
        'Jenis_Kelamin': ['Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan'],
        'Minat': ['Olahraga', 'Seni', 'Seni', 'Olahraga', 'Seni', 'Olahraga', 'Olahraga', 'Seni'],
        'Nilai_Pribadi': ['Baik', 'Baik', 'Baik', 'Baik', 'Buruk', 'Buruk', 'Buruk', 'Baik'],
        'MBTI': ['INFJ', 'ENTP', 'INTJ', 'INTP', 'ENFJ', 'INFP', 'ENFP', 'INTJ'],
        'Twin_Flame': [1, 0, 1, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    return df

# Fungsi untuk melatih model
def train_model(df):
    X = df[['Usia', 'Jenis_Kelamin', 'Minat', 'Nilai_Pribadi', 'MBTI']]
    y = df['Twin_Flame']
    
    # Encode variabel kategorikal
    label_encoder = LabelEncoder()
    X['Jenis_Kelamin'] = label_encoder.fit_transform(X['Jenis_Kelamin'])
    X['Minat'] = label_encoder.fit_transform(X['Minat'])
    X['Nilai_Pribadi'] = label_encoder.fit_transform(X['Nilai_Pribadi'])
    X['MBTI'] = label_encoder.fit_transform(X['MBTI'])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inisialisasi dan pelatihan model
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    return rf_model

# Fungsi untuk memprediksi kemungkinan twin flame berdasarkan input pengguna
def predict_twin_flame(model, user_data):
    label_encoder = LabelEncoder()
    user_data['Jenis_Kelamin'] = label_encoder.transform([user_data['Jenis_Kelamin']])
    user_data['Minat'] = label_encoder.transform([user_data['Minat']])
    user_data['Nilai_Pribadi'] = label_encoder.transform([user_data['Nilai_Pribadi']])
    user_data['MBTI'] = label_encoder.transform([user_data['MBTI']])

    user_df = pd.DataFrame(user_data)

    prediction = model.predict_proba(user_df)
    return prediction[0][1]

# Fungsi untuk interpretasi hasil prediksi
def interpret_prediction(probability):
    if probability >= 0.8:
        return "Anda memiliki kemungkinan tinggi untuk memiliki Twin Flame. Anda mungkin merasa kuat tarik menarik secara spiritual dan emosional."
    elif probability >= 0.5:
        return "Anda memiliki kemungkinan moderat untuk memiliki Twin Flame. Mungkin ada hubungan yang dalam dan bermakna dalam hidup Anda."
    else:
        return "Anda memiliki kemungkinan rendah untuk memiliki Twin Flame. Tetapi itu tidak berarti Anda tidak akan menemukan hubungan yang istimewa di masa depan."

def main():
    st.title("Deteksi Kemungkinan Twin Flame")

    # Load data
    df = load_data()

    # Train model
    rf_model = train_model(df)

    # Input data pengguna
    st.subheader("Masukkan data pengguna:")
    usia = st.slider("Usia", min_value=1, max_value=100, value=30)
    jenis_kelamin = st.radio("Jenis Kelamin", ['Laki-laki', 'Perempuan'])
    minat = st.selectbox("Minat", ['Olahraga', 'Seni', 'Musik', 'Kuliner', 'Pendidikan', 'Teknologi'])
    nilai_pribadi = st.radio("Nilai Pribadi", ['Baik', 'Buruk'])
    mbti = st.selectbox("MBTI", ['INFJ', 'ENTP', 'INTJ', 'INTP', 'ENFJ', 'INFP', 'ENFP', 'INTJ'])

    user_data = {'Usia': usia, 'Jenis_Kelamin': jenis_kelamin, 'Minat': minat, 'Nilai_Pribadi': nilai_pribadi, 'MBTI': mbti}

    # Tombol untuk melakukan prediksi
    if st.button("Prediksi Kemungkinan Twin Flame"):
        pred_tf = predict_twin_flame(rf_model, user_data)
        interpretation = interpret_prediction(pred_tf)
        st.write("Perkiraan kemungkinan Twin Flame:", pred_tf)
        st.write("Interpretasi:", interpretation)

if __name__ == "__main__":
    main()
