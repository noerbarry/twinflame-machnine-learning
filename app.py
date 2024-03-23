import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Fungsi untuk memuat data tentang pengguna dan kemungkinan twin flame
# Fungsi untuk memuat data tentang pengguna dan kemungkinan twin flame
def load_data():
    data = {
        'Usia': [25, 30, 28, 35, 23, 40, 22, 27],
        'Jenis_Kelamin': ['Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan'],
        'Minat': ['Olahraga', 'Seni', 'Seni', 'Olahraga', 'Seni', 'Olahraga', 'Olahraga', 'Seni'],
        'Nilai_Pribadi': ['Baik', 'Baik', 'Baik', 'Baik', 'Buruk', 'Buruk', 'Buruk', 'Baik'],
        'MBTI': ['INFP', 'INTJ', 'ENFP', 'INFJ', 'ENTP', 'ISFJ', 'ISTP', 'ESTJ'],
        'Twin_Flame': [1, 0, 1, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    return df

# Fungsi untuk prediksi kemungkinan twin flame dan twin flame kembarannya
def predict_twin_flame(model, user_data):
    label_encoder = LabelEncoder()
    user_data['Jenis_Kelamin'] = label_encoder.fit_transform([user_data['Jenis_Kelamin']])
    user_data['Minat'] = label_encoder.fit_transform([user_data['Minat']])
    user_data['Nilai_Pribadi'] = label_encoder.fit_transform([user_data['Nilai_Pribadi']])
    
    # Encode MBTI
    user_data['MBTI'] = user_data['MBTI'].astype('category')
    user_data['MBTI_encoded'] = user_data['MBTI'].cat.codes
    
    user_df = pd.DataFrame(user_data)

    prediction = model.predict_proba(user_df)
    return prediction[0][1]

def main():
    st.title("Deteksi Kemungkinan Twin Flame")

    # Load data
    df = load_data()

    # Train model
    rf_model = train_model(df)

    # Input data pengguna
    st.subheader("Masukkan data pengguna:")
    usia = st.slider("Usia", min_value=1, max_value=100, value=30)
    jenis_kelamin = st.selectbox("Jenis Kelamin", ['Laki-laki', 'Perempuan'])
    minat = st.selectbox("Minat", ['Olahraga', 'Seni'])
    nilai_pribadi = st.selectbox("Nilai Pribadi", ['Baik', 'Buruk'])
    mbti = st.text_input("MBTI (contoh: INFP)")

    user_data = {'Usia': usia, 'Jenis_Kelamin': jenis_kelamin, 'Minat': minat, 'Nilai_Pribadi': nilai_pribadi, 'MBTI': mbti}

    # Tombol untuk melakukan prediksi
    if st.button("Prediksi Kemungkinan Twin Flame"):
        pred_tf = predict_twin_flame(rf_model, user_data)
        interpretation = interpret_prediction(pred_tf)
        st.write("Perkiraan kemungkinan Twin Flame:", pred_tf)
        st.write("Interpretasi:", interpretation)

if __name__ == "__main__":
    main()

