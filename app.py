import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Remove the "Made in Streamlit" watermark
st.beta_set_page_config(footer="")

# Add your app content here
st.write("Welcome to My Streamlit App!")

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
    user_data['Jenis_Kelamin'] = label_encoder.fit_transform([user_data['Jenis_Kelamin']])
    user_data['Minat'] = label_encoder.fit_transform([user_data['Minat']])
    user_data['Nilai_Pribadi'] = label_encoder.fit_transform([user_data['Nilai_Pribadi']])
    user_data['MBTI'] = label_encoder.fit_transform([user_data['MBTI']])

    user_df = pd.DataFrame(user_data)

    prediction = model.predict_proba(user_df)
    return prediction[0][1]

# Fungsi untuk interpretasi hasil prediksi
def interpret_prediction(probability):
    if probability >= 0.95:
        return "Anda memiliki peluang sangat tinggi untuk memiliki Twin Flame. Hubungan yang sangat dalam dan bermakna mungkin sedang menanti Anda."
    elif probability >= 0.90:
        return "Anda memiliki peluang tinggi untuk memiliki Twin Flame. Mungkin ada tarikan spiritual yang kuat antara Anda dan seseorang yang potensial."
    elif probability >= 0.85:
        return "Anda memiliki peluang yang cukup tinggi untuk memiliki Twin Flame. Ada kemungkinan hubungan yang mendalam sedang dalam proses berkembang."
    elif probability >= 0.80:
        return "Anda memiliki kemungkinan tinggi untuk memiliki Twin Flame. Ini menunjukkan bahwa Anda mungkin merasakan tarikan yang kuat secara spiritual dan emosional, serta memiliki ikatan yang mendalam dengan seseorang yang potensial menjadi Twin Flame Anda."
    elif probability >= 0.75:
        return "Anda memiliki kemungkinan yang cukup tinggi untuk memiliki Twin Flame. Mungkin ada koneksi emosional yang mendalam yang dapat menjadi dasar hubungan yang bermakna."
    elif probability >= 0.70:
        return "Anda memiliki kemungkinan yang cukup besar untuk memiliki Twin Flame. Ada potensi untuk merasakan hubungan yang lebih dalam dari sekadar hubungan biasa."
    elif probability >= 0.65:
        return "Anda memiliki kemungkinan moderat untuk memiliki Twin Flame. Mungkin ada ikatan khusus yang terbentuk antara Anda dan seseorang yang istimewa."
    elif probability >= 0.60:
        return "Anda memiliki kemungkinan sedang untuk memiliki Twin Flame. Ini menunjukkan bahwa Anda mungkin memiliki ikatan yang kuat dengan seseorang, tetapi masih perlu eksplorasi lebih lanjut untuk memastikan."
    elif probability >= 0.55:
        return "Anda memiliki kemungkinan moderat untuk memiliki Twin Flame. Mungkin ada hubungan yang dalam yang perlu dieksplorasi lebih lanjut."
    elif probability >= 0.50:
        return "Anda memiliki kemungkinan sedang untuk memiliki Twin Flame. Mungkin ada hubungan yang dalam dan bermakna dalam hidup Anda."
    elif probability >= 0.45:
        return "Anda memiliki kemungkinan rendah untuk memiliki Twin Flame. Namun, jangan menutup kemungkinan untuk menemukan hubungan yang istimewa di masa depan."
    elif probability >= 0.40:
        return "Anda memiliki kemungkinan rendah untuk memiliki Twin Flame. Namun, tetaplah terbuka terhadap kemungkinan-kemungkinan baru dalam kehidupan cinta Anda."
    elif probability >= 0.35:
        return "Anda memiliki kemungkinan yang cukup rendah untuk memiliki Twin Flame. Fokuslah pada pengembangan diri dan bersiaplah untuk menemukan hubungan yang bermakna di masa depan."
    elif probability >= 0.30:
        return "Anda memiliki kemungkinan yang cukup rendah untuk memiliki Twin Flame. Tetapi ingatlah bahwa hubungan yang bermakna bisa datang dari berbagai tempat."
    elif probability >= 0.25:
        return "Anda memiliki kemungkinan rendah untuk memiliki Twin Flame. Tetapi jangan ragu untuk terus mencari koneksi yang mendalam."
    elif probability >= 0.20:
        return "Anda memiliki kemungkinan yang rendah untuk memiliki Twin Flame. Fokuslah pada pertumbuhan pribadi dan hubungan yang sehat."
    elif probability >= 0.15:
        return "Anda memiliki kemungkinan sangat rendah untuk memiliki Twin Flame. Tetapi bersikaplah terbuka terhadap hubungan baru yang mungkin hadir dalam hidup Anda."
    elif probability >= 0.10:
        return "Anda memiliki kemungkinan sangat rendah untuk memiliki Twin Flame. Namun, percayalah bahwa cinta dapat ditemukan dalam berbagai bentuk."
    elif probability >= 0.05:
        return "Anda memiliki kemungkinan sangat rendah untuk memiliki Twin Flame. Tetapi jangan ragu untuk menjelajahi hubungan yang memberi Anda kebahagiaan dan kedamaian."
    else:
        return "Anda memiliki kemungkinan yang sangat rendah untuk memiliki Twin Flame. Ingatlah bahwa cinta dan hubungan yang bermakna dapat ditemukan di berbagai tempat."

def main():
    st.title("Deteksi Kemungkinan Twin Flame")

    # Load data
    df = load_data()

    # Train model
    rf_model = train_model(df)

    # Input data pengguna
    st.subheader("Masukkan data pengguna:")
    usia = st.slider("Usia", min_value=1, max_value=100, value=30)
    jenis_kelamin = st.selectbox("Jenis Kelamin", df['Jenis_Kelamin'].unique())
    minat = st.selectbox("Minat", df['Minat'].unique())
    nilai_pribadi = st.selectbox("Nilai Pribadi", df['Nilai_Pribadi'].unique())
    mbti = st.selectbox("MBTI", df['MBTI'].unique())

    user_data = {'Usia': usia, 'Jenis_Kelamin': jenis_kelamin, 'Minat': minat, 'Nilai_Pribadi': nilai_pribadi, 'MBTI': mbti}

    # Tombol untuk melakukan prediksi
    if st.button("Prediksi Kemungkinan Twin Flame"):
        pred_tf = predict_twin_flame(rf_model, user_data)
        interpretation = interpret_prediction(pred_tf)
        st.write("Perkiraan kemungkinan Twin Flame:", pred_tf)
        st.write("Interpretasi:", interpretation)

if __name__ == "__main__":
    main()
