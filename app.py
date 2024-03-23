import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load data
@st.cache
def load_data():
    data = {
        'Usia': [25, 30, 28, 35, 23, 40, 22, 27],
        'Jenis_Kelamin': ['Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan', 'Laki-laki', 'Perempuan'],
        'Minat': ['Olahraga', 'Seni', 'Seni', 'Olahraga', 'Seni', 'Olahraga', 'Olahraga', 'Seni'],
        'Nilai_Pribadi': ['Baik', 'Baik', 'Baik', 'Baik', 'Buruk', 'Buruk', 'Buruk', 'Baik'],
        'Twin_Flame': [1, 0, 1, 0, 1, 0, 1, 0],
        'Soulmate': [0, 1, 0, 1, 0, 1, 0, 1],
        'MBTI': ['INTJ', 'ENFP', 'ISTJ', 'ENFJ', 'ISTP', 'ENTJ', 'INFJ', 'ENTP']
    }
    df = pd.DataFrame(data)
    return df

# Train model
@st.cache
def train_model(df):
    X = df[['Usia', 'Jenis_Kelamin', 'Minat', 'Nilai_Pribadi']]  # Fitur-fitur
    y_tf = df['Twin_Flame']  # Target Twin Flame
    y_mbti = df['MBTI']  # Target MBTI

    # Encode categorical variables
    X_encoded = pd.get_dummies(X)

    # Split data into train and test sets
    X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(X_encoded, y_tf, test_size=0.2, random_state=42)

    # Initialize models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

    # Train Twin Flame model
    rf_model.fit(X_train_tf, y_train_tf)
    gb_model.fit(X_train_tf, y_train_tf)

    return rf_model, gb_model

# Predict Twin Flame probability
def predict_twin_flame(rf_model, gb_model, user_data):
    # Encode user input
    user_encoded = pd.get_dummies(user_data)

    # Check if input features match trained model features
    if set(user_encoded.columns) != set(X_encoded.columns):
        missing_features = set(X_encoded.columns) - set(user_encoded.columns)
        st.error(f"Incomplete input data, missing features: {missing_features}")
        return None

    # Predict Twin Flame probability using models
    pred_rf = rf_model.predict_proba(user_encoded)[:, 1]
    pred_gb = gb_model.predict_proba(user_encoded)[:, 1]

    # Average the probabilities from both models
    pred_tf = (pred_rf + pred_gb) / 2
    return pred_tf

# Main function
def main():
    st.title("Twin Flame Detection with Ensemble Learning")

    # Load data
    df = load_data()

    # Train models
    rf_model, gb_model = train_model(df)

    # User input
    st.subheader("User Input:")
    usia = st.text_input("Usia")
    jenis_kelamin = st.selectbox("Jenis Kelamin", ['Laki-laki', 'Perempuan'])
    minat = st.selectbox("Minat", ['Olahraga', 'Seni'])
    nilai_pribadi = st.selectbox("Nilai Pribadi", ['Baik', 'Buruk'])

    # Button to predict Twin Flame probability
    if st.button("Predict Twin Flame Probability"):
        user_data = {'Usia': usia, 'Jenis_Kelamin': jenis_kelamin, 'Minat': minat, 'Nilai_Pribadi': nilai_pribadi}
        pred_tf = predict_twin_flame(rf_model, gb_model, user_data)

        # Calculate MBTI based on user input
        mbti = 'INFJ'  # Placeholder for MBTI calculation based on user input
        
        if pred_tf is not None:
            st.write("Predicted Twin Flame Probability:", pred_tf)
            st.write("MBTI Type based on user input:", mbti)

if __name__ == "__main__":
    main()
