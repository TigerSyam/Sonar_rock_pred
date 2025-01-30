import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("sonar_model.pkl")

# Streamlit app title
st.title("Sonar Rock vs Mine Prediction")
st.write("""
This app predicts whether a given sonar signal corresponds to a rock or a mine.
Enter the 60 feature values below to make a prediction.
""")

# Create input fields for all 60 features
input_features = []
for i in range(60):
    value = st.number_input(f"Feature {i+1}", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    input_features.append(value)

# Convert input into numpy array
input_data = np.array(input_features).reshape(1, -1)

# Prediction button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    if prediction == 1:
        st.error(f"Prediction: Mine Detected! (Confidence: {probability[1]*100:.2f}%)")
    else:
        st.success(f"Prediction: Rock Detected. (Confidence: {probability[0]*100:.2f}%)")
