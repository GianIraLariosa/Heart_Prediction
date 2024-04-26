import streamlit as st
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the saved model
model = load_model("heart_disease_model.h5")

# Function to preprocess input data
def preprocess_input(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# Main function for Streamlit app
def main():
    st.title("Heart Disease Prediction")

    # Create input fields for user input
    age = st.number_input("Age")
    sex = st.radio("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)")
    chol = st.number_input("Serum Cholesterol (mg/dl)")
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
    restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved")
    exang = st.radio("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest")
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels Colored by Flouroscopy", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

    # Preprocess user input
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [1 if sex == "Male" else 0],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [1 if fbs == "True" else 0],
        "restecg": [restecg],
        "thalach": [thalach],
        "exang": [1 if exang == "Yes" else 0],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal]
    })

    processed_data = preprocess_input(input_data)

    # Make predictions
    prediction = model.predict(processed_data)
    result = "Heart Disease Present" if prediction > 0.5 else "No Heart Disease"

    # Display prediction result
    st.write("Prediction:", result)

if __name__ == "__main__":
    main()
