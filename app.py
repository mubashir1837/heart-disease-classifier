import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

@st.cache_resource
def load_model():
    return joblib.load("heart_disease_model.pkl")  

# Function to make predictions
def predict_heart_disease(data, model):
    prediction = model.predict(data)
    probability = model.predict_proba(data)
    return prediction, probability

# Function to visualize feature importance
def plot_feature_importance(model, features):
    # For Logistic Regression, use the coefficients as feature importance
    importance = np.abs(model.coef_[0])  # Get absolute values of coefficients
    sorted_idx = np.argsort(importance)
    fig, ax = plt.subplots()
    ax.barh(range(len(sorted_idx)), importance[sorted_idx], align="center")
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels(features[sorted_idx])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Feature Importance in Heart Disease Prediction")
    st.pyplot(fig)

# Set up Streamlit app
st.set_page_config(page_title="Heart Disease Classifier", page_icon="❤️", layout="wide", initial_sidebar_state="expanded")

# Theme toggle: Light/Dark Mode
theme = st.sidebar.radio("Select Theme", ["Light", "Dark"])

# Apply theme styles
if theme == "Dark":
    st.markdown(
        """
        <style>
        body {
            background-color: #1e1e1e;
            color: white;
        }
        .stButton>button {
            background-color: #444444;
            color: white;
        }
        .stSelectbox>div>div>input {
            color: black;
            background-color: white;
        }
        .stTextInput>div>div>input {
            color: black;
            background-color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f0f0;
            color: black;
        }
        .stButton>button {
            background-color: #dddddd;
            color: black;
        }
        .stSelectbox>div>div>input {
            color: black;
            background-color: white;
        }
        .stTextInput>div>div>input {
            color: black;
            background-color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.title("Heart Disease Classification Tool")
st.write("This tool uses a machine learning model to predict the likelihood of heart disease based on patient data.")

# Sidebar for navigation
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio(
    "Select an Option:",
    ["Predict Heart Disease", "Explore Feature Importance"]
)

# Load model
model = load_model()

if app_mode == "Predict Heart Disease":
    st.header("Heart Disease Prediction")
    
    # Input form for patient data
    st.subheader("Enter Patient Data")
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=0, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=["True", "False"])
    restecg = st.selectbox("Resting Electrocardiographic Results", options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, value=150)
    exang = st.selectbox("Exercise-Induced Angina", options=["Yes", "No"])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=["Upsloping", "Flat", "Downsloping"])
    ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4, value=0)
    thal = st.selectbox("Thalassemia", options=["Normal", "Fixed Defect", "Reversible Defect"])

    # Map categorical inputs to numeric values
    sex_map = {"Male": 1, "Female": 0}
    cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
    fbs_map = {"True": 1, "False": 0}
    restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    exang_map = {"Yes": 1, "No": 0}
    slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex_map[sex]],
        "cp": [cp_map[cp]],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs_map[fbs]],
        "restecg": [restecg_map[restecg]],
        "thalach": [thalach],
        "exang": [exang_map[exang]],
        "oldpeak": [oldpeak],
        "slope": [slope_map[slope]],
        "ca": [ca],
        "thal": [thal_map[thal]]
    })

    if st.button("Predict Heart Disease"):
        prediction, probability = predict_heart_disease(input_data, model)
        if prediction[0] == 1:
            st.error("The model predicts that the patient is likely to have heart disease.")
        else:
            st.success("The model predicts that the patient is unlikely to have heart disease.")
        
        st.write("Prediction Confidence:")
        st.write(f"No Heart Disease: {probability[0][0] * 100:.2f}%")
        st.write(f"Heart Disease: {probability[0][1] * 100:.2f}%")

elif app_mode == "Explore Feature Importance":
    st.header("Feature Importance")
    st.write("Visualize which features contribute the most to the model's predictions.")
    
    # Plot feature importance
    feature_names = np.array(["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"])
    plot_feature_importance(model, feature_names)

# Footer with GitHub icon and link
st.markdown("---")
st.write("**Disclaimer:** This tool is for educational purposes only and should not be used for medical diagnosis or treatment.")
st.markdown(
    """
    <div style="text-align: center; margin-top: 20%;">
        <a href="https://github.com/mubashir1837/heart-disease-classifier" target="_blank">
            <img src="https://img.icons8.com/ios-filled/50/000000/github.png" alt="GitHub" />
            <h4>Source Code<h4/>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
