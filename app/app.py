import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="AI Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Load models
breast_model = joblib.load("../models/breast_cancer_model.pkl")
diabetes_model = joblib.load("../models/diabetes_model.pkl")
heart_model = joblib.load("../models/heart_disease_model.pkl")

# Load scalers
breast_scaler = joblib.load("../models/breast_scaler.pkl")
diabetes_scaler = joblib.load("../models/diabetes_scaler.pkl")
heart_scaler = joblib.load("../models/heart_scaler.pkl")

st.title("🏥 AI Disease Prediction System")

st.markdown("Predict diseases using Machine Learning models trained on medical datasets.")

# Sidebar navigation
page = st.sidebar.radio(
    "Select Disease",
    ["Home", "Diabetes", "Heart Disease", "Breast Cancer"]
)

# ================= HOME =================
if page == "Home":

    st.header("Dashboard Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Breast Cancer Model", "98.25%")
    col2.metric("Diabetes Model", "75.32%")
    col3.metric("Heart Disease Model", "90.85%")

    st.success("Select a disease from sidebar to start prediction.")

# ================= DIABETES =================
elif page == "Diabetes":

    st.header("Diabetes Prediction")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0)
        glucose = st.number_input("Glucose")
        blood_pressure = st.number_input("Blood Pressure")
        skin_thickness = st.number_input("Skin Thickness")

    with col2:
        insulin = st.number_input("Insulin")
        bmi = st.number_input("BMI")
        dpf = st.number_input("Diabetes Pedigree Function")
        age = st.number_input("Age")

    if st.button("Predict"):

        data = pd.DataFrame([[
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf, age
        ]])

        data = diabetes_scaler.transform(data)
        result = diabetes_model.predict(data)

        if result[0] == 1:
            st.error("Diabetes Detected")
        else:
            st.success("No Diabetes Detected")

# ================= HEART =================
elif page == "Heart Disease":

    st.header("Heart Disease Prediction")

    def binary_input(label):
        return 1 if st.selectbox(label, ["No", "Yes"]) == "Yes" else 0

    col1, col2, col3 = st.columns(3)

    with col1:
        highbp = binary_input("High Blood Pressure")
        highchol = binary_input("High Cholesterol")
        cholcheck = binary_input("Cholesterol Check")
        smoker = binary_input("Smoker")
        stroke = binary_input("Stroke")
        physactivity = binary_input("Physical Activity")
        fruits = binary_input("Fruits")

    with col2:
        veggies = binary_input("Veggies")
        alcohol = binary_input("Heavy Alcohol Consumption")
        healthcare = binary_input("Healthcare Access")
        nodoc = binary_input("No Doctor Due to Cost")
        diffwalk = binary_input("Difficulty Walking")
        sex = binary_input("Sex (Yes=Male)")
        diabetes = st.number_input("Diabetes (0/1)", 0, 1)

    with col3:
        bmi = st.number_input("BMI")
        genhlth = st.number_input("General Health (1-5)")
        menthlth = st.number_input("Mental Health Days")
        physhlth = st.number_input("Physical Health Days")
        age = st.number_input("Age Category")
        education = st.number_input("Education")
        income = st.number_input("Income")

    if st.button("Predict Heart Disease"):

        data = pd.DataFrame([[
            highbp, highchol, cholcheck, bmi, smoker,
            stroke, diabetes, physactivity, fruits, veggies,
            alcohol, healthcare, nodoc, genhlth, menthlth,
            physhlth, diffwalk, sex, age, education, income
        ]])

        data = heart_scaler.transform(data)
        result = heart_model.predict(data)

        if result[0] == 1:
            st.error("Heart Disease Detected")
        else:
            st.success("No Heart Disease Detected")

# ================= BREAST CANCER =================
elif page == "Breast Cancer":

    st.header("Breast Cancer Prediction")

    st.info("Enter diagnostic values below")

    col1, col2, col3 = st.columns(3)

    with col1:
        radius_mean = st.number_input("Radius Mean")
        texture_mean = st.number_input("Texture Mean")
        perimeter_mean = st.number_input("Perimeter Mean")
        area_mean = st.number_input("Area Mean")
        smoothness_mean = st.number_input("Smoothness Mean")
        compactness_mean = st.number_input("Compactness Mean")
        concavity_mean = st.number_input("Concavity Mean")

    with col2:
        concave_points_mean = st.number_input("Concave Points Mean")
        symmetry_mean = st.number_input("Symmetry Mean")
        fractal_dimension_mean = st.number_input("Fractal Dimension Mean")
        radius_se = st.number_input("Radius SE")
        texture_se = st.number_input("Texture SE")
        perimeter_se = st.number_input("Perimeter SE")
        area_se = st.number_input("Area SE")

    with col3:
        smoothness_se = st.number_input("Smoothness SE")
        compactness_se = st.number_input("Compactness SE")
        concavity_se = st.number_input("Concavity SE")
        concave_points_se = st.number_input("Concave Points SE")
        symmetry_se = st.number_input("Symmetry SE")
        fractal_dimension_se = st.number_input("Fractal Dimension SE")

    if st.button("Predict Breast Cancer"):

        data = pd.DataFrame([[
            radius_mean, texture_mean, perimeter_mean, area_mean,
            smoothness_mean, compactness_mean, concavity_mean,
            concave_points_mean, symmetry_mean, fractal_dimension_mean,
            radius_se, texture_se, perimeter_se, area_se,
            smoothness_se, compactness_se, concavity_se,
            concave_points_se, symmetry_se, fractal_dimension_se
        ]])

        result = breast_model.predict(data)

        if result[0] == 1:
            st.error("Cancer Detected")
        else:
            st.success("No Cancer Detected")