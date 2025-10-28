import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ========== Load Model & Scaler ==========
model = joblib.load("Frontend/exam_score_model.pkl")
scaler = joblib.load("Frontend/scaler.pkl")

# Define the expected feature columns (must match training data exactly)
expected_columns = [
    'age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
    'attendance_percentage', 'sleep_hours', 'exercise_frequency',
    'gender_Male', 'gender_Other', 'part_time_job_Yes', 
    'diet_quality_Good', 'diet_quality_Poor',
    'parental_education_level_High School', 'parental_education_level_Master',
    'internet_quality_Good', 'internet_quality_Poor', 
    'extracurricular_participation_Yes'
]

st.set_page_config(
    page_title="🎓 Student Exam Score Predictor",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ========== Custom Styling ==========
st.markdown("""
    <style>
        /* Background and text */
        body {
            background-color: #0E1117;
            color: white;
        }
        .main {
            background-color: #0E1117;
        }
        /* Headings */
        h1, h2, h3 {
            color: #4FC3F7;
            text-align: center;
        }
        /* Input widgets */
        div[data-baseweb="input"] input {
            background-color: #1C1E24;
            color: white;
            border-radius: 10px;
        }
        /* Buttons */
        div.stButton > button {
            background: linear-gradient(135deg, #4FC3F7, #0288D1);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6em 1em;
            font-size: 1.1em;
            font-weight: bold;
        }
        div.stButton > button:hover {
            background: linear-gradient(135deg, #0288D1, #4FC3F7);
        }
        /* Footer */
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ========== App Header ==========
st.markdown("## 🎓 Student Exam Score Prediction")
st.write("Enter student lifestyle and study habits to estimate their **expected exam score**.")

st.markdown("---")

# ========== User Input Form ==========
with st.form("student_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("📅 Age", 16, 30, 20)
        study_hours_per_day = st.number_input("📚 Study Hours per Day", 0.0, 10.0, 5.0)
        sleep_hours = st.number_input("💤 Sleep Hours", 0.0, 12.0, 7.0)
        attendance_percentage = st.number_input("🏫 Attendance Percentage", 0.0, 100.0, 85.0)
        exercise_frequency = st.slider("🏋️‍♂️ Exercise Frequency (days/week)", 0, 7, 3)

    with col2:
        social_media_hours = st.number_input("📱 Social Media Hours", 0.0, 10.0, 2.0)
        netflix_hours = st.number_input("🎬 Netflix Hours", 0.0, 10.0, 1.0)
        gender = st.selectbox("👤 Gender", ["Male", "Female", "Other"])
        part_time_job = st.selectbox("💼 Part-time Job", ["Yes", "No"])

    diet_quality = st.selectbox("🥗 Diet Quality", ["Good", "Fair", "Poor"])
    parental_education = st.selectbox("🎓 Parental Education", ["High School", "Master", "Other"])
    internet_quality = st.selectbox("🌐 Internet Quality", ["Good", "Average", "Poor"])
    extracurricular = st.selectbox("🎭 Extracurricular Participation", ["Yes", "No"])

    submitted = st.form_submit_button("🔍 Predict Exam Score")

# ========== Prediction Logic ==========
if submitted:
    # Construct dataframe (must match training columns exactly)
    input_dict = {
        "age": age,
        "study_hours_per_day": study_hours_per_day,
        "social_media_hours": social_media_hours,
        "netflix_hours": netflix_hours,
        "attendance_percentage": attendance_percentage,
        "sleep_hours": sleep_hours,
        "exercise_frequency": exercise_frequency,
        "gender_Male": 1 if gender == "Male" else 0,
        "gender_Other": 1 if gender == "Other" else 0,
        "part_time_job_Yes": 1 if part_time_job == "Yes" else 0,
        "diet_quality_Good": 1 if diet_quality == "Good" else 0,
        "diet_quality_Poor": 1 if diet_quality == "Poor" else 0,
        "parental_education_level_High School": 1 if parental_education == "High School" else 0,
        "parental_education_level_Master": 1 if parental_education == "Master" else 0,
        "internet_quality_Good": 1 if internet_quality == "Good" else 0,
        "internet_quality_Poor": 1 if internet_quality == "Poor" else 0,
        "extracurricular_participation_Yes": 1 if extracurricular == "Yes" else 0
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    st.success(f"🎯 **Predicted Exam Score:** {round(prediction, 2)} / 100")

    st.markdown("""
        <hr>
        <h4 style='text-align:center; color:#4FC3F7;'>Insight:</h4>
        <p style='text-align:center;'>Consistent study habits and healthy lifestyle are key to strong exam performance!</p>
    """, unsafe_allow_html=True)

