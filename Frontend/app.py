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

# ========== Modern Custom Styling ==========
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        * {
            font-family: 'Poppins', sans-serif;
        }
        
        /* Main container */
        .main {
            background-color: #0E1117;
        }
        
        .block-container {
            padding-top: 3rem;
            padding-bottom: 3rem;
            max-width: 900px;
        }
        
        /* Custom header */
        .app-header {
            text-align: center;
            margin-bottom: 3rem;
            animation: fadeIn 0.8s ease-in;
        }
        
        .app-title {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(120deg, #00F5FF, #0080FF, #FF00FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
            letter-spacing: -1px;
        }
        
        .app-subtitle {
            color: #B0B3B8;
            font-size: 1.1rem;
            font-weight: 400;
            letter-spacing: 0.5px;
        }
        
        /* Form styling */
        .stForm {
            background: linear-gradient(145deg, #1A1D29 0%, #252836 100%);
            border-radius: 24px;
            padding: 2.5rem;
            border: 1px solid #2D3142;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
        }
        
        /* Input labels */
        label {
            color: #E4E6EB !important;
            font-weight: 500 !important;
            font-size: 0.95rem !important;
            letter-spacing: 0.3px !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Number inputs */
        .stNumberInput input {
            background-color: #16191F !important;
            border: 2px solid #2D3142 !important;
            border-radius: 12px !important;
            color: #FFFFFF !important;
            font-size: 1rem !important;
            padding: 0.75rem !important;
            transition: all 0.3s ease !important;
        }
        
        .stNumberInput input:focus {
            border-color: #00D9FF !important;
            box-shadow: 0 0 0 3px rgba(0, 217, 255, 0.15) !important;
            background-color: #1C2028 !important;
        }
        
        /* Select boxes */
        .stSelectbox > div > div {
            background-color: #16191F !important;
            border: 2px solid #2D3142 !important;
            border-radius: 12px !important;
            color: #FFFFFF !important;
            transition: all 0.3s ease !important;
        }
        
        .stSelectbox > div > div:hover {
            border-color: #00D9FF !important;
        }
        
        /* Slider */
        .stSlider > div > div > div {
            background: linear-gradient(90deg, #00D9FF 0%, #0080FF 100%) !important;
            height: 6px !important;
        }
        
        .stSlider > div > div > div > div {
            background-color: #FFFFFF !important;
            width: 20px !important;
            height: 20px !important;
            box-shadow: 0 2px 8px rgba(0, 217, 255, 0.6) !important;
            border: 3px solid #00D9FF !important;
        }
        
        .stSlider > div > div > div > div:hover {
            transform: scale(1.2);
        }
        
        /* Submit button */
        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #00D9FF 0%, #0080FF 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 16px !important;
            padding: 1.2rem 2rem !important;
            font-size: 1.15rem !important;
            font-weight: 600 !important;
            letter-spacing: 1px !important;
            text-transform: uppercase;
            margin-top: 2rem !important;
            box-shadow: 0 10px 30px rgba(0, 217, 255, 0.3) !important;
            transition: all 0.3s ease !important;
            cursor: pointer;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 40px rgba(0, 217, 255, 0.5) !important;
            background: linear-gradient(135deg, #00F5FF 0%, #0099FF 100%) !important;
        }
        
        .stButton > button:active {
            transform: translateY(-1px);
        }
        
        /* Success alert */
        .element-container:has(.stSuccess) {
            animation: slideUp 0.5s ease-out;
        }
        
        .stSuccess {
            background: linear-gradient(135deg, #00C9A7 0%, #00B4D8 100%) !important;
            color: white !important;
            border-radius: 16px !important;
            padding: 2rem !important;
            border: none !important;
            box-shadow: 0 10px 30px rgba(0, 201, 167, 0.3) !important;
        }
        
        .stSuccess p {
            color: white !important;
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            text-align: center;
            margin: 0 !important;
        }
        
        /* Result card */
        .result-card {
            background: linear-gradient(145deg, #1E2130 0%, #2A2D3E 100%);
            border: 2px solid #00D9FF;
            border-radius: 20px;
            padding: 2rem;
            margin-top: 2rem;
            text-align: center;
            box-shadow: 0 15px 40px rgba(0, 217, 255, 0.2);
            animation: slideUp 0.6s ease-out;
        }
        
        .result-score {
            font-size: 3.5rem;
            font-weight: 700;
            background: linear-gradient(120deg, #00F5FF, #00D9FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 1rem 0;
        }
        
        .result-label {
            color: #B0B3B8;
            font-size: 1rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .insight-box {
            background: linear-gradient(135deg, #FF006E 0%, #8338EC 100%);
            border-radius: 16px;
            padding: 1.5rem;
            margin-top: 1.5rem;
            text-align: center;
            box-shadow: 0 10px 30px rgba(255, 0, 110, 0.2);
        }
        
        .insight-title {
            color: white;
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .insight-text {
            color: rgba(255, 255, 255, 0.95);
            font-size: 0.95rem;
            line-height: 1.6;
        }
        
        /* Column spacing */
        [data-testid="column"] {
            padding: 0 0.5rem;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .viewerBadge_container__1QSob {display: none;}
    </style>
""", unsafe_allow_html=True)

# ========== App Header ==========
st.markdown("""
    <div class="app-header">
        <div class="app-title">🎓 Student Habit Performance</div>
        <div class="app-subtitle">Predict student performance using AI-powered analytics</div>
    </div>
""", unsafe_allow_html=True)

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

    st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Predicted Exam Score</div>
            <div class="result-score">{round(prediction, 2)}<span style="font-size: 2rem; color: #B0B3B8;">/100</span></div>
            <div class="insight-box">
                <div class="insight-title">💡 Key Insight</div>
                <div class="insight-text">Consistent study habits and a healthy lifestyle are crucial factors for achieving strong academic performance!</div>
            </div>
        </div>
    """, unsafe_allow_html=True)