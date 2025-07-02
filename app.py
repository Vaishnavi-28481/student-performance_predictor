#Load and clean data
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Title & Intro
st.title("🎓 Student Performance Predictor")
st.markdown("""
Predict subject scores and overall performance using socio-economic inputs.
""")

# Load sample data for dropdowns
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/dell/Downloads/StudentsPerformance.csv")
    df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]
    return df

data = load_data()

# Centered form
with st.form(key="student_form"):
    st.subheader("📝 Enter Student Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", data["gender"].unique())
        education = st.selectbox("Parental Education", data["parental_level_of_education"].unique())
        prep = st.selectbox("Test Preparation", data["test_preparation_course"].unique())

    with col2:
        race = st.selectbox("Race/Ethnicity", data["race/ethnicity"].unique())
        lunch = st.selectbox("Lunch Type", data["lunch"].unique())

    submitted = st.form_submit_button("🔮 Predict Performance")

# Prepare input only after button click
if submitted:
    # Create input DataFrame
    input_dict = {
        "gender": [gender],
        "race/ethnicity": [race],
        "parental_level_of_education": [education],
        "lunch": [lunch],
        "test_preparation_course": [prep]
    }
    input_df = pd.DataFrame(input_dict)

    # One-hot encode
    input_encoded = pd.get_dummies(input_df)
    full_data = pd.get_dummies(data[["gender", "race/ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]], drop_first=True)
    input_encoded = input_encoded.reindex(columns=full_data.columns, fill_value=0)

    # Load models
    try:
        reg_model = joblib.load("regression_model.pkl")
        clf_model = joblib.load("classifier_model.pkl")
    except FileNotFoundError:
        st.error("Model files not found. Train and save them first.")
        st.stop()

    # Predict
    scores = reg_model.predict(input_encoded)[0]
    math, read, write = scores
    avg = np.mean(scores)
    label = clf_model.predict(input_encoded)[0]

    # Display predictions
    st.subheader("📊 Predicted Scores")
    st.write(f"**Math Score:** {math:.1f}")
    st.write(f"**Reading Score:** {read:.1f}")
    st.write(f"**Writing Score:** {write:.1f}")
    st.write(f"**Average Score:** {avg:.1f}")

    st.subheader("🎯 Performance Category")
    if label == 1:
        st.success("High Performer ✅")
    else:
        st.warning("Needs Improvement ⚠️")