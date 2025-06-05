#Load and clean data
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = pd.read_csv("/Users/dell/Downloads/StudentsPerformance.csv")
data.columns = [col.strip().replace(" ", "_").lower() for col in data.columns]
data['average_score'] = data[['math_score', 'reading_score', 'writing_score']].mean(axis=1)
data["high_performer"] = (data["average_score"] > 70).astype(int)
#
# # One-hot encode categorical variables
categorical_cols = ["gender", "race/ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
#
# # Train Random Forest model
X = data_encoded.drop(columns=["math_score", "reading_score", "writing_score", "average_score", "high_performer"])
y = data_encoded["high_performer"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
#
# # Streamlit UI
st.title("🎓 Student Performance Predictor")
#
st.sidebar.header("Input Student Information")
#
# # Input selectors
gender = st.sidebar.selectbox("Gender", data["gender"].unique())
race = st.sidebar.selectbox("Race/Ethnicity", data["race/ethnicity"].unique())
education = st.sidebar.selectbox("Parental Education", data["parental_level_of_education"].unique())
lunch = st.sidebar.selectbox("Lunch Type", data["lunch"].unique())
prep = st.sidebar.selectbox("Test Preparation", data["test_preparation_course"].unique())
#
# # Build input DataFrame
input_dict = {
    "gender_male": 1 if gender == "male" else 0,
    "lunch_standard": 1 if lunch == "standard" else 0,
    "test_preparation_course_none": 1 if prep == "none" else 0
}
#
# # Race/Ethnicity one-hot
for group in ["group B", "group C", "group D", "group E"]:
    input_dict[f"race/ethnicity_{group}"] = 1 if race == group else 0

# Parental education one-hot
for level in ["bachelor's degree", "high school", "master's degree", "some college", "some high school"]:
    input_dict[f"parental_level_of_education_{level}"] = 1 if education == level else 0
#


# Step 1: Store expected columns from training data
expected_cols = X_train.columns.tolist()

# Example: Suppose user input dict (you will build this from Streamlit inputs)
user_input = {
    'gender_male': 1,
    'race/ethnicity_group B': 0,
    'race/ethnicity_group C': 1,
    'race/ethnicity_group D': 0,
    'race/ethnicity_group E': 0,
    "parental_level_of_education_bachelor's degree": 0,
    "parental_level_of_education_high school": 1,
    "parental_level_of_education_master's degree": 0,
    "parental_level_of_education_some college": 0,
    "parental_level_of_education_some high school": 0,
    "lunch_standard": 1,
    "test_preparation_course_none": 0
}

# Step 2: Create DataFrame from input dict
input_df = pd.DataFrame([user_input])

# Step 3: Add missing columns with 0
for col in expected_cols:
    if col not in input_df.columns:
        input_df[col] = 0

# Step 4: Reorder columns to match training data
input_df = input_df[expected_cols]

# Step 5: Predict
prediction = model.predict(input_df)[0]

print("Prediction:", prediction)

# # # Convert to DataFrame
# input_df = pd.DataFrame([input_dict])
# #
# # # Predict
# prediction = model.predict(input_df)[0]
# st.subheader("Prediction Result")
# st.write("✅ **High Performer**" if prediction == 1 else "⚠️ **Low Performer**")
# #
# # Optional: EDA
st.subheader("📊 EDA: Average Score by Gender")
fig, ax = plt.subplots()
sns.boxplot(x="gender", y="average_score", data=data, ax=ax)
st.pyplot(fig)
#
st.subheader("📈 Test Preparation Distribution")
fig2, ax2 = plt.subplots()
sns.countplot(x="test_preparation_course", data=data, ax=ax2)
st.pyplot(fig2)