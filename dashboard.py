import streamlit as st
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.metrics import mean_squared_error
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Title
st.title("C:/Users/dell/Downloads/StudentsPerformance.csv")

@st.cache_data
def load_data():
    data = pd.read_csv("C:/Users/dell/Downloads/StudentsPerformance.csv")
    data.columns = [col.strip().replace(" ", "_").lower() for col in data.columns]
    return data

data = load_data()

st.subheader("Raw Dataset")
st.dataframe(data)

# Preprocessing
target = "math_score"
categorical_cols = ['gender','race/ethnicity','parental_level_of_education','lunch','test_preparation_course']
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

X = data_encoded.drop(columns=["math_score","reading_score","writing_score"])
y = data_encoded["math_score"]
X = X.astype(int)

# Multicollinearity - VIF
X_const = add_constant(X)
vif_df = pd.DataFrame()
vif_df["Variable"] = X_const.columns
vif_df["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]

st.subheader("Multicollinearity Check - Variance Inflation Factor (VIF)")
st.dataframe(vif_df)

# Fit model
model = sm.OLS(y, X_const).fit()

st.subheader("Regression Model Summary")
st.text(model.summary())

# Predictions and RMSE
y_pred = model.predict(X_const)
rmse = np.sqrt(mean_squared_error(y, y_pred))
st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.3f}")

# Residual plot
residuals = y - y_pred

st.subheader("Residual Distribution")
fig, ax = plt.subplots()
sns.histplot(residuals, kde=True, ax=ax)
ax.set_xlabel("Residuals")
ax.set_title("Residual Distribution")
st.pyplot(fig)
