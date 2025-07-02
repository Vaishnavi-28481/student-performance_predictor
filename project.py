# ============================================
# STEP 1:DATA CLEANING & PREPROCESSING
# ===========================================


#Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data=pd.read_csv("C:/Users/dell/Downloads/StudentsPerformance.csv")

#Check Normality and Feature Strength
print(data[['math score', 'reading score', 'writing score']].describe())

#clean columns name

data.columns=[col.strip().replace(" ","_").lower()for col in data.columns]

#one-hot encode

categorical_cols=['gender','race/ethnicity','parental_level_of_education','lunch','test_preparation_course']
data_encoded=pd.get_dummies(data,columns=categorical_cols,drop_first=True)

# ===============================================
# STEP 2: REGRESSION MODELS
# -OLS
# -MultiOutput Linear
# -Random Forest
# -XGBoost
# ==============================================

# OLS for Math Score

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

# Define features and target for math
X_math = data_encoded.drop(columns=['math_score', 'reading_score', 'writing_score'])
y_math = data_encoded['math_score']
X_math = X_math.astype(int)

# VIF to check multicollinearity
X_const = add_constant(X_math)
vif_df = pd.DataFrame()
vif_df["Variable"] = X_const.columns
vif_df["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
print("VIF Table:\n", vif_df)

# Fit OLS regression model
ols_model = sm.OLS(y_math, X_const).fit()
print(ols_model.summary())

# Predict and evaluate
y_pred_math = ols_model.predict(X_const)
rmse_math = np.sqrt(mean_squared_error(y_math, y_pred_math))
print("RMSE (Math - OLS):", rmse_math)

# Plot residuals
residuals_math = y_math - y_pred_math
sns.histplot(residuals_math, kde=True)
plt.title("Residuals Distribution - Math (OLS)")
plt.xlabel("Residuals")
plt.savefig("graphs/math_score_distribution_plot.png")
plt.show()

# ===================================

# Multi-output ML Regression


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score

# Define features and multi-targets
X = data_encoded.drop(columns=['math_score', 'reading_score', 'writing_score'])
y = data_encoded[['math_score', 'reading_score', 'writing_score']]
X = X.astype(int)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the multi-output model
ml_model = MultiOutputRegressor(LinearRegression())
ml_model.fit(X_train, y_train)

# Predict
y_pred_all = ml_model.predict(X_test)

# Evaluate RMSE and R²
for i, subject in enumerate(['math_score', 'reading_score', 'writing_score']):
    rmse = np.sqrt(mean_squared_error(y_test[subject], y_pred_all[:, i]))
    r2 = r2_score(y_test[subject], y_pred_all[:, i])
    print(f"{subject} - RMSE: {rmse:.2f}, R²: {r2:.2f}")

# Plot residuals for all 3
residuals_reading = y_test['reading_score'] - y_pred_all[:, 1]
residuals_writing = y_test['writing_score'] - y_pred_all[:, 2]

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
sns.histplot(residuals_math, kde=True, color='blue')
plt.title("Residuals - Math (ML)")

plt.subplot(1, 3, 2)
sns.histplot(residuals_reading, kde=True, color='green')
plt.title("Residuals - Reading (ML)")

plt.subplot(1, 3, 3)
sns.histplot(residuals_writing, kde=True, color='red')
plt.title("Residuals - Writing (ML)")

plt.tight_layout()
plt.savefig("graphs/all_score_distribution_plot.png")
plt.show()

# ================================================
# --- Random Forest Model ---
from sklearn.ensemble import RandomForestRegressor

rf_model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("\n🎯 Random Forest Results:")
for i, subject in enumerate(['math_score', 'reading_score', 'writing_score']):
    rmse = np.sqrt(mean_squared_error(y_test[subject], rf_preds[:, i]))
    r2 = r2_score(y_test[subject], rf_preds[:, i])
    print(f"{subject} - RMSE: {rmse:.2f}, R²: {r2:.2f}")


# ===============================================
# --- XGBoost Model ---
from xgboost import XGBRegressor
xgb_model = MultiOutputRegressor(XGBRegressor(objective='reg:squarederror', random_state=42))
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

print("\n🎯 XGBoost Results:")
for i, subject in enumerate(['math_score', 'reading_score', 'writing_score']):
    rmse = np.sqrt(mean_squared_error(y_test[subject], xgb_preds[:, i]))
    r2 = r2_score(y_test[subject], xgb_preds[:, i])
    print(f"{subject} - RMSE: {rmse:.2f}, R²: {r2:.2f}")

# =======================================
#STEP 3: FEATURE ENGINEERING
# ========================================

data['average_score'] = data[['math_score', 'reading_score', 'writing_score']].mean(axis=1)
print(data['average_score'])

data["high_performer"] = (data["average_score"] > 70).astype(int)

categorical_cols = ["gender", "race/ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
df_encoded = pd.get_dummies(data,columns=categorical_cols, drop_first=True)
print(df_encoded)

from scipy.stats import ttest_ind, f_oneway

# Distribution by gender
sns.boxplot(x="gender", y="average_score", data=data)
plt.title("Average Score by Gender")
plt.savefig("graphs/distribution_gender_plot.png")
plt.show()

sns.countplot(x="test_preparation_course", data=data)
plt.title("Test Preparation Course Distribution")
plt.savefig("graphs/distribution_testprep_plot.png")
plt.show()

# ANOVA for parental education
groups = [data[data['parental_level_of_education'] == level]['average_score'] for level in data['parental_level_of_education'].unique()]
f_stat, p_val = f_oneway(*groups)
print("ANOVA p-value:", p_val)


# =========================================
# STEP 4: CLASSIFICATION
# =========================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df_encoded["high_performer"] = (df_encoded["average_score"] > 70).astype(int)

X = df_encoded.drop(columns=["math_score", "reading_score", "writing_score", "average_score", "high_performer"])
y = df_encoded["high_performer"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

from scipy.stats import ttest_ind

prep = data[data["test_preparation_course"] == "completed"]["average_score"]
no_prep = data[data["test_preparation_course"] == "none"]["average_score"]

t_stat, p_val = ttest_ind(prep, no_prep)
print("T-test p-value:", p_val)




