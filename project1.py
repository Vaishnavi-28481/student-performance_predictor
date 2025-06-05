#load the data
import pandas as pd
data=pd.read_csv("C:/Users/dell/Downloads/StudentsPerformance.csv")
print(data)

#clean columns name

data.columns=[col.strip().replace(" ","_").lower()for col in data.columns]
print(data.columns)

#target variable
target="math_score"

#one-hot encode

categorical_cols=['gender','race/ethnicity','parental_level_of_education','lunch','test_preparation_course']
data_encoded=pd.get_dummies(data,columns=categorical_cols,drop_first=True)


#features variable(x) & target variable(y)
X=data_encoded.drop(columns=["math_score","reading_score","writing_score"])
y=data_encoded["math_score"]
print(X)
print(y)

print(X.dtypes)
X = X.astype(int)


# Multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X_const = add_constant(X)

vif_df = pd.DataFrame()
vif_df["Variable"] = X_const.columns
vif_df["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]

print(vif_df)

#multiple regression model
import statsmodels.api as sm

X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()
print(model.summary())

#model with RMSE
from sklearn.metrics import mean_squared_error
import numpy as np

y_pred = model.predict(X_const)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print("RMSE:", rmse)


#visualize residual
import seaborn as sns
import matplotlib.pyplot as plt

residuals = y - y_pred

sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.xlabel("Residuals")
plt.show()