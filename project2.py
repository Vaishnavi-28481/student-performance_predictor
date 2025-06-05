import pandas as pd
data=pd.read_csv('/Users/dell/Downloads/StudentsPerformance.csv')
print(data)
print(data.head())

print(data.nunique())

print(data.info())

print(data.describe())

print(data.isnull().sum())


data.columns = [col.strip().replace(" ", "_").lower() for col in data.columns]
print(data.columns)

data['average_score'] = data[['math_score', 'reading_score', 'writing_score']].mean(axis=1)
print(data['average_score'])

data["high_performer"] = (data["average_score"] > 70).astype(int)

categorical_cols = ["gender", "race/ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
df_encoded = pd.get_dummies(data,columns=categorical_cols, drop_first=True)
print(df_encoded)


import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway

# Distribution by gender
sns.boxplot(x="gender", y="average_score", data=data)
plt.title("Average Score by Gender")
plt.show()

sns.countplot(x="test_preparation_course", data=data)
plt.title("Test Preparation Course Distribution")
plt.show()

# ANOVA for parental education
groups = [data[data['parental_level_of_education'] == level]['average_score'] for level in data['parental_level_of_education'].unique()]
f_stat, p_val = f_oneway(*groups)
print("ANOVA p-value:", p_val)
#
#
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
#
df_encoded["high_performer"] = (df_encoded["average_score"] > 70).astype(int)
#
X = df_encoded.drop(columns=["math_score", "reading_score", "writing_score", "average_score", "high_performer"])
y = df_encoded["high_performer"]
#
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
model = RandomForestClassifier()
model.fit(X_train, y_train)
#
y_pred = model.predict(X_test)
#
print(classification_report(y_test, y_pred))

from scipy.stats import ttest_ind

prep = data[data["test_preparation_course"] == "completed"]["average_score"]
no_prep = data[data["test_preparation_course"] == "none"]["average_score"]

t_stat, p_val = ttest_ind(prep, no_prep)
print("T-test p-value:", p_val)
