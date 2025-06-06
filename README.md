# Student Performance Predictor 🎓

This project predicts whether a student is a high performer based on socioeconomic and academic features using machine learning and statistical analysis. It includes EDA, t-tests, ANOVA, model building, evaluation, and a Streamlit dashboard.

## 📊 Dataset

The dataset used is **StudentsPerformance.csv**, which contains:

- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch Type
- Test Preparation Course
- Scores in Math, Reading, and Writing

## 🔍 Project Highlights

- ✅ Data Cleaning and Preprocessing  
- 📈 Exploratory Data Analysis (EDA)  
- 📊 Statistical Tests (T-test, ANOVA)  
- 🤖 Random Forest Classification  
- 📉 Model Evaluation  
- 🌐 Interactive Streamlit Dashboard  

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Student-Performance-Predictor.git
   cd Student-Performance-Predictor


## Install Requirements
```
 pip install -r requirements.txt
```
## Run the App
```
streamlit run app.py
```


## 📊 Model Details
- Model Used: Random Forest Classifier

- Target Variable: High Performer (1 = average_score > 70)

- Evaluation: Classification Report with Precision, Recall, F1-score

- Data Source: [StudentsPerformance.csv]

## 📁 File Structure
```
Student-Performance-Predictor/
├── app.py                   # Streamlit dashboard
├── README.md                # Project documentation
├── requirements.txt         # Required Python libraries
├── model.pkl                # Trained ML model (if saved)
├── data/
│   └── StudentsPerformance.csv
└── images/
    └── dashboard.png        # Screenshot (optional)
```

## 👩‍💻 Author
```
Vaishnavi Metkar
github.com/Vaishnavi-28481|vaishanavi.metkar2802@gmail.com
```