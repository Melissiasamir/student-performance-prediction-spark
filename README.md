# Student Performance Prediction

## Project Idea
This project aims to predict student performance using **Apache Spark** and machine learning techniques.  
We use student demographic and academic data to classify or predict performance levels.

---

## Dataset
- **File:** `student_performance.csv`  
- **Source:** [Kaggle – student_performance Dataset](https://www.kaggle.com/)  

---

## Project Structure

- project student-performance-prediction/
  - data/
    - raw/               # Original / raw datasets
      - student_performance_updated_1000.csv
    - cleaned/           # Cleaned datasets (ready for modeling)
      - student_performance_cleaned.csv
  - notebooks/
    - student_performance.ipynb   # EDA, Data Cleaning, Preprocessing & Modeling
  - .gitignore
  - README.md
  - requirements.txt

---

## Team Members & Tasks

| Member Name       | Role & Tasks | Files / Folders |
|------------------|-------------|----------------|
| **Melissia**        | **Data Engineering & Cleaning** <br> - Describe data source and collection <br> - Read and inspect dataset <br> - Check dataset shape and schema <br> - Handle missing values <br> - Remove duplicates <br> - Detect outliers <br> - Produce final cleaned dataset | `notebooks/student_performance.ipynb` <br> `data/cleaned/student_performance_cleaned` |
| **Nadine**          | **Statistical Analysis & Summaries** <br> - Compute numerical summaries <br> - Compute categorical counts <br> - Compute correlation matrix <br> - Check multicollinearity <br> - Prepare numeric and categorical summary tables | `notebooks/student_performance.ipynb` |
| **Bassant**         | **Preprocessing & ML Modeling** <br> - Encode categorical features <br> - Scale numerical features <br> - Feature engineering using Spark <br> - Split dataset into train and test sets <br> - Train ML models (Decision Tree, Random Forest, Logistic Regression…) <br> - Hyperparameter tuning <br> - Evaluate models (Accuracy, F1-score, ROC-AUC) <br> - Select best performing model | `notebooks/student_performance.ipynb` |
| **Makady**          | **Deployment & Documentation** <br> - Integrate trained model for predictions <br> - Write final project report <br> - Create presentation slides <br> - Explain ML pipeline and results | `notebooks/student_performance.ipynb` <br> `README.md` <br> Presentation slides (external) |

---

## Workflow Overview
1. **Data Cleaning & EDA** → handled in notebook by Melissia & Nadine.  
2. **Preprocessing & ML Modeling** → handled in notebook by Bassant.  
3. **Deployment & Documentation** → handled by Makady.  


