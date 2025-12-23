---
marp: true
theme: default
paginate: true
backgroundColor: #fff
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 28px;
  }
  h1 { color: #2c3e50; font-size: 42px; }
  h2 { color: #3498db; font-size: 32px; }
  h3 { font-size: 26px; }
  table { font-size: 22px; }
  ul, ol { font-size: 24px; }
  code { font-size: 20px; }
---

# Student Performance Prediction
## Using Apache Spark & Machine Learning

---

# Problem Statement

### The Challenge
- Teachers can't monitor every student
- Hard to identify **at-risk students** early

### Our Solution
- Predicts performance BEFORE exams
- Enables **early intervention**

---

# Dataset Overview

**Source:** Kaggle - Student Performance Dataset

| Metric | Value |
|--------|-------|
| Records | 1,000 students |
| Features | 13 |

<!-- ### Key Features
`AttendanceRate` · `StudyHoursPerWeek` · `PreviousGrade`
`ParentalSupport` · `ExtracurricularActivities` · `FinalGrade` -->

---

# Data Cleaning (1/2)

### Problems Found
- Missing values (~40-50 per column)
- Outliers in numerical features
- No target variable

### Solutions

| Problem | Solution |
|---------|----------|
| Missing (Numerical) | Median |
| Missing (Categorical) | Mode |
| Outliers | IQR capping |

---

# Data Cleaning (2/2)

### Target Variable Created

| Level | Condition |
|-------|-----------|
| **High** | FinalGrade ≥ 85 |
| **Medium** | FinalGrade 70-84 |
| **Low** | FinalGrade < 70 |

---

# Exploratory Data Analysis

### Summary Statistics

| Metric | Value |
|--------|-------|
| Avg Final Grade | 80.03 |
| Avg Attendance | 85.6% |
| Avg Study Hours | 17.6 hrs/week |

### Key Findings
- **High parental support** → better performance
- Strong correlation: **study hours** ↔ **grades**

---

# Data Preparation

### Encoding
- Gender: Male/Female → 0/1
- ParentalSupport: Low/Med/High → 0/1/2
- OnlineClasses: False/True → 0/1

### Other Steps
- **Scaling:** StandardScaler on numerical features
- **Removed:** StudentID, Name
- **Split:** 80% train / 20% test

---

# Machine Learning Models

| Model | Pros | Cons |
|-------|------|------|
| **Logistic Regression** | Fast, good baseline | Less flexible |
| **Decision Tree** | Simple, interpretable | Overfitting |
| **Random Forest** | Better accuracy | Slower |

---

# Model Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| F1-Score | Balance of precision & recall |

### Results

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Decision Tree | __% | __% |
| Random Forest | __% | __% |
| Logistic Regression | __% | __% |

---

# Confusion Matrix

*[Insert confusion matrix heatmap]*

| Class | Precision | Recall |
|-------|-----------|--------|
| High | __% | __% |
| Medium | __% | __% |
| Low | __% | __% |

---

# Feature Importance

### Top Predictive Features

1. **Previous Grade** ⭐
2. **Study Hours Per Week**
3. **Attendance Rate**
4. **Parental Support**
5. **Extracurricular Activities**

*[Insert bar chart]*

---

# Streamlit Web App

*[Insert screenshot]*

**Demo:** [Streamlit Cloud URL]

---

<!-- # Key Findings

### Results
- ML predicts performance with **__%** accuracy
- **Previous grades** = strongest predictor
- **Parental involvement** matters

### Recommendations
1. Monitor low-performing students early
2. Encourage 17+ hrs/week study
3. Involve parents in support

--- -->

# Thank You!

## Questions?

**Team:** Melissia · Nadine · Bassant · Makady

**GitHub:** [Repository Link]
**Demo:** [Streamlit Cloud Link]
