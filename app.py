import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned/student_performance_cleaned.csv")
    return df

# ============================================================
# PLACEHOLDER MODEL - Bassant: Replace this with real model
# ============================================================
# To use a trained model:
# 1. Save your model: joblib.dump(model, 'model.pkl')
# 2. Uncomment and modify the code below:
#
# import joblib
# model = joblib.load('model.pkl')
#
# def predict_performance(features):
#     prediction = model.predict([features])[0]
#     probabilities = model.predict_proba([features])[0]
#     return prediction, probabilities
# ============================================================

def predict_performance_placeholder(attendance, study_hours, prev_grade,
                                    extracurricular, parental_support, online_classes):
    """
    Placeholder prediction function using simple rules.
    Replace this with actual ML model prediction.
    """
    # Simple scoring logic (mimics what a real model might do)
    score = 0
    score += attendance * 0.25
    score += study_hours * 1.2
    score += prev_grade * 0.35
    score += extracurricular * 2

    if parental_support == "High":
        score += 8
    elif parental_support == "Medium":
        score += 4

    if online_classes:
        score += 2

    # Determine prediction and fake probabilities
    if score >= 75:
        prediction = "High"
        probs = {"High": 0.75, "Medium": 0.20, "Low": 0.05}
    elif score >= 55:
        prediction = "Medium"
        probs = {"High": 0.20, "Medium": 0.65, "Low": 0.15}
    else:
        prediction = "Low"
        probs = {"High": 0.10, "Medium": 0.25, "Low": 0.65}

    return prediction, probs

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🎯 Prediction", "📊 Data Exploration", "🤖 Model Info"])

# ============================================================
# PAGE 1: PREDICTION
# ============================================================
if page == "🎯 Prediction":
    st.title("🎓 Student Performance Prediction")
    st.markdown("Enter student information to predict their academic performance level.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Student Information")

        gender = st.selectbox("Gender", ["Male", "Female"])

        attendance = st.slider(
            "Attendance Rate (%)",
            min_value=70, max_value=95, value=85,
            help="Student's class attendance percentage"
        )

        study_hours = st.slider(
            "Study Hours Per Week",
            min_value=8, max_value=30, value=17,
            help="Average hours spent studying per week"
        )

        prev_grade = st.slider(
            "Previous Grade",
            min_value=60, max_value=90, value=75,
            help="Grade from previous semester/year"
        )

    with col2:
        st.subheader("Additional Factors")

        extracurricular = st.slider(
            "Extracurricular Activities",
            min_value=0, max_value=3, value=1,
            help="Number of extracurricular activities (0-3)"
        )

        parental_support = st.selectbox(
            "Parental Support Level",
            ["Low", "Medium", "High"],
            index=1,
            help="Level of parental involvement in education"
        )

        online_classes = st.checkbox(
            "Takes Online Classes",
            help="Whether student participates in online classes"
        )

    st.divider()

    # Predict button
    if st.button("🔮 Predict Performance", type="primary", use_container_width=True):
        prediction, probabilities = predict_performance_placeholder(
            attendance, study_hours, prev_grade,
            extracurricular, parental_support, online_classes
        )

        # Display result
        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            # Color-coded result
            if prediction == "High":
                st.success(f"### Predicted Performance: {prediction} ✅")
                st.balloons()
            elif prediction == "Medium":
                st.warning(f"### Predicted Performance: {prediction} ⚠️")
            else:
                st.error(f"### Predicted Performance: {prediction} ❌")

        with col2:
            # Probability chart
            fig = go.Figure(go.Bar(
                x=list(probabilities.values()),
                y=list(probabilities.keys()),
                orientation='h',
                marker_color=['#ff6b6b', '#feca57', '#48dbfb']
            ))
            fig.update_layout(
                title="Prediction Confidence",
                xaxis_title="Probability",
                yaxis_title="Performance Level",
                height=250
            )
            st.plotly_chart(fig, use_container_width=True)

        # Interpretation
        st.info("""
        **What does this mean?**
        - **High**: Student is expected to score 85+ (Excellent performance)
        - **Medium**: Student is expected to score 70-84 (Good performance)
        - **Low**: Student is expected to score below 70 (Needs improvement)
        """)

# ============================================================
# PAGE 2: DATA EXPLORATION
# ============================================================
elif page == "📊 Data Exploration":
    st.title("📊 Data Exploration")
    st.markdown("Explore the student performance dataset.")

    df = load_data()

    # Dataset overview
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", len(df))
    col2.metric("Features", len(df.columns))
    col3.metric("Performance Levels", df['PerformanceLevel'].nunique())

    st.divider()

    # Sample data
    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

    st.divider()

    # Statistics
    st.subheader("Numerical Statistics")
    numeric_cols = ['AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade', 'FinalGrade']
    st.dataframe(df[numeric_cols].describe(), use_container_width=True)

    st.divider()

    # Visualizations
    st.subheader("Performance Distribution")

    col1, col2 = st.columns(2)

    with col1:
        # Performance level counts
        perf_counts = df['PerformanceLevel'].value_counts()
        fig = px.pie(
            values=perf_counts.values,
            names=perf_counts.index,
            title="Distribution of Performance Levels",
            color_discrete_sequence=['#ff6b6b', '#48dbfb', '#feca57']
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Final grade distribution
        fig = px.histogram(
            df, x='FinalGrade',
            title="Distribution of Final Grades",
            nbins=20,
            color_discrete_sequence=['#48dbfb']
        )
        fig.update_layout(xaxis_title="Final Grade", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    # Performance by parental support
    st.subheader("Performance by Parental Support")
    fig = px.histogram(
        df, x='ParentalSupport', color='PerformanceLevel',
        barmode='group',
        title="Performance Levels by Parental Support",
        color_discrete_sequence=['#ff6b6b', '#48dbfb', '#feca57']
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 3: MODEL INFO
# ============================================================
elif page == "🤖 Model Info":
    st.title("🤖 Model Information")
    st.markdown("Details about the machine learning model used for predictions.")

    # Model description
    st.subheader("About the Model")
    st.info("""
    This application uses a **classification model** to predict student performance levels.

    **Target Variable:** PerformanceLevel (High / Medium / Low)
    - **High**: Final Grade ≥ 85
    - **Medium**: Final Grade 70-84
    - **Low**: Final Grade < 70

    **Models Considered:**
    - Decision Tree Classifier
    - Random Forest Classifier
    - Logistic Regression
    """)

    st.divider()

    # Feature importance (placeholder)
    st.subheader("Feature Importance")
    st.caption("⚠️ Placeholder data - will be updated with actual model results")

    # Mock feature importance data
    features = ['Previous Grade', 'Study Hours/Week', 'Attendance Rate',
                'Parental Support', 'Extracurricular', 'Online Classes', 'Gender']
    importance = [0.28, 0.22, 0.18, 0.14, 0.09, 0.05, 0.04]

    fig = px.bar(
        x=importance, y=features,
        orientation='h',
        title="Feature Importance (Placeholder)",
        color=importance,
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Model performance (placeholder)
    st.subheader("Model Performance Metrics")
    st.caption("⚠️ Placeholder metrics - will be updated after model training")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "85.2%", help="Overall prediction accuracy")
    col2.metric("Precision", "83.7%", help="Positive predictive value")
    col3.metric("Recall", "84.1%", help="True positive rate")
    col4.metric("F1-Score", "83.9%", help="Harmonic mean of precision and recall")

    st.divider()

    # How to use
    st.subheader("How to Update with Real Model")
    st.code("""
# In model.py, after training:
import joblib
joblib.dump(trained_model, 'model.pkl')

# Then modify app.py predict function:
import joblib
model = joblib.load('model.pkl')

def predict_performance(features):
    prediction = model.predict([features])[0]
    probabilities = model.predict_proba([features])[0]
    return prediction, probabilities
    """, language="python")

# Footer
st.sidebar.divider()
st.sidebar.caption("Student Performance Prediction")
st.sidebar.caption("Big Data Project - 2024")
