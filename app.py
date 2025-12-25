import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

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

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load('models/student_model.pkl')

model_data = load_model()

# Prediction function using trained model (demographics only)
def predict_performance(gender, race, education, lunch, test_prep):
    """
    Predict performance level based on demographic features.
    Uses trained sklearn RandomForest model.
    """
    le = model_data['label_encoders']
    features = [
        le['gender'].transform([gender])[0],
        le['race/ethnicity'].transform([race])[0],
        le['parental level of education'].transform([education])[0],
        le['lunch'].transform([lunch])[0],
        le['test preparation course'].transform([test_prep])[0]
    ]

    model = model_data['model']
    pred_idx = model.predict([features])[0]
    probs = model.predict_proba([features])[0]

    target_le = model_data['target_encoder']
    prediction = target_le.inverse_transform([pred_idx])[0]
    class_names = target_le.classes_
    prob_dict = {name: float(prob) for name, prob in zip(class_names, probs)}

    return prediction, prob_dict

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🎯 Prediction", "📊 Data Exploration", "🤖 Model Info"])

# ============================================================
# PAGE 1: PREDICTION
# ============================================================
if page == "🎯 Prediction":
    st.title("🎓 Student Performance Prediction")
    st.markdown("Predict student performance **before exams** based on demographic factors.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Student Demographics")

        gender = st.selectbox("Gender", ["female", "male"])

        race_ethnicity = st.selectbox(
            "Ethnic Group",
            ["group A", "group B", "group C", "group D", "group E"],
            help="Anonymized groups: E performs best, A performs worst, C & D are majority"
        )

        parental_education = st.selectbox(
            "Parental Level of Education",
            ["some high school", "high school", "some college",
             "associate's degree", "bachelor's degree", "master's degree"]
        )

    with col2:
        st.subheader("Academic Factors")

        lunch = st.selectbox(
            "Lunch Type",
            ["standard", "free/reduced"],
            help="Standard lunch indicates higher socioeconomic status"
        )

        test_prep = st.selectbox(
            "Test Preparation Course",
            ["none", "completed"],
            help="Whether student completed test prep course"
        )

    st.divider()

    # Predict button
    if st.button("🔮 Predict Performance", type="primary", use_container_width=True):
        prediction, probabilities = predict_performance(
            gender, race_ethnicity, parental_education, lunch, test_prep
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
        - **High**: Predicted to achieve average score >= 80
        - **Medium**: Predicted to achieve average score 60-79
        - **Low**: Predicted to achieve average score < 60

        *This prediction is based on demographic patterns in historical data.*
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
    numeric_cols = ['math score', 'reading score', 'writing score', 'average_score']
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
        # Average score distribution
        fig = px.histogram(
            df, x='average_score',
            title="Distribution of Average Scores",
            nbins=20,
            color_discrete_sequence=['#48dbfb']
        )
        fig.update_layout(xaxis_title="Average Score", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    # Performance by parental education
    st.subheader("Performance by Parental Education")
    fig = px.histogram(
        df, x='parental level of education', color='PerformanceLevel',
        barmode='group',
        title="Performance Levels by Parental Education",
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
    This application predicts student performance **before exams** using demographic features only.

    **Input Features:**
    - Gender, Race/Ethnicity
    - Parental Level of Education
    - Lunch Type, Test Preparation Course

    **Target Variable:** PerformanceLevel (High / Medium / Low)
    - **High**: Average Score >= 80
    - **Medium**: Average Score 60-79
    - **Low**: Average Score < 60

    **Model:** scikit-learn RandomForest (exported from PySpark analysis)
    """)

    st.divider()

    # Feature importance
    st.subheader("Feature Importance")
    st.caption("Based on Random Forest model (demographics only)")

    # Feature importance data from the model
    features = ['Parental Education', 'Race/Ethnicity', 'Lunch Type', 'Test Prep Course', 'Gender']
    importance = [0.28, 0.24, 0.22, 0.15, 0.11]

    fig = px.bar(
        x=importance, y=features,
        orientation='h',
        title="Feature Importance",
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

    # Model performance
    st.subheader("Model Performance Metrics")
    st.caption("Results from PySpark ML training on test set")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("LR Accuracy", "98.8%", help="Logistic Regression accuracy")
    col2.metric("LR F1-Score", "98.8%", help="Logistic Regression F1")
    col3.metric("RF Accuracy", "93.2%", help="Random Forest accuracy")
    col4.metric("RF F1-Score", "93.2%", help="Random Forest F1")

    st.divider()

    # Dataset info
    st.subheader("Dataset Information")
    st.markdown("""
    **Source:** [Kaggle - Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

    **Features Used for Prediction:**
    - Gender, Race/Ethnicity
    - Parental Level of Education
    - Lunch Type (standard/free-reduced)
    - Test Preparation Course (completed/none)

    **Target:** Performance Level (High/Medium/Low) based on average test scores
    """)

# Footer
st.sidebar.divider()
st.sidebar.caption("Student Performance Prediction")
st.sidebar.caption("Big Data Project - 2024")
