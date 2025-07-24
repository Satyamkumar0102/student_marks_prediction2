import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

# Load saved model components
model = joblib.load("exam_score_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Load original dataset for reference and plotting
original_df = pd.read_csv("data.csv")
columns_to_drop = ['Learning_Disabilities', 'Gender', 'School_Type']
original_df.drop(columns=[col for col in columns_to_drop if col in original_df.columns], inplace=True)
for col in original_df.select_dtypes(include='object').columns:
    original_df[col] = label_encoders[col].transform(original_df[col])

st.set_page_config(page_title="Exam Score Predictor", layout="wide")
st.title("📘 Student Exam Score Prediction UI")

# --- Sidebar inputs --- #
st.sidebar.header("🔧 Student Input Features")
input_data = {
    'Hours_Studied': st.sidebar.slider("Hours Studied", 0, 12, 6),
    'Attendance': st.sidebar.slider("Attendance (%)", 50, 100, 90),
    'Parental_Involvement': st.sidebar.slider("Parental Involvement", 1, 10, 7),
    'Access_to_Resources': st.sidebar.slider("Access to Resources", 1, 10, 7),
    'Extracurricular_Activities': st.sidebar.slider("Extracurricular Activities", 1, 10, 5),
    'Sleep_Hours': st.sidebar.slider("Sleep Hours", 4, 10, 7),
    'Previous_Scores': st.sidebar.slider("Previous Scores", 0, 100, 85),
    'Motivation_Level': st.sidebar.slider("Motivation Level", 1, 10, 7),
    'Internet_Access': label_encoders['Internet_Access'].transform([st.sidebar.selectbox("Internet Access", ['Yes', 'No'])])[0],
    'Tutoring_Sessions': st.sidebar.slider("Tutoring Sessions", 0, 10, 2),
    'Family_Income': st.sidebar.slider("Family Income (k)", 10, 150, 80),
    'Teacher_Quality': st.sidebar.slider("Teacher Quality", 1, 10, 7),
    'Peer_Influence': label_encoders['Peer_Influence'].transform([st.sidebar.selectbox("Peer Influence", ['Positive', 'Negative'])])[0],
    'Physical_Activity': st.sidebar.slider("Physical Activity", 1, 10, 5),
    'Parental_Education_Level': label_encoders['Parental_Education_Level'].transform([
        st.sidebar.selectbox("Parental Education", label_encoders['Parental_Education_Level'].classes_.tolist())
    ])[0],
    'Distance_from_Home': st.sidebar.slider("Distance from Home (km)", 0, 20, 5),
}

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)

# --- Prediction and Confidence --- #
if st.button("🔍 Predict Exam Score"):
    predicted_score = model.predict(input_scaled)[0]
    preds = np.stack([tree.predict(input_scaled) for tree in model.estimators_])
    confidence = np.std(preds, axis=0)[0]

    st.subheader("🎯 Prediction Result")
    st.metric("Predicted Exam Score", f"{predicted_score:.2f} marks")
    st.metric("Confidence ±", f"{confidence:.2f} marks")

    # --- SHAP Contribution --- #
    st.subheader("📊 Feature Contribution (SHAP Values)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)
    shap.summary_plot(shap_values, features=input_df, feature_names=input_df.columns, plot_type="bar", show=False)
    fig = plt.gcf()
    st.pyplot(fig)

    # --- Input vs Dataset Average --- #
    st.subheader("📈 Input vs Dataset Average")
    avg_vals = original_df.mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(avg_vals.values, label="Dataset Average", marker='o')
    ax.plot(input_df.iloc[0].values, label="Input Data", marker='x')
    ax.set_xticks(range(len(input_df.columns)))
    ax.set_xticklabels(input_df.columns, rotation=90)
    ax.set_title("Input Features vs Dataset Average")
    ax.legend()
    st.pyplot(fig)

    # --- Partial Dependence Plot for Previous Scores --- #
    st.subheader("🧠 Partial Dependence Plot: Previous Scores")
    fig, ax = plt.subplots(figsize=(8, 4))
    display = PartialDependenceDisplay.from_estimator(model, original_df.drop(columns='Exam_Score'), ['Previous_Scores'], ax=ax)
    st.pyplot(fig)
else:
    st.info("Enter student features from the sidebar and click 'Predict Exam Score'")
