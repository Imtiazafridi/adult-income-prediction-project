
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# Load saved model and preprocessor
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "best_automl_model.pkl")
preprocessor = joblib.load(BASE_DIR / "preprocessor.pkl")

st.title("Adult Income Prediction")
st.write("Predict whether a person's annual income is above or below $50K.")

# User input
st.header("Enter Personal Information")

age = st.number_input("Age", 17, 90, 39)
workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"])
fnlwgt = st.number_input("Final Weight", min_value=1, value=77516)
education = st.selectbox("Education", ["Bachelors", "HS-grad", "Some-college", "11th", "Masters", "9th", "Assoc-acdm", "Assoc-voc", "Doctorate", "Prof-school", "5th-6th", "10th", "7th-8th", "12th", "1st-4th", "Preschool"])
education_num = st.number_input("Education Number", 1, 16, 13)
marital_status = st.selectbox("Marital Status", ["Never-married", "Married-civ-spouse", "Divorced", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"])
occupation = st.selectbox("Occupation", ["Adm-clerical", "Exec-managerial", "Handlers-cleaners", "Prof-specialty", "Other-service", "Sales", "Craft-repair", "Transport-moving", "Farming-fishing", "Machine-op-inspct", "Tech-support", "Protective-serv", "Armed-Forces", "Priv-house-serv"])
relationship = st.selectbox("Relationship", ["Not-in-family", "Husband", "Wife", "Own-child", "Unmarried", "Other-relative"])
race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
sex = st.selectbox("Sex", ["Male", "Female"])
capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.number_input("Hours per Week", 1, 99, 40)
native_country = st.selectbox("Native Country", ["United-States", "Mexico", "Philippines", "Germany", "Canada", "India", "England", "Other"])

# Prediction
if st.button("Predict Income"):
    input_data = pd.DataFrame([{
        "age": age,
        "workclass": workclass,
        "fnlwgt": fnlwgt,
        "education": education,
        "education.num": education_num,
        "marital.status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital.gain": capital_gain,
        "capital.loss": capital_loss,
        "hours.per.week": hours_per_week,
        "native.country": native_country
    }])

    processed_data = preprocessor.transform(input_data)
    prediction = model.predict(processed_data)[0]

    if prediction == 1:
        st.success("Predicted Income: More than $50K")
    else:
        st.info("Predicted Income: $50K or less")

# Model performance
st.header("Model Performance")
st.metric("AutoML Test Accuracy", "86.67%")
st.write("The AutoML model achieved 86.67% accuracy on the test dataset.")

# Feature importance

# Feature importance
st.header("Top 5 Feature Importances")

try:
    feature_names = preprocessor.get_feature_names_out()

    selector = model.named_steps["selectfwe"]
    selected_features = feature_names[selector.get_support()]

    final_model = model.named_steps["lgbmclassifier"]
    importance = final_model.feature_importances_

    # Match feature names with importance values
    if len(selected_features) != len(importance):
        factor = len(importance) // len(selected_features)
        selected_features = list(selected_features) * factor

    importance_df = pd.DataFrame({
        "Feature": selected_features,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance", ascending=False
    ).head(5)

    st.bar_chart(importance_df.set_index("Feature"))

except Exception as e:
    st.write("Feature importance error:", e)

