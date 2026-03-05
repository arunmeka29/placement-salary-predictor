import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Title
st.title("Placement Salary Prediction")

# Read dataset
df = pd.read_csv("placements.csv")

# Select required columns
df = df[['coding_skills','salary_package_lpa']]
df = df.dropna()
df = df.head(1000)

# Features and target
X = df[['coding_skills']]
y = df['salary_package_lpa']

# Train model
model = LinearRegression()
model.fit(X, y)

st.success("Model Trained Successfully")

# User input
coding_skill = st.number_input(
    "Enter Coding Skills Score",
    min_value=0.0,
    max_value=10.0,
    step=0.1
)

# Prediction
if st.button("Predict Salary"):
    prediction = model.predict([[coding_skill]])
    st.write("Predicted Salary Package (LPA):", round(prediction[0],2))