import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)

# Train the Logistic Regression model
clf = LogisticRegression(random_state=0, max_iter=5000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

# Streamlit app
st.title("Breast Cancer Prediction using Logistic Regression")
st.write("This app uses a Logistic Regression model to predict whether breast cancer is malignant or benign.")

# Display the dataset description
if st.checkbox("Show Dataset Description"):
    st.write(data.DESCR)

# Display the model's accuracy
st.subheader("Model Accuracy")
st.write(f"Accuracy of the model: {accuracy:.2f}%")

# User input for prediction
st.subheader("Predict Breast Cancer")
st.write("Input the features to predict the outcome.")

# Create input fields for the 30 features
input_features = []
for i, feature in enumerate(data.feature_names):
    value = st.number_input(f"{feature}", value=0.0)
    input_features.append(value)

# Predict button
if st.button("Predict"):
    input_data = np.array(input_features).reshape(1, -1)
    prediction = clf.predict(input_data)[0]
    result = "Malignant" if prediction == 0 else "Benign"
    st.write(f"The predicted outcome is: **{result}**")
