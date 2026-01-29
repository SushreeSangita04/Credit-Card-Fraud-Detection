import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

@st.cache_data
def load_and_train():
    # Load small dataset from GitHub
    url = "https://raw.githubusercontent.com/SushreeSangita04/Credit-Card-Fraud-Detection/main/new.csv"
    data = pd.read_csv(url)

    X = data.drop(columns='Class', axis=1)
    y = data['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, X.columns

model, feature_names = load_and_train()

st.title("Credit Card Fraud Detection")
st.write("Enter transaction feature values separated by commas:")

input_data = st.text_input(f"Enter {len(feature_names)} values:")

if st.button("Predict"):
    try:
        input_list = input_data.split(",")
        features = np.array(input_list, dtype=float)
        prediction = model.predict(features.reshape(1, -1))

        if prediction[0] == 0:
            st.success("Legitimate Transaction")
        else:
            st.error("Fraudulent Transaction")

    except:
        st.warning("⚠ Please enter valid numeric values separated by commas.")
