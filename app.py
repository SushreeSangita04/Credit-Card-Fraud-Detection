import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

@st.cache_data
def load_and_train():
    data = pd.read_csv("creditcard.csv")

    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]

    legit_sample = legit.sample(n=492, random_state=2)
    new_dataset = pd.concat([legit_sample, fraud], axis=0)

    X = new_dataset.drop(columns='Class', axis=1)
    y = new_dataset['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, X.columns

model, feature_names = load_and_train()

st.title("Credit Card Fraud Detection")
st.write("Enter feature values separated by commas:")

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
        st.warning("Please enter valid numeric values separated by commas.")
