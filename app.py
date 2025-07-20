# app.py

import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

data = fetch_california_housing()
feature_names = data.feature_names

st.title("🏠 House Price Prediction App")
st.markdown("Enter the details below to predict the **house price** (in $100,000s).")

# User inputs (create sliders for each feature)
MedInc = st.slider("Median Income (10k USD)", 0.0, 15.0, 3.0)
HouseAge = st.slider("House Age (years)", 1, 52, 20)
AveRooms = st.slider("Average Rooms", 1.0, 10.0, 5.0)
AveBedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0)
Population = st.slider("Population", 100.0, 5000.0, 1000.0)
AveOccup = st.slider("Average Occupants", 1.0, 10.0, 3.0)
Latitude = st.slider("Latitude", 32.0, 42.0, 36.0)
Longitude = st.slider("Longitude", -124.0, -114.0, -120.0)


features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                      Population, AveOccup, Latitude, Longitude]])

# Makes prediction
prediction = model.predict(features)[0]

st.subheader("Predicted House Price")
st.success(f"${prediction * 100000:.2f}")

# Feature importance visualization
st.subheader(" Feature Importances (Model Explanation)")
importances = model.feature_importances_
fig, ax = plt.subplots()
ax.barh(feature_names, importances)
ax.set_xlabel("Importance")
st.pyplot(fig)
