import streamlit as st
import pandas as pd
import numpy as np
import joblib

kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Segmentation App")
st.write("This app allows you to segment customers based on their features using a pre-trained KMeans model.")



income = st.number_input("Income", min_value=0, max_value=200000, value=50000)
num_store_purchases = st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=10)
num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100,value=10)
num_web_visits_month = st.number_input("Number of Web Visits per Month", min_value=0, max_value=50, value=3)
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)




input_data = pd.DataFrame({
    "Income": [income],
     "NumStorePurchases": [num_store_purchases],
    "NumWebPurchases": [num_web_purchases],
    "NumWebVisitsMonth": [num_web_visits_month],
    "Recency": [recency]
})  



input_scaled = scaler.transform(input_data)


if st.button("Predict segment"):

    cluster = kmeans.predict(input_scaled) [0]


    st.success (f"predicted segment: cluster {cluster}")