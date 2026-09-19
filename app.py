import streamlit as st
import joblib
import pandas as pd


try:
    model = joblib.load("house_price_model.pkl")
except FileNotFoundError:
    st.error("Model file not found. Please train the model first by running house_price.py.")
    st.stop()

st.title("🏠 House Price Prediction")
st.write("Enter the house details to predict the estimated price.")

area = st.number_input("House Area (sq ft)", min_value=1.0, max_value=20000.0, value=2000.0, step=100.0)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=20, value=3, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=20, value=2, step=1)
age = st.number_input("House Age (years)", min_value=0, max_value=100, value=5, step=1)

if st.button("Predict Price"):
    if area <= 0:
        st.error("Please enter a valid house area greater than 0.")
    elif bedrooms <= 0:
        st.error("Bedrooms must be greater than 0.")
    elif bathrooms <= 0:
        st.error("Bathrooms must be greater than 0.")
    else:
        new_house = pd.DataFrame(
            [[area, bedrooms, bathrooms, age]],
            columns=["area", "bedrooms", "bathrooms", "age"],
        )

        predicted_price = model.predict(new_house)[0]
        predicted_price = max(predicted_price, 0)
        st.success(f"Estimated House Price: ₹{predicted_price:,.2f}")