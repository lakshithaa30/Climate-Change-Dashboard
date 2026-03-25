import streamlit as st
import pandas as pd

st.title("🌍 Climate Change Prediction Dashboard")

st.write("This app predicts climate impact based on Temperature and Humidity.")

temp = st.slider("Temperature (°C)", 0, 50, 25)
humidity = st.slider("Humidity (%)", 0, 100, 50)

if st.button("Predict"):
    prediction = (temp * 0.6) + (humidity * 0.4)
    st.subheader("Prediction Result")
    st.write("Predicted Climate Impact:", round(prediction, 2))

st.header("Temperature & Humidity Trends")

data = pd.DataFrame({
    "Temperature": [20, 22, 25, 27, 30],
    "Humidity": [40, 45, 50, 55, 60]
})

st.write(data)
st.line_chart(data)
