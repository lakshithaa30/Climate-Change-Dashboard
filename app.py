
import streamlit as st
import pandas as pd
import numpy as np

st.title("Climate Change Prediction Dashboard")

st.header("Project Overview")
st.write("Climate Change Prediction using Data Science")

data = pd.DataFrame({
    'Temperature':[30,32,34,35,36],
    'Humidity':[60,65,70,75,80]
})

st.header("Dataset")
st.write(data)

st.header("Temperature Chart")
st.line_chart(data['Temperature'])

st.success("Dashboard Loaded Successfully")
