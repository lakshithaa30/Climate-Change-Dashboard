import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

st.title("Climate Change Prediction Dashboard")

# Load dataset from ZIP file
zip_file = "GlobalWeatherRepository.zip"

with zipfile.ZipFile(zip_file, 'r') as z:
    file_name = z.namelist()[0]   # Get CSV inside ZIP
    data = pd.read_csv(z.open(file_name))

st.subheader("Dataset Preview")
st.write(data.head())

# Select Column for Visualization
st.subheader("Data Visualization")

numeric_columns = data.select_dtypes(include=['float64','int64']).columns

selected_column = st.selectbox("Select column to visualize", numeric_columns)

# Histogram
st.subheader("Histogram")
fig, ax = plt.subplots()
ax.hist(data[selected_column])
st.pyplot(fig)

# Line Chart
st.subheader("Line Chart")
st.line_chart(data[selected_column])

# Bar Chart
st.subheader("Bar Chart")
st.bar_chart(data[selected_column].value_counts().head(10))
