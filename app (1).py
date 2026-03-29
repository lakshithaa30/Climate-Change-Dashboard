import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import io

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4


st.title("Climate Change Prediction Dashboard")
st.write("This app predicts climate impact based on Temperature and Humidity.")


# User Input
st.header("Enter Climate Data")

temp = st.slider("Temperature (°C)", 0, 50, 25)
humidity = st.slider("Humidity (%)", 0, 100, 50)


# Prediction Logic
if st.button("Predict"):
    prediction = (temp * 0.6) + (humidity * 0.4)

    st.subheader("Prediction Result")
    st.write("Predicted Climate Impact:", round(prediction, 2))


# Sample Data Visualization
st.header("Temperature & Humidity Trends")

data = pd.DataFrame({
    "Temperature": [20, 22, 25, 27, 30],
    "Humidity": [40, 45, 50, 55, 60]
})

# Show table
st.write(data)

# Show chart
st.line_chart(data)


# Load dataset from ZIP file
zip_file = "GlobalWeatherRepository.zip"

with zipfile.ZipFile(zip_file, 'r') as z:
    file_name = z.namelist()[0]
    data = pd.read_csv(z.open(file_name))


st.subheader("Dataset Preview")
st.write(data.head())


# Data Visualization
st.subheader("Data Visualization")

numeric_columns = data.select_dtypes(include=['float64','int64']).columns

selected_column = st.selectbox(
    "Select column to visualize",
    numeric_columns
)


# Histogram
st.subheader("Histogram")
fig, ax = plt.subplots()
ax.hist(data[selected_column])
st.pyplot(fig)


# Bar Chart
st.subheader("Bar Chart")
st.bar_chart(data[selected_column].value_counts().head(10))


# -----------------------------
# Report Generation Function
# -----------------------------
def create_report():
    buffer = io.BytesIO()
    styles = getSampleStyleSheet()
    story = []

    report_text = """
    Climate Change Prediction Dashboard Report

    1. Introduction
    Climate change is one of the most significant global challenges affecting ecosystems,
    weather patterns, and human life. This project focuses on developing a climate prediction dashboard.

    2. Objectives
    - Analyze climate-related data
    - Visualize weather trends
    - Predict climate impact
    - Build interactive dashboard

    3. Dataset Description
    Global Weather Repository dataset is used which contains:
    - Temperature
    - Humidity
    - Wind Speed
    - Air Quality
    - Location Details

    4. Methodology
    Data collection, preprocessing, visualization and prediction steps were performed.

    5. Prediction Model
    Climate Impact = (Temperature × 0.6) + (Humidity × 0.4)

    6. Dashboard Features
    - User Input
    - Prediction
    - Visualization
    - Dataset preview
    - Download report

    7. Tools Used
    - Python
    - Streamlit
    - Pandas
    - Matplotlib

    8. Results
    The dashboard successfully predicts climate impact.

    9. Future Work
    - Add machine learning
    - Improve accuracy
    """

    story.append(Paragraph("Climate Change Prediction Report", styles['Heading1']))
    story.append(Spacer(1,12))
    story.append(Paragraph(report_text, styles['BodyText']))

    doc = SimpleDocTemplate(buffer)
    doc.build(story)

    buffer.seek(0)
    return buffer


# -----------------------------
# Download Button
# -----------------------------

st.subheader("Download Report")

report = create_report()

st.download_button(
    label="Download Climate Report",
    data=report,
    file_name="Climate_Report.pdf",
    mime="application/pdf"
)
