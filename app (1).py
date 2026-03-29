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

    story.append(Paragraph(
        "Climate Change Prediction Report",
        styles['Heading1']
    ))

    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "This dashboard predicts climate impact using temperature and humidity.",
        styles['BodyText']
    ))

    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "Dataset: Global Weather Repository",
        styles['BodyText']
    ))

    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "Model Used: Simple Weighted Prediction",
        styles['BodyText']
    ))

    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "Visualization: Histogram, Bar Chart, Line Chart",
        styles['BodyText']
    ))

    doc = SimpleDocTemplate(buffer, pagesize=A4)
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
