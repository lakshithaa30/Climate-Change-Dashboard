
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import io

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_JUSTIFY


st.title("Climate Change Prediction Dashboard")
st.write("This app predicts climate impact based on Temperature and Humidity.")


# User Input
st.header("Enter Climate Data")

temp = st.slider("Temperature (°C)", 0, 50, 25)
humidity = st.slider("Humidity (%)", 0, 100, 50)


# Prediction
if st.button("Predict"):
    prediction = (temp * 0.6) + (humidity * 0.4)

    st.subheader("Prediction Result")
    st.write("Predicted Climate Impact:", round(prediction, 2))


# Sample Data
st.header("Temperature & Humidity Trends")

data = pd.DataFrame({
    "Temperature": [20, 22, 25, 27, 30],
    "Humidity": [40, 45, 50, 55, 60]
})

st.write(data)
st.line_chart(data)


# Load dataset
zip_file = "GlobalWeatherRepository.zip"

with zipfile.ZipFile(zip_file, 'r') as z:
    file_name = z.namelist()[0]
    data = pd.read_csv(z.open(file_name))


st.subheader("Dataset Preview")
st.write(data.head())


# Visualization
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
# Report Generation
# -----------------------------

def create_report():

    buffer = io.BytesIO()
    styles = getSampleStyleSheet()

    story = []

    # Title
    story.append(Paragraph(
        "Climate Change Prediction Dashboard Report",
        styles['Heading1']
    ))

    story.append(Spacer(1,12))


    # Introduction
    story.append(Paragraph(
        "<b>1. Introduction</b><br/>"
        "Climate change is one of the most critical global challenges affecting "
        "weather patterns, ecosystems, and human life. Rising temperatures, "
        "extreme weather conditions, and environmental changes require proper "
        "analysis and prediction. This project develops a climate prediction "
        "dashboard using data science techniques to predict climate impact.",
        styles['BodyText']
    ))

    story.append(Spacer(1,12))


    # Objectives
    story.append(Paragraph(
        "<b>2. Objectives</b><br/>"
        "- Analyze climate-related data<br/>"
        "- Visualize weather trends<br/>"
        "- Predict climate impact<br/>"
        "- Build interactive dashboard",
        styles['BodyText']
    ))

    story.append(Spacer(1,12))


    # Dataset
    story.append(Paragraph(
        "<b>3. Dataset Description</b><br/>"
        "Global Weather Repository dataset is used which contains weather data "
        "including temperature, humidity, wind speed, air quality and location details.",
        styles['BodyText']
    ))

    story.append(Spacer(1,12))


    # Methodology
    story.append(Paragraph(
        "<b>4. Methodology</b><br/>"
        "The dataset is collected, cleaned, and visualized. Prediction model "
        "is applied to predict climate impact.",
        styles['BodyText']
    ))

    story.append(Spacer(1,12))


    # Model
    story.append(Paragraph(
        "<b>5. Prediction Model</b><br/>"
        "Climate Impact = (Temperature × 0.6) + (Humidity × 0.4)",
        styles['BodyText']
    ))

    story.append(PageBreak())


    # Page 2

    story.append(Paragraph(
        "<b>6. Dashboard Features</b><br/>"
        "- User Input<br/>"
        "- Prediction<br/>"
        "- Visualization<br/>"
        "- Dataset Preview<br/>"
        "- Download Report",
        styles['BodyText']
    ))

    story.append(Spacer(1,12))


    # Tools
    story.append(Paragraph(
        "<b>7. Tools and Technologies</b><br/>"
        "Python, Streamlit, Pandas, Matplotlib, Seaborn",
        styles['BodyText']
    ))

    story.append(Spacer(1,12))


    # Results
    story.append(Paragraph(
        "<b>8. Results</b><br/>"
        "The dashboard successfully predicts climate impact and visualizes "
        "weather trends effectively.",
        styles['BodyText']
    ))

    story.append(Spacer(1,12))


    # Advantages
    story.append(Paragraph(
        "<b>9. Advantages</b><br/>"
        "- Easy to use<br/>"
        "- Interactive dashboard<br/>"
        "- Fast prediction",
        styles['BodyText']
    ))

    story.append(Spacer(1,12))


    # Future Work
    story.append(Paragraph(
        "<b>10. Future Enhancements</b><br/>"
        "- Add Machine Learning Models<br/>"
        "- Add Time Series Prediction<br/>"
        "- Improve accuracy",
        styles['BodyText']
    ))

    story.append(Spacer(1,12))


    # Conclusion
    story.append(Paragraph(
        "<b>11. Conclusion</b><br/>"
        "The project successfully builds a climate prediction dashboard "
        "using data science techniques.",
        styles['BodyText']
    ))


    doc = SimpleDocTemplate(buffer, pagesize=A4)
    doc.build(story)

    buffer.seek(0)
    return buffer


# Download Button

st.subheader("Download Report")

report = create_report()

st.download_button(
    label="Download Climate Report",
    data=report,
    file_name="Climate_Report.pdf",
    mime="application/pdf"
)
