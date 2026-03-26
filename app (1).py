import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import zipfile

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


# ----------------------------------
# Dashboard Title
# ----------------------------------
st.title("🌍 Climate Change Prediction Dashboard")

st.write("This app predicts climate impact based on Temperature and Humidity.")


# ----------------------------------
# Prediction Section
# ----------------------------------
st.header("🔮 Climate Impact Prediction")

temp = st.slider("Temperature (°C)", 0, 50, 25)
humidity = st.slider("Humidity (%)", 0, 100, 50)

if st.button("Predict"):
    prediction = (temp * 0.6) + (humidity * 0.4)
    st.subheader("Prediction Result")
    st.write("Predicted Climate Impact:", round(prediction, 2))


# ----------------------------------
# Sample Visualization
# ----------------------------------
st.header("📊 Temperature & Humidity Trends")

sample_data = pd.DataFrame({
    "Temperature": [20, 22, 25, 27, 30],
    "Humidity": [40, 45, 50, 55, 60]
})

st.write(sample_data)
st.line_chart(sample_data)


# ----------------------------------
# Load Dataset from ZIP
# ----------------------------------
st.header("📂 Dataset Analysis")

zip_path = "GlobalWeatherRepository.zip"
csv_file = "GlobalWeatherRepository.csv"

if os.path.exists(zip_path):

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()

    data = pd.read_csv(csv_file)

    st.subheader("Dataset Preview")
    st.write(data.head())

    # ----------------------------------
    # Model Training
    # ----------------------------------
    st.header("🤖 Model Comparison")

    target = st.selectbox("Select Target Column", data.columns)

    X = data.drop(target, axis=1)
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    # Models
    lr = LinearRegression()
    rf = RandomForestRegressor()
    gb = GradientBoostingRegressor()
    xgb = XGBRegressor()


    # Training
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    xgb.fit(X_train, y_train)


    # Predictions
    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)
    gb_pred = gb.predict(X_test)
    xgb_pred = xgb.predict(X_test)


    # Metrics
    models = ["Linear Regression", "Random Forest", "Gradient Boost", "XGBoost"]

    r2_scores = [
        r2_score(y_test, lr_pred),
        r2_score(y_test, rf_pred),
        r2_score(y_test, gb_pred),
        r2_score(y_test, xgb_pred)
    ]

    mae_scores = [
        mean_absolute_error(y_test, lr_pred),
        mean_absolute_error(y_test, rf_pred),
        mean_absolute_error(y_test, gb_pred),
        mean_absolute_error(y_test, xgb_pred)
    ]


    # ----------------------------------
    # Visualization
    # ----------------------------------
    st.subheader("📈 Model Comparison - R2 Score")

    fig, ax = plt.subplots()
    ax.bar(models, r2_scores)
    plt.xticks(rotation=45)
    st.pyplot(fig)


    st.subheader("📉 Model Comparison - MAE")

    fig, ax = plt.subplots()
    ax.bar(models, mae_scores)
    plt.xticks(rotation=45)
    st.pyplot(fig)

else:
    st.error("Dataset ZIP file not found. Please upload GlobalWeatherRepository.zip")
