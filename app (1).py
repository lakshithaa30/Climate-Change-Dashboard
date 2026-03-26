import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import zipfile

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor


# -----------------------------
# Title
# -----------------------------
st.title("🌍 Climate Change Model Visualization Dashboard")


# -----------------------------
# Load Dataset
# -----------------------------
zip_path = "GlobalWeatherRepository.zip"

if os.path.exists(zip_path):

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()

    # Automatically find CSV file
    csv_files = [f for f in os.listdir() if f.endswith('.csv')]

    if csv_files:
        data = pd.read_csv(csv_files[0])

        st.success("Dataset Loaded Successfully ✅")
        st.write(data.head())

        # -----------------------------
        # Select Target
        # -----------------------------
        target = st.selectbox("Select Target Column", data.columns)

        X = data.drop(target, axis=1)
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Models
        lr = LinearRegression()
        dt = DecisionTreeRegressor()
        rf = RandomForestRegressor()
        gb = GradientBoostingRegressor()
        xgb = XGBRegressor()

        # Train
        lr.fit(X_train, y_train)
        dt.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train)
        xgb.fit(X_train, y_train)

        # Predict
        lr_pred = lr.predict(X_test)
        dt_pred = dt.predict(X_test)
        rf_pred = rf.predict(X_test)
        gb_pred = gb.predict(X_test)
        xgb_pred = xgb.predict(X_test)

        # Visualization
        st.header("Model Visualizations")

        # Linear Regression
        st.subheader("Linear Regression")
        fig, ax = plt.subplots()
        ax.plot(y_test.values)
        ax.plot(lr_pred)
        st.pyplot(fig)

        # Decision Tree
        st.subheader("Decision Tree")
        fig, ax = plt.subplots()
        ax.plot(y_test.values)
        ax.plot(dt_pred)
        st.pyplot(fig)

        # Random Forest
        st.subheader("Random Forest")
        fig, ax = plt.subplots()
        ax.plot(y_test.values)
        ax.plot(rf_pred)
        st.pyplot(fig)

        # Gradient Boosting
        st.subheader("Gradient Boosting")
        fig, ax = plt.subplots()
        ax.plot(y_test.values)
        ax.plot(gb_pred)
        st.pyplot(fig)

        # XGBoost
        st.subheader("XGBoost")
        fig, ax = plt.subplots()
        ax.plot(y_test.values)
        ax.plot(xgb_pred)
        st.pyplot(fig)

    else:
        st.error("CSV file not found inside ZIP")
    st.error("ZIP file not found")
