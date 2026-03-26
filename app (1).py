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

st.write("Machine Learning Model Visualization")


# -----------------------------
# Load Dataset from ZIP
# -----------------------------
zip_path = "GlobalWeatherRepository.zip"

if os.path.exists(zip_path):

    # Extract ZIP
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("data")

    # Find CSV inside ZIP
    csv_file = None
    for root, dirs, files in os.walk("data"):
        for file in files:
            if file.endswith(".csv"):
                csv_file = os.path.join(root, file)

    if csv_file:

        data = pd.read_csv(csv_file)

        st.success("Dataset Loaded Successfully ✅")

        # -----------------------------
        # Dataset Preview
        # -----------------------------
        st.subheader("Dataset Preview")
        st.write(data.head())


        # -----------------------------
        # Select Target Column
        # -----------------------------
        st.subheader("Select Target Column")

        target = st.selectbox("Target Column", data.columns)

        X = data.drop(target, axis=1)
        y = data[target]


        # -----------------------------
        # Train Test Split
        # -----------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )


        # -----------------------------
        # Models
        # -----------------------------
        lr = LinearRegression()
        dt = DecisionTreeRegressor()
        rf = RandomForestRegressor()
        gb = GradientBoostingRegressor()
        xgb = XGBRegressor()


        # -----------------------------
        # Train Models
        # -----------------------------
        lr.fit(X_train, y_train)
        dt.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train)
        xgb.fit(X_train, y_train)


        # -----------------------------
        # Predictions
        # -----------------------------
        lr_pred = lr.predict(X_test)
        dt_pred = dt.predict(X_test)
        rf_pred = rf.predict(X_test)
        gb_pred = gb.predict(X_test)
        xgb_pred = xgb.predict(X_test)


        # -----------------------------
        # Visualization
        # -----------------------------
        st.header("📊 Model Visualization")


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

else:
    st.error("ZIP file not found")
