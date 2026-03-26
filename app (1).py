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

# Try importing XGBoost safely
try:
    from xgboost import XGBRegressor
    xgb_available = True
except:
    xgb_available = False


st.title("🌍 Climate Change Model Visualization Dashboard")

zip_path = "GlobalWeatherRepository.zip"

if os.path.exists(zip_path):

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_names = zip_ref.namelist()

        csv_file = None
        for file in file_names:
            if file.endswith(".csv"):
                csv_file = file

        if csv_file:

            data = pd.read_csv(zip_ref.open(csv_file))

            st.success("Dataset Loaded Successfully")
            st.write(data.head())

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

            lr.fit(X_train, y_train)
            dt.fit(X_train, y_train)
            rf.fit(X_train, y_train)
            gb.fit(X_train, y_train)

            lr_pred = lr.predict(X_test)
            dt_pred = dt.predict(X_test)
            rf_pred = rf.predict(X_test)
            gb_pred = gb.predict(X_test)

            st.header("Model Visualization")

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

            # XGBoost (Optional)
            if xgb_available:
                xgb = XGBRegressor()
                xgb.fit(X_train, y_train)
                xgb_pred = xgb.predict(X_test)

                st.subheader("XGBoost")
                fig, ax = plt.subplots()
                ax.plot(y_test.values)
                ax.plot(xgb_pred)
                st.pyplot(fig)

            else:
                st.warning("XGBoost not available")

        else:
            st.error("CSV not found inside ZIP")

else:
    st.error("ZIP file not found")
