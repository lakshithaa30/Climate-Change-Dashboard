import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

st.title("Climate Change Prediction Dashboard")

# Load Dataset
data = pd.read_csv("GlobalWeatherRepository.csv")

st.subheader("Dataset Preview")
st.write(data.head())

# Select Target Column
target = st.selectbox("Select Target Column", data.columns)

X = data.drop(target, axis=1)
y = data[target]

# Train Test Split
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

# Visualization
st.subheader("Model Comparison - R2 Score")

fig, ax = plt.subplots()
ax.bar(models, r2_scores)
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("Model Comparison - MAE")

fig, ax = plt.subplots()
ax.bar(models, mae_scores)
plt.xticks(rotation=45)
st.pyplot(fig)
