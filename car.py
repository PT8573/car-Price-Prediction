import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

# -----------------------------
# BACKGROUND IMAGE + STYLING
# -----------------------------
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1492144534655-ae79c964c9d7");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Title */
h1 {
    color: white !important;
    text-shadow: 2px 2px 8px black;
}

/* Sub headers */
h2, h3 {
    color: #f1f1f1 !important;
    text-shadow: 1px 1px 5px black;
}

/* Text */
p, label {
    color: white !important;
    font-weight: 500;
}

/* Selectbox styling */
div[data-baseweb="select"] {
    background-color: rgba(255,255,255,0.9);
    border-radius: 8px;
}

/* Button */
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    font-size: 16px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.title("🚗 Car Price Prediction App")
st.markdown("Predict used car price using Machine Learning")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("carprice.csv")
df = df.dropna()

# -----------------------------
# LABEL ENCODING
# -----------------------------
encoders = {}

for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# -----------------------------
# FEATURES & TARGET
# -----------------------------
X = df.drop("price", axis=1)
y = df["price"]

# -----------------------------
# TRAIN MODEL
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------
st.subheader("📊 Model Performance")
st.write("R2 Score:", r2_score(y_test, y_pred))
st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))



# -----------------------------
# INPUT SECTION
# -----------------------------
st.subheader("Enter Car Details")

input_dict = {}

for col in X.columns:
    if col in encoders:
        options = encoders[col].classes_
        input_dict[col] = st.selectbox(col, options)
    else:
        input_dict[col] = st.slider(
            col,
            float(df[col].min()),
            float(df[col].max()),
            float(df[col].mean())
        )

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🚗 Predict Price"):
    input_df = pd.DataFrame([input_dict])

    for col in encoders:
        if col in input_df.columns:
            le = encoders[col]
            input_df[col] = input_df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )

    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Car Price: ₹ {prediction:,.2f}")