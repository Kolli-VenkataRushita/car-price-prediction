# Car Price Prediction

## Table of Contents
- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Usage](#usage)
- [Results](#results)
- [Contributors](#contributors)
- [Deployment Process](#deployment-process)

---

## Project Overview
The **Car Price Prediction** project aims to predict the selling price of used cars based on various features such as age, brand, mileage, fuel type, transmission type, etc. The goal is to build a predictive model using **Linear Regression** to estimate the car price accurately, helping users or businesses make informed decisions when buying or selling cars.

This project includes data analysis, feature selection, and model building using a Jupyter notebook, leveraging Python libraries for data preprocessing and machine learning.

---

## Tech Stack
- **Programming Language**: Python
- **Libraries**:
  - `pandas` - For data manipulation and cleaning
  - `numpy` - For numerical operations
  - `matplotlib` and `seaborn` - For data visualization
  - `scikit-learn` - For model building and evaluation
  - `streamlit` - For web app deployment

---

## Dataset
The dataset used for this project contains information on used cars with the following features:
- **Car Name**: Brand and model of the car
- **Year**: The year the car was purchased
- **Selling Price**: The price at which the car was sold (Target variable)
- **Present Price**: The current ex-showroom price of the car
- **Kms Driven**: The distance the car has been driven in kilometers
- **Fuel Type**: Type of fuel (Petrol/Diesel)
- **Seller Type**: Whether the seller is a dealer or an individual
- **Transmission**: Manual or automatic transmission
- **Owner**: Number of previous owners

You can find the dataset in the `car_price_data.csv` file in the project folder.

---

## Installation
1. **Install Python and Required Libraries**:
   - Install Python (>= 3.7) from [here](https://www.python.org/downloads/).
   - Install the necessary libraries:
     ```bash
     pip install pandas numpy matplotlib seaborn scikit-learn streamlit
     ```
2. **Dataset**:
   - Place the `car_price_data.csv` file in the project directory.

---

## Model Training and Evaluation

### Data Preprocessing:
- Cleaned the dataset by removing irrelevant or missing values.
- Categorical features like **Fuel Type**, **Seller Type**, and **Transmission** were converted into numerical form using **One-Hot Encoding**.
- Features like **Year** were transformed into **Car Age** to give a better indication of the car's value depreciation.

### Model Selection:
- A **Linear Regression** model was chosen due to its effectiveness in predicting continuous variables such as price.
- The dataset was split into training and test sets using **train_test_split** to validate the model.

### Model Evaluation:
- The model was evaluated using **Mean Absolute Error (MAE)** and **R-squared scores** to measure how well the predicted prices matched the actual selling prices.

---

## Usage

To run the project:
1. Open the **Jupyter notebook** (`carprice_prediction.ipynb`) to explore the data analysis and model training process.
2. Run the cells in the notebook to:
   - Load the dataset
   - Perform data preprocessing
   - Build the linear regression model
   - Evaluate the model’s performance on the test set
3. If needed, modify the notebook to experiment with different algorithms or improve the prediction accuracy.

---

## Results
- The **Mean Absolute Error (MAE)** for the model was **1.7**.
- The **R-squared score** was **1.8**, indicating how well the model explains the variance in the car prices.
- Detailed analysis, visualizations, and evaluation metrics are provided in the notebook.

---

## Contributors
- **Kolli Venkata Rushita** - [GitHub Profile](https://github.com/Kolli-VenkataRushita)

Feel free to explore the project and contribute!

---

## Deployment Process

To make the **Car Price Prediction** model accessible online, we can deploy it as a **Streamlit Web Application**.

### 1. **Create a Streamlit Web App**:
Create a Python script named `carprice_prediction_app.py` for the Streamlit app.

```python
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model
with open('car_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🚗 Car Price Prediction App")

st.markdown("Fill in the car details to get the estimated price.")

# Numeric features
cylinders = st.number_input("Cylinders", min_value=1, max_value=16, value=4)
airbags = st.number_input("Airbags", min_value=0, max_value=10, value=2)
levy_log = st.number_input("Levy (log transformed)", value=0.0)
prod_year_log = st.number_input("Production Year (log transformed)", value=7.6)
engine_volume_log = st.number_input("Engine Volume (log transformed)", value=1.3)
mileage_log = st.number_input("Mileage (log transformed)", value=9.4)

# Categorical - one-hot encoded (single choice each)
category = st.selectbox("Category", [
    "Cabriolet", "Coupe", "Goods wagon", "Hatchback", "Jeep", "Limousine",
    "Microbus", "Minivan", "Pickup", "Sedan", "Universal"
])

leather_interior = st.radio("Leather Interior", ["Yes", "No"])

fuel_type = st.selectbox("Fuel Type", [
    "CNG", "Diesel", "Hybrid", "Hydrogen", "LPG", "Petrol", "Plug-in Hybrid"
])

gear_box_type = st.selectbox("Gear Box Type", [
    "Automatic", "Manual", "Tiptronic", "Variator"
])

drive_wheels = st.selectbox("Drive Wheels", [
    "All-wheel drive", "Front", "Rear"
])

wheel = st.radio("Wheel", ["Left wheel", "Right-hand drive"])

color_other = st.checkbox("Color is not standard (Other color)?")

# Initialize all feature columns
features = {
    'Cylinders': cylinders,
    'Airbags': airbags,
    'Levy_log': levy_log,
    'Prod. year_log': prod_year_log,
    'Engine volume_log': engine_volume_log,
    'Mileage_log': mileage_log,
    'Color_Other': int(color_other)
}

# One-hot for category
category_list = [
    "Cabriolet", "Coupe", "Goods wagon", "Hatchback", "Jeep", "Limousine",
    "Microbus", "Minivan", "Pickup", "Sedan", "Universal"
]
for cat in category_list:
    features[f"Category_{cat}"] = 1 if category == cat else 0

# Leather Interior
features["Leather interior_Yes"] = 1 if leather_interior == "Yes" else 0
features["Leather interior_No"] = 1 if leather_interior == "No" else 0

# Fuel type
fuel_list = ["CNG", "Diesel", "Hybrid", "Hydrogen", "LPG", "Petrol", "Plug-in Hybrid"]
for ft in fuel_list:
    features[f"Fuel type_{ft}"] = 1 if fuel_type == ft else 0

# Gear box type
gear_list = ["Automatic", "Manual", "Tiptronic", "Variator"]
for gb in gear_list:
    features[f"Gear box type_{gb}"] = 1 if gear_box_type == gb else 0

# Drive wheels
drive_list = ["All-wheel drive", "Front", "Rear"]
for dw in drive_list:
    features[f"Drive wheels_{dw}"] = 1 if drive_wheels == dw else 0

# Wheel position
features["Wheel_Left wheel"] = 1 if wheel == "Left wheel" else 0
features["Wheel_Right-hand drive"] = 1 if wheel == "Right-hand drive" else 0

# Convert to DataFrame with correct order
input_df = pd.DataFrame([features])

if st.button("Predict Car Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Price: ₹ {round(prediction, 2):,}")
