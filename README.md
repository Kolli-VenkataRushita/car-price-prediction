# Car Price Prediction

## Table of Contents
- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributors](#contributors)

## Project Overview
The **Car Price Prediction** project aims to predict the selling price of used cars based on various features such as age, brand, mileage, fuel type, transmission type, etc. The goal is to build a predictive model using **Linear Regression** to estimate the car price accurately, helping users or businesses make informed decisions when buying or selling cars.

This project includes data analysis, feature selection, and model building using a Jupyter notebook, leveraging Python libraries for **data preprocessing** and **machine learning**.

## Tech Stack
- **Programming Language**: Python
- **Libraries**:
  - `pandas` - For data manipulation and cleaning
  - `numpy` - For numerical operations
  - `matplotlib` and `seaborn` - For data visualization
  - `scikit-learn` - For model building and evaluation

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

You can find the dataset at `car_price_data.csv` in the project folder.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/car-price-prediction.git
   cd car-price-prediction
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the dataset (`car_price_data.csv`) in the `data/` directory.


## Model Training and Evaluation
1. **Data Preprocessing**:
   - Cleaned the dataset by removing irrelevant or missing values.
   - Categorical features like `Fuel Type`, `Seller Type`, and `Transmission` were converted into numerical form using **One-Hot Encoding**.
   - Features like `Year` were transformed into `Car Age` to give a better indication of the car's value depreciation.

2. **Model Selection**:
   - A **Linear Regression** model was chosen due to its effectiveness in predicting continuous variables such as price.
   - The dataset was split into training and test sets using **train_test_split** to validate the model.

3. **Model Evaluation**:
   - The model was evaluated using **Mean Absolute Error (MAE)** and **R-squared** scores to measure how well the predicted prices matched the actual selling prices.

## Usage
To run the project:
1. Open the **Jupyter notebook** (`carprice_prediction.ipynb`) to explore the data analysis and model training process.
   
2. Run the cells in the notebook to:
   - Load the dataset
   - Perform data preprocessing
   - Build the linear regression model
   - Evaluate the model’s performance on the test set
   
3. If needed, modify the notebook to experiment with different algorithms or improve the prediction accuracy.

## Results
- The **Mean Absolute Error (MAE)** for the model was **1.7**.
- The **R-squared score** was **1.8**, indicating how well the model explains the variance in the car prices.

Detailed analysis, visualizations, and evaluation metrics are provided in the notebook.

## Contributors
- **Kolli Venkata Rushita** - [GitHub](https://github.com/Kolli-VenkataRushita)

Feel free to explore the project and contribute!
```
