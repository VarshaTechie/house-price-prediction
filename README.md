# 🏠 California Housing Price Prediction

## 📌 Project Overview

This project demonstrates the process of building a Machine Learning model to predict California housing prices using the California Housing Dataset. The workflow includes:

- Data Loading
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Model Training using XGBoost Regressor
- Model Evaluation
- Prediction Visualization

---

## 📊 Dataset

The project uses the California Housing Dataset provided by Scikit-Learn.

### Dataset Information

- Total Samples: 20,640
- Features: 8 numerical attributes
- Target Variable: Median House Value

### Features

| Feature | Description |
|----------|------------|
| MedInc | Median income in block group |
| HouseAge | Median house age in block group |
| AveRooms | Average number of rooms per household |
| AveBedrms | Average number of bedrooms per household |
| Population | Block group population |
| AveOccup | Average number of household members |
| Latitude | Block group latitude |
| Longitude | Block group longitude |

### Target Variable

**MedHouseVal** — Median house value (in $100,000 units)

---

## 🔍 Data Analysis

The following steps were performed:

- Loaded dataset into a Pandas DataFrame
- Checked for missing values
- Generated descriptive statistics
- Created correlation heatmap
- Analyzed relationships between features and target variable

---

## 🤖 Model Training

### Algorithm Used

**XGBoost Regressor**

### Train-Test Split

- Training Data: 80%
- Testing Data: 20%
- Random State: 2

---

## 📈 Model Performance

### Training Results

| Metric | Value |
|----------|---------|
| R² Score | 0.94365 |
| MAE | 0.19336 |

### Test Results

| Metric | Value |
|----------|---------|
| R² Score | 0.83380 |
| MAE | 0.31086 |

### Interpretation

- The model explains approximately **83.38%** of the variance in unseen data.
- Average prediction error is approximately **$31,086**.

---

## 📉 Visualization

A scatter plot of Actual Prices vs Predicted Prices was generated to evaluate model performance visually.

Add your plot screenshot below:

![Prediction Results](screenshots/prediction_plot.png)

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost

---



---

## 🚀 Future Improvements

- Hyperparameter tuning
- Feature engineering
- Model deployment using Streamlit
- Comparison with other regression algorithms

---

## 👨‍💻 Author

Varsha
