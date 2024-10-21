# KTU S5 DATA ANALYTICS MICROPROJECT
---
# 📊 [Bank Customer Behavior Prediction - Streamlit App](https://bank-customer-behaviour.streamlit.app/)

Welcome to the **Bank Customer Behavior Prediction** project! This tool predicts whether a customer is likely to subscribe to a term deposit based on various features from the **Bank Marketing Dataset**. The prediction model leverages multiple machine learning algorithms, and the best-performing model is selected based on accuracy and MCC (Matthews Correlation Coefficient).

## 🚀 Project Overview

This project aims to create an interactive web interface where users can upload their own datasets, perform data preprocessing, and evaluate various classification models. The tool also allows users to predict the likelihood of a new customer subscribing to a term deposit based on input features. The interface is developed using **Streamlit** and supports model comparisons based on metrics like **Accuracy** and **MCC**.

You can access the live version of the project [here](https://bank-customer-behaviour.streamlit.app/).

## 🎯 Features

- **Upload CSV Files**: Users can upload datasets for custom predictions.
- **Data Preprocessing**: Automatic handling of missing values, Min-Max scaling for numerical features, and one-hot encoding for categorical variables.
- **Feature Selection**: Utilizes `SelectKBest` to select the top 10 features based on chi-squared scores.
- **Model Comparisons**: Includes models like **Random Forest**, **Gradient Boosting**, **SVM**, **k-NN**, **Naive Bayes**, **Decision Tree**, **Logistic Regression**, and **MLP**. Results for each model are evaluated based on:
  - Accuracy
  - MCC (Matthews Correlation Coefficient)
  - Confusion Matrix
  - ROC Curve (AUC)
- **Best Model Selection**: Automatically selects the best model based on accuracy and MCC.
- **Prediction for New Customers**: Allows input of new customer data to predict the likelihood of subscription.
- **Visualization**: Displays classification reports, confusion matrices, and ROC curves for in-depth performance analysis.

## 📂 Repository Structure

- **`app.py`**: The main script containing the code for the Streamlit app.
- **`requirements.txt`**: Lists all the dependencies required to run the application.
- **`README.md`**: Project documentation (this file).
- **`data/`**: Contains sample datasets for testing purposes.
  
## 🛠️ Technologies Used

- **Python**: The programming language used to develop the project.
- **Streamlit**: The web framework used to build the interactive interface.
- **Scikit-learn**: Used for implementing machine learning models and preprocessing.
- **Pandas & NumPy**: For data manipulation and analysis.
- **Matplotlib & Seaborn**: For data visualization.

## 🔧 Installation and Setup

To run the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/RAHULKRISHNAKR/Bank-Customer-Behaviour.git
   cd Bank-Customer-Behaviour
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.7 or higher installed. Run the following command to install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**:
   After installation, you can start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. **Access the App**:
   Once the app is running, you can access it locally in your browser at `http://localhost:8501/`.

## 📊 Machine Learning Models Used

- **Random Forest**
- **Gradient Boosting**
- **Support Vector Machine (SVM)**
- **k-Nearest Neighbors (k-NN)**
- **Naive Bayes**
- **Decision Tree**
- **Logistic Regression**
- **Multi-Layer Perceptron (MLP)**

Each model's performance is evaluated based on **accuracy** and **MCC** to ensure a balanced evaluation, especially for imbalanced datasets. The app also features **ROC curves** and **confusion matrices** for visual performance analysis.

## 🏆 Best Model

The best-performing model is dynamically selected based on evaluation metrics. Once trained, it is used for predicting new customer outcomes and is highlighted on the interface.

## 📊 Dataset

This project uses the **Bank Marketing Dataset**, available on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). The dataset consists of customer information such as age, job, marital status, balance, etc., along with the target variable indicating whether the customer subscribed to a term deposit.

## 👥 Team

- [**Rahul Krishna K R**](https://github.com/RAHULKRISHNAKR) 
- [**Nibras**](https://github.com/Nibras-10)
- [**Vignesh**](https://github.com/Vignesh2004gh)

## 📄 License

This project is licensed under the MIT License. Feel free to use and modify the code as per your requirements.
