# Bank-Customer-Behaviour

### KTU S5 DATA ANALYTICS LAB MICRO PROJECT  
**Machine Learning Model Comparison for Bank Marketing Dataset**

---

## Project Overview  
This project evaluates and compares the performance of several machine learning models in predicting whether customers will subscribe to a term deposit using the **Bank Marketing Dataset**. The dataset includes customer-related information and marketing campaign data, and the models are evaluated for their predictive accuracy, including advanced metrics such as Matthews Correlation Coefficient (MCC) and ROC-AUC scores.

We built an interactive frontend using **Streamlit**, allowing users to upload their dataset and visualize the model results through a user-friendly interface.

---

## Academic Background and Relevance  
The project is inspired by research such as *"Modelling Bank Customer Behaviour Using Feature Engineering and Classification Techniques"* by Abedin et al. (2023), which emphasizes the importance of **feature engineering** and **machine learning** for predicting bank customer behaviors.

### Key Insights from Related Research:
- **Feature Engineering**: Transforming raw data into meaningful features is crucial for improving model accuracy. Techniques like **Min-Max normalization** and **Chi-Square feature selection** were both implemented in this project.
- **Machine Learning Models**: Classification algorithms like **Random Forest**, **Gradient Boosting**, and **Support Vector Machine (SVM)** have been shown to be effective, and we applied these models to evaluate customer behavior predictions.

---

## Features
- **Streamlit Web Interface**: Upload your own dataset and interactively train models.
- **Real-Time Visualizations**: Confusion matrices, ROC curves, and classification reports are dynamically generated.
- **Model Comparison**: Compare multiple machine learning models, including **Random Forest**, **Gradient Boosting**, **SVM**, and **k-NN**.

---

## Key Steps:
1. **Data Preprocessing**:
   - Min-Max scaling applied to numerical features.
   - One-Hot Encoding used for categorical variables.

2. **Feature Selection**:
   - **Chi-Square** test applied to select the top 10 features for modeling.

3. **Model Training and Evaluation**:
   - Multiple models trained and evaluated using:
     - Accuracy
     - Matthews Correlation Coefficient (MCC)
     - Classification reports
     - Confusion matrices
     - ROC curves

---

## Dataset  
The dataset used is a sample of the Bank Marketing Dataset with 2,000 rows (`bank_sample_2000.csv`). It includes attributes like age, balance, duration, campaign, and more. The target variable (`y_yes`) indicates whether the customer subscribed to a term deposit (yes or no).

### Features:
- **Numerical**: age, balance, day, duration, campaign, pdays, previous
- **Categorical**: job, marital, education, default, housing, loan, contact, month, poutcome, y

---

## Models Evaluated
- **Random Forest**
- **Gradient Boosting**
- **Support Vector Machine (SVM)**
- **k-Nearest Neighbors (k-NN)**

---

## Performance Metrics
- **Accuracy**: The overall effectiveness of the model at correctly predicting the target.
- **Matthews Correlation Coefficient (MCC)**: A balanced measure of classification quality, especially for imbalanced datasets.
- **Classification Report**: Provides precision, recall, and F1-score.
- **Confusion Matrix**: Visualizes true positives, true negatives, false positives, and false negatives.
- **ROC Curve**: Shows the trade-off between the true positive rate and false positive rate.

---

## Streamlit App  
We built an interactive **Streamlit** web app that allows users to:
- Upload their own CSV dataset.
- View the dataset and preprocess it.
- Select individual models to train and evaluate.
- View **confusion matrices**, **classification reports**, and **ROC curves** for each model.
- Automatically compare models and display the best-performing model.

### Demo  
Check out the [live demo](https://data-analytics-microproject-x5juxtrpdwaavzukqykess.streamlit.app/)).

---

## Installation and Usage

### Prerequisites:
- **Python 3.x**
- Required libraries: `pandas`, `numpy`, `sklearn`, `matplotlib`, `seaborn`, `streamlit`

You can install the necessary libraries using:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

### Running the Project

Clone the repository:
```bash
git clone https://github.com/RAHULKRISHNAKR/Bank-Customer-Behaviour.git
cd Bank-Customer-Behaviour
```

Make sure the dataset (`bank_sample_2000.csv`) is in the root directory.

Run the Streamlit app:
```bash
streamlit run app.py
```

---

## Output Examples
- **Confusion Matrix (Random Forest)**: ![Confusion Matrix](confusion_matrix_example.png)
- **ROC Curve Comparison**: ![ROC Curve](roc_curve_example.png)

---

## Conclusion
- **Gradient Boosting** and **k-NN** showed the best balance between accuracy and MCC, with Gradient Boosting slightly outperforming k-NN.
- **Random Forest** and **SVM** performed well but had relatively weaker prediction ability for the minority class.

---

## Contact  
If you have any questions or suggestions, feel free to open an issue or contact me via email at **rahulkridhna@gmail.com**.
