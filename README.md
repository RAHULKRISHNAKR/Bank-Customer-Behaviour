# Bank-Customer-Behaviour
## KTU S5 DATA ANALYTICS LAB MICRO PROJECT

Here’s a professional `README.md` file you can use for your GitHub repository:

---

# Machine Learning Model Comparison for Bank Marketing Dataset

## Project Overview

This project aims to evaluate and compare the performance of several machine learning models in predicting whether customers will subscribe to a term deposit based on the **Bank Marketing Dataset**. The dataset contains customer-related information and campaign data used for marketing the bank's term deposits.

### Additional Information for the README

The provided paper, *"Modelling bank customer behaviour using feature engineering and classification techniques"* by Abedin et al., focuses on advanced methods for analyzing bank customer behavior using feature engineering and machine learning techniques. Below are key takeaways that can be added to the README file of your project, which align with your work:

---

### Academic Background and Relevance

This project is inspired by research that highlights the importance of **feature engineering** and **machine learning** models in predicting bank customer behavior. Recent studies (such as Abedin et al., 2023) demonstrate that feature transformation and selection are crucial for improving model performance, especially in the financial sector.

**Key Insights from Related Research:**
- **Feature Engineering**: The paper emphasizes transforming raw data into meaningful features to boost model accuracy. Common techniques include Min-Max normalization and Chi-Square feature selection, both of which are applied in this project.
- **Machine Learning Models**: Several classification algorithms such as Random Forest, Gradient Boosting, and Support Vector Machine (SVM) have been shown to effectively predict customer behaviors like churn or product subscription. This project evaluates similar models using accuracy, MCC, and ROC-AUC scores, which are crucial for model comparison.

### Key Similarities with the Paper:
- **Data Preprocessing**: Similar to our approach, the research underscores the use of scaling (e.g., Min-Max) and categorical encoding techniques (e.g., One-Hot Encoding).
- **Feature Selection**: The Chi-Square feature selection method used in our project is discussed as a significant pre-processing step in improving prediction accuracy, particularly for bank customer classification.
- **Model Comparison**: The paper validates the effectiveness of ensemble methods like Random Forest and boosting algorithms. Our project evaluates Random Forest, Gradient Boosting, SVM, and k-NN, similar to the approaches discussed.

**Source**: Abedin et al., "Modelling bank customer behaviour using feature engineering and classification techniques", *Research in International Business and Finance*, 2023【12†source】.

---
### Key Steps:
1. **Data Preprocessing**:
   - Min-Max scaling applied to numerical features.
   - One-Hot Encoding used for categorical variables.
2. **Feature Selection**:
   - Chi-Square test applied to select the top 10 features for modeling.
3. **Model Training and Evaluation**:
   - Multiple models (Random Forest, Gradient Boosting, SVM, k-NN) are trained and evaluated using:
     - Accuracy
     - Matthews Correlation Coefficient (MCC)
     - Classification reports
     - Confusion matrices
     - ROC curves

## Dataset

The dataset used for this project is a sample of the **Bank Marketing Dataset** with 2000 rows (`bank_sample_2000.csv`), including attributes like age, balance, duration, campaign, etc. The target variable (`y_yes`) indicates whether the customer subscribed to a term deposit (`yes` or `no`).

### Features:
- **Numerical Features**:
  - `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`
- **Categorical Features**:
  - `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`, `y`

## Models Evaluated

The following models were trained and evaluated:

- **Random Forest**
- **Gradient Boosting**
- **Support Vector Machine (SVM)**
- **k-Nearest Neighbors (k-NN)**

## Performance Metrics

We evaluated the models using several key performance metrics:

- **Accuracy**: The overall effectiveness of the model at correctly predicting the target.
- **Matthews Correlation Coefficient (MCC)**: A more balanced measure of classification quality, especially for imbalanced datasets.
- **Classification Report**: Provides precision, recall, and F1-score.
- **Confusion Matrix**: Visualizes true positives, true negatives, false positives, and false negatives.
- **ROC Curve**: Shows the trade-off between the true positive rate and false positive rate.

## Results

### Selected Features (Chi-Square Test)

The top 10 features selected by the Chi-Square test are:

- `duration`, `job_retired`, `loan_yes`, `contact_unknown`, `month_dec`, `month_mar`, `month_oct`, `month_sep`, `poutcome_success`, `poutcome_unknown`

### Model Performance

| Model               | Accuracy (%) | MCC  |
|-------------------- |--------------|------|
| **Random Forest**   | 87.67        | 0.27 |
| **Gradient Boosting** | 90.67        | 0.37 |
| **SVM**             | 90.67        | 0.24 |
| **k-NN**            | 90.17        | 0.39 |

- **Random Forest** had an accuracy of 87.67% but struggled with predicting the minority class (`y_yes`).
- **Gradient Boosting** performed better, with an accuracy of 90.67% and improved Matthews correlation.
- **SVM** achieved high accuracy but had lower performance in predicting positive samples (`y_yes`).
- **k-NN** had a good balance of accuracy (90.17%) and MCC (0.39).

## Confusion Matrices and ROC Curves

The confusion matrices and ROC curves for each model can be found in the output plots, providing insights into how well each model balances sensitivity and specificity.

## Installation and Usage

### Prerequisites

- Python 3.x
- Required libraries: `pandas`, `numpy`, `sklearn`, `matplotlib`, `seaborn`

You can install the necessary libraries using:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Running the Project

1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/bank-marketing-ml-comparison.git
    cd bank-marketing-ml-comparison
    ```

2. Make sure the dataset (`bank_sample_2000.csv`) is in the root directory.

3. Run the Python script:
    ```bash
    python bank_marketing_ml.py
    ```

The script will load the dataset, preprocess the data, perform feature selection, train the models, and display the results including accuracy, MCC, classification reports, confusion matrices, and ROC curves.

## Output Examples

### Confusion Matrix (Random Forest)

![Confusion Matrix for Random Forest](images/random_forest_confusion_matrix.png)

### ROC Curve Comparison

![ROC Curve Comparison](images/roc_curve_comparison.png)

## Conclusion

- Gradient Boosting and k-NN models showed the best balance between accuracy and MCC, with Gradient Boosting slightly outperforming k-NN in most cases.
- Random Forest and SVM performed well, but their prediction ability for the minority class was relatively weaker.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions or suggestions, feel free to open an issue or contact me via email at [rahulkridhna@gmail.com](mailto:rahulkridhna.gmail.com).

---
