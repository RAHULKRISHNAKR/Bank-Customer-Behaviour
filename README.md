# Bank-Customer-Behaviour
## KTU S5 DATA ANALYTICS LAB MICRO PROJECT


### Step 1: Load the Dataset
```python
data = pd.read_csv("bank_sample_2000.csv")
```
This line loads a CSV file named `bank_sample_2000.csv` into a pandas DataFrame called `data`. The dataset might include various features like age, balance, and whether the customer subscribed to a term deposit (`y_yes`).

---

### Step 2: Data Pre-processing

#### Check for Missing Values
```python
print("Missing Values:\n", data.isnull().sum())
```
Here, we are checking if there are any missing values in the dataset. The `isnull()` function returns `True` for missing values and `False` for others. The `sum()` function counts how many `True` values (missing data) each column has. This gives an overview of whether or not you need to handle missing data.

#### Min-Max Normalization
```python
numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
scaler = MinMaxScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])
```
Min-Max normalization ensures that all numerical values are scaled between 0 and 1. This helps when applying machine learning algorithms that are sensitive to the scale of the input data. We apply this normalization to the selected numerical columns. 
- `MinMaxScaler()` transforms the values into a 0-1 range.
- `fit_transform()` computes the minimum and maximum values for scaling and transforms the data accordingly.

#### One-Hot Encoding (Dummy Variables)
```python
data_encoded = pd.get_dummies(data, drop_first=True)
```
One-hot encoding is used to convert categorical variables (like "job" or "marital status") into a binary format (0s and 1s) since machine learning models can't directly interpret categories as labels. Each unique value of a categorical column becomes a new binary column.

`drop_first=True` ensures we don't get redundant columns that are perfect inverses of one another (because if, for example, you know that someone isn't "married" or "single," you can infer they're "divorced"). It helps avoid multicollinearity.

---

### Step 3: Feature Selection using Chi-Square Test
```python
X = data_encoded.drop("y_yes", axis=1)  # Target column is 'y_yes'
y = data_encoded["y_yes"]
```
Here, we define `X` as the input data (all the columns except the target variable) and `y` as the target label (which in this case is the `y_yes` column that represents whether the customer subscribed to a term deposit).

Next, we apply a Chi-Square feature selection:
```python
selector = SelectKBest(score_func=chi2, k=10)
X_selected = selector.fit_transform(X, y)
print("Selected Features (Chi-Square):\n", X.columns[selector.get_support()])
```
The Chi-Square test helps in selecting the most important features for predicting the target (`y_yes`). The `SelectKBest` class is used here to select the top 10 features with the highest Chi-Square scores.

- `score_func=chi2` tells `SelectKBest` to use the Chi-Square test.
- `k=10` means that we are selecting the top 10 features.
- `fit_transform()` fits the selector to the data and then transforms `X` to include only the top 10 features.
- The final line prints out the names of the selected features.

---

### Step 4: Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
```
Here, the dataset is split into training and test sets. We use the `train_test_split()` function to randomly divide the data:
- **70%** of the data is used for training (`X_train`, `y_train`).
- **30%** of the data is set aside for testing (`X_test`, `y_test`).
- `random_state=42` ensures that the split is reproducible (the random selection of rows will be the same every time).

This step is crucial because it helps us evaluate how well the model generalizes to unseen data (the test set) after training on the training set.

---

### Code Summary:
- **Loading the data**: Importing the CSV file into a DataFrame.
- **Data preprocessing**: 
  - Checking for missing values,
  - Normalizing the numerical features,
  - Encoding categorical variables into numerical format via one-hot encoding.
- **Feature selection**: Using the Chi-Square test to select the most important features for model training.
- **Train-test split**: Dividing the dataset into training and testing sets for evaluation.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2

data = pd.read_csv("bank_sample_2000.csv")

print("Missing Values:\n", data.isnull().sum())

numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
scaler = MinMaxScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

data_encoded = pd.get_dummies(data, drop_first=True)

X = data_encoded.drop("y_yes", axis=1)  # Target column is 'y_yes'
y = data_encoded["y_yes"]

selector = SelectKBest(score_func=chi2, k=10)
X_selected = selector.fit_transform(X, y)
print("Selected Features (Chi-Square):\n", X.columns[selector.get_support()])

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
```
This partial code prepares the data for training models but does not yet include model training or evaluation.

```python
Missing Values:
 age          0
job          0
marital      0
education    0
default      0
balance      0
housing      0
loan         0
contact      0
day          0
month        0
duration     0
campaign     0
pdays        0
previous     0
poutcome     0
y            0
dtype: int64
Selected Features (Chi-Square):
 Index(['duration', 'job_retired', 'loan_yes', 'contact_unknown', 'month_dec',
       'month_mar', 'month_oct', 'month_sep', 'poutcome_success',
       'poutcome_unknown'],
      dtype='object')
```
