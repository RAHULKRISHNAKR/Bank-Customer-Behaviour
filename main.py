import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns  # Import Seaborn for heatmap

# Step 1: Load the Dataset
data = pd.read_csv("bank_sample_2000.csv")

# Step 2: Data Pre-processing
# Check for missing values
print("Missing Values:\n", data.isnull().sum())

# Apply Min-Max Normalization to all numerical features to ensure non-negative values
numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
scaler = MinMaxScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Encode categorical variables (One-Hot Encoding)
data_encoded = pd.get_dummies(data, drop_first=True)

# Step 3: Feature Selection using Chi-Square Test
X = data_encoded.drop("y_yes", axis=1)  # Target column is 'y_yes'
y = data_encoded["y_yes"]

# Apply SelectKBest to select top 10 features
selector = SelectKBest(score_func=chi2, k=10)
X_selected = selector.fit_transform(X, y)
print("Selected Features (Chi-Square):\n", X.columns[selector.get_support()])

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

# Step 5: Train and Evaluate Multiple Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "k-NN": KNeighborsClassifier(n_neighbors=5)
}

# Function to plot the confusion matrix as a heatmap
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

# Evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    print(f"{name} - Accuracy: {accuracy * 100:.2f}%, MCC: {mcc:.2f}")

    # Print Classification Report
    print(f"Classification Report for {name}:\n", classification_report(y_test, y_pred))

    # Plot Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, model_name=name)

    # ROC Curve
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_probs):.2f})")

# Step 6: Plot ROC Curves for all models
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='best')
plt.show()
