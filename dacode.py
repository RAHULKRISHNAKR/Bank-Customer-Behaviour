import pandas as pd
import numpy as np
import streamlit as st
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
import seaborn as sns

# Custom CSS to set background image
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://i.postimg.cc/4d0sht5Y/black-elegant-background-with-copy-space.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);  /* Make header transparent */
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# App Title with Custom Styling
st.markdown('<h1 style="font-family: serif; color: white; text-align: center;">✨ Bank Marketing Model Interface </h1>', unsafe_allow_html=True)

# File Upload
uploaded_file = st.file_uploader("📁 Upload your CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Display DataFrame
    st.write("🗂 **Dataset Preview**:")
    st.dataframe(data.head())

    # Data Pre-processing
    st.subheader("🛠 Data Pre-processing")

    # Check for missing values
    missing_values = data.isnull().sum()
    st.write("❌ **Missing Values**:", missing_values)

    # Apply Min-Max Normalization
    numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    scaler = MinMaxScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # One-Hot Encoding
    data_encoded = pd.get_dummies(data, drop_first=True)

    # Feature Selection
    X = data_encoded.drop("y_yes", axis=1)
    y = data_encoded["y_yes"]
    selector = SelectKBest(score_func=chi2, k=10)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    st.write("📊 **Selected Features:**", selected_features)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

    # List of models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "k-NN": KNeighborsClassifier(n_neighbors=5)
    }

    # Dropdown to select individual model
    selected_model_name = st.selectbox("🔍 Choose a model to display results individually", list(models.keys()))

    # Show results for the selected model
    if selected_model_name:
        model = models[selected_model_name]
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        st.write(f"### {selected_model_name} Results")
        st.write(f"✅ **Accuracy:** {accuracy * 100:.2f}%")
        st.write(f"📏 **MCC:** {mcc:.2f}")

        # Classification Report
        st.text("📝 **Classification Report:**")
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False, ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix for {selected_model_name}')
        st.pyplot(fig)

        # ROC Curve
        y_probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"{selected_model_name} (AUC = {roc_auc_score(y_test, y_probs):.2f})")
        ax.plot([0, 1], [0, 1], 'k--')  # Random Guess Line
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='best')
        st.pyplot(fig)

    # Train and Evaluate All Models
    if st.button("🚀 Train and Evaluate All Models"):
        # Create layout for displaying models' results
        cols = st.columns(2)
        results = {}

        for i, (name, model) in enumerate(models.items()):
            # Train the model
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            results[name] = accuracy

            with cols[i % 2]:  # Display in columns
                st.markdown(f"### {name}")
                st.write(f"✅ **Accuracy:** {accuracy * 100:.2f}%")
                st.write(f"📏 **MCC:** {mcc:.2f}")

                # Classification Report
                st.text("📝 **Classification Report:**")
                st.text(classification_report(y_test, y_pred))

                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False, ax=ax)
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                ax.set_title(f'Confusion Matrix for {name}')
                st.pyplot(fig)

                # ROC Curve
                y_probs = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_probs)

                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_probs):.2f})")
                ax.plot([0, 1], [0, 1], 'k--')  # Random Guess Line
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve')
                ax.legend(loc='best')
                st.pyplot(fig)

        # Determine the best model based on accuracy
        best_model_name = max(results, key=results.get)
        best_accuracy = results[best_model_name]

        # Display best model message
        st.markdown(f"## 🏆 The best model is **{best_model_name}** with an accuracy of **{best_accuracy * 100:.2f}%**!")
        st.write(f"### Reason: {best_model_name} achieved the highest accuracy compared to other models, making it the most effective for this dataset.")
