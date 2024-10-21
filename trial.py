import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize session state
if 'best_model_name' not in st.session_state:
    st.session_state['best_model_name'] = None

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
    background: rgba(0, 0, 0, 0);
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# App Title with Custom Styling
st.markdown('<h1 style="font-family: serif; color: white; text-align: center;">✨ Bank Marketing Model Interface </h1>', unsafe_allow_html=True)

# File Upload
uploaded_file = st.file_uploader("📁 Upload your CSV file", type="csv")

# Global variables to store the model and preprocessing objects
global_scaler = None
global_selector = None
global_best_model = None
global_feature_names = None
numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

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
    global_scaler = MinMaxScaler()
    data[numerical_features] = global_scaler.fit_transform(data[numerical_features])

    # One-Hot Encoding
    data_encoded = pd.get_dummies(data, drop_first=True)
    global_feature_names = data_encoded.columns.tolist()

    # Feature Selection
    X = data_encoded.drop("y_yes", axis=1)
    y = data_encoded["y_yes"]
    global_selector = SelectKBest(score_func=chi2, k=10)
    X_selected = global_selector.fit_transform(X, y)
    selected_features = X.columns[global_selector.get_support()]
    st.write("📊 **Selected Features:**", selected_features)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

    # List of models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "k-NN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
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
        ax.plot([0, 1], [0, 1], 'k--')
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
        best_model_name = None
        best_accuracy = 0
        best_mcc = -1
        
        for i, (name, model) in enumerate(models.items()):
            # Train the model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            results[name] = (accuracy, mcc)
            
            # Update best model if current model is better
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_mcc = mcc
                best_model_name = name
                global_best_model = model
            elif accuracy == best_accuracy and mcc > best_mcc:
                best_mcc = mcc
                best_model_name = name
                global_best_model = model
            
            with cols[i % 2]:
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
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve')
                ax.legend(loc='best')
                st.pyplot(fig)

        # Store the best model name in session state
        st.session_state['best_model_name'] = best_model_name
        
        # Display best model message
        st.markdown(f"""
        ## 🏆 Best Model Selected
        - Model: **{best_model_name}**
        - Accuracy: **{best_accuracy * 100:.2f}%**
        - MCC Score: **{best_mcc:.3f}**
        """)

    # New Customer Prediction Section
    st.markdown("## 🎯 Predict for New Customer")
    
    if global_best_model is not None:
        st.info(f"🤖 Using **{st.session_state['best_model_name']}** for predictions")
    else:
        st.warning("Please train the models first by clicking 'Train and Evaluate All Models' button above.")

    # Create input fields for numerical features
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    balance = st.number_input("Balance", value=0)
    day = st.number_input("Day of Month", min_value=1, max_value=31, value=1)
    duration = st.number_input("Duration", min_value=0, value=0)
    campaign = st.number_input("Campaign", min_value=0, value=1)
    pdays = st.number_input("Previous Days", min_value=-1, value=-1)
    previous = st.number_input("Previous", min_value=0, value=0)

    # Create input fields for categorical features
    job = st.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", 
                              "retired", "self-employed", "services", "student", "technician", 
                              "unemployed", "unknown"])
    marital = st.selectbox("Marital Status", ["divorced", "married", "single"])
    education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
    default = st.selectbox("Has Credit in Default?", ["no", "yes"])
    housing = st.selectbox("Has Housing Loan?", ["no", "yes"])
    loan = st.selectbox("Has Personal Loan?", ["no", "yes"])
    contact = st.selectbox("Contact Communication Type", ["cellular", "telephone", "unknown"])
    month = st.selectbox("Last Contact Month", ["jan", "feb", "mar", "apr", "may", "jun", 
                                              "jul", "aug", "sep", "oct", "nov", "dec"])
    poutcome = st.selectbox("Previous Campaign Outcome", ["failure", "success", "nonexistent"])

    if st.button("🔮 Predict"):

        # Create a dictionary with the input values
        input_data = {
            'age': [age],
            'balance': [balance],
            'day': [day],
            'duration': [duration],
            'campaign': [campaign],
            'pdays': [pdays],
            'previous': [previous],
            'job': [job],
            'marital': [marital],
            'education': [education],
            'default': [default],
            'housing': [housing],
            'loan': [loan],
            'contact': [contact],
            'month': [month],
            'poutcome': [poutcome]
        }

        # Convert to DataFrame
        input_df = pd.DataFrame(input_data)

        # Apply the same preprocessing steps
        # Scale numerical features
        input_df[numerical_features] = global_scaler.transform(input_df[numerical_features])

        # One-hot encode
        input_encoded = pd.get_dummies(input_df, drop_first=True)

        # Ensure all features from training are present
        for col in global_feature_names:
            if col not in input_encoded.columns and col != "y_yes":
                input_encoded[col] = 0

        # Select features using the same selector
        input_selected = global_selector.transform(input_encoded[X.columns])

        # Make prediction
        prediction = global_best_model.predict(input_selected)
        prediction_prob = global_best_model.predict_proba(input_selected)

        # Display result with custom styling and model name
        if prediction[0] == 1:
            st.markdown(f'''
            <div style="padding:20px;background-color:rgba(0,255,0,0.1);border-radius:10px;">
                <h3 style="color:green;">✅ Customer is likely to subscribe!</h3>
                <p>Model: {st.session_state['best_model_name']}</p>
                <p>Confidence: {prediction_prob[0][1]:.2%}</p>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div style="padding:20px;background-color:rgba(255,0,0,0.1);border-radius:10px;">
                <h3 style="color:red;">❌ Customer is unlikely to subscribe</h3>
                <p>Model: {st.session_state['best_model_name']}</p>
                <p>Confidence: {prediction_prob[0][0]:.2%}</p>
            </div>
            ''', unsafe_allow_html=True)
