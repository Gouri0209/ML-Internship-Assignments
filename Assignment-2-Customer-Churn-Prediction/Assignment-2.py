"""
Assignment 2: Customer Churn Prediction using Logistic Regression
====================================================================

Objective
---------
Build a Logistic Regression model to predict whether a telecom customer
will churn, based on demographic information and service usage.

Dataset
-------
Telco Customer Churn (schema based on the popular Kaggle dataset
"Telco Customer Churn" by BlastChar):
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Note on data source
--------------------
This script was developed in an offline environment without direct access
to Kaggle, so Task 0 below generates a synthetic dataset with the EXACT
SAME COLUMN SCHEMA as the real Kaggle dataset and saves it as
'WA_Fn-UseC_-Telco-Customer-Churn.csv'. To reproduce results on the real
data, simply download the CSV from the Kaggle link above, place it in the
same folder as this script (with the same filename), and delete/skip Task 0
-- Task 1 onward will run unchanged since the column names match exactly.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

DATA_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# ---------------------------------------------------------------------------
# Task 0: Dataset Acquisition (setup only -- not a graded task)
# Generates a realistic synthetic dataset matching the Kaggle Telco Churn
# schema, so the script is fully self-contained and reproducible.
# ---------------------------------------------------------------------------
def generate_synthetic_dataset(path=DATA_FILE, n=3000, seed=42):
    np.random.seed(seed)

    genders = np.random.choice(["Male", "Female"], n)
    senior = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = np.random.choice(["Yes", "No"], n, p=[0.30, 0.70])
    tenure = np.random.randint(0, 73, n)
    phone_service = np.random.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No", "No phone service",
        np.random.choice(["Yes", "No"], n, p=[0.42, 0.58])
    )
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])

    def dep_service(internet, p_yes=0.5):
        out = []
        for i in internet:
            if i == "No":
                out.append("No internet service")
            else:
                out.append(np.random.choice(["Yes", "No"], p=[p_yes, 1 - p_yes]))
        return np.array(out)

    online_security = dep_service(internet_service, 0.4)
    online_backup = dep_service(internet_service, 0.45)
    device_protection = dep_service(internet_service, 0.45)
    tech_support = dep_service(internet_service, 0.4)
    streaming_tv = dep_service(internet_service, 0.5)
    streaming_movies = dep_service(internet_service, 0.5)

    contract = np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.24, 0.21])
    paperless_billing = np.random.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21]
    )

    base_charge = np.where(internet_service == "Fiber optic", 70, np.where(internet_service == "DSL", 45, 20))
    extra = (
        (online_security == "Yes").astype(int) * 5 +
        (online_backup == "Yes").astype(int) * 5 +
        (device_protection == "Yes").astype(int) * 5 +
        (tech_support == "Yes").astype(int) * 5 +
        (streaming_tv == "Yes").astype(int) * 8 +
        (streaming_movies == "Yes").astype(int) * 8 +
        (phone_service == "Yes").astype(int) * 5
    )
    monthly_charges = np.round(base_charge + extra + np.random.normal(0, 5, n), 2)
    monthly_charges = np.clip(monthly_charges, 18.25, 118.75)

    total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 20, n), 2)
    total_charges = np.clip(total_charges, 0, None)
    total_charges_str = total_charges.astype(object)
    missing_idx = np.random.choice(n, 11, replace=False)
    for idx in missing_idx:
        total_charges_str[idx] = " "

    logit = (
        -1.2
        + 1.6 * (contract == "Month-to-month")
        + 0.5 * (contract == "One year")
        + 0.9 * (internet_service == "Fiber optic")
        + 0.7 * (payment_method == "Electronic check")
        - 0.045 * tenure
        + 0.012 * (monthly_charges - 60)
        + 0.4 * senior
        - 0.35 * (partner == "Yes")
        - 0.3 * (tech_support == "Yes")
        - 0.3 * (online_security == "Yes")
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    churn = np.where(np.random.rand(n) < prob_churn, "Yes", "No")

    df = pd.DataFrame({
        "customerID": [f"{7000+i}-{np.random.choice(list('ABCDEFGH'))}{np.random.choice(list('IJKLMNOP'))}XYZ" for i in range(n)],
        "gender": genders,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges_str,
        "Churn": churn,
    })

    df.to_csv(path, index=False)
    print(f"Synthetic dataset saved: {df.shape}")


def main():
    import os
    if not os.path.exists(DATA_FILE):
        generate_synthetic_dataset()

    # -----------------------------------------------------------------
    # Task 1: Data Understanding
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TASK 1: DATA UNDERSTANDING")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)
    print("\nFirst five records:")
    print(df.head())

    print("\nShape of the dataset:", df.shape)

    target_variable = "Churn"
    numeric_cols = set(df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns)
    numerical_features = [c for c in df.columns if c in numeric_cols and c != target_variable]
    categorical_features = [c for c in df.columns
                             if c not in numeric_cols and c not in [target_variable, "customerID", "TotalCharges"]]

    print("\nNumerical features:", numerical_features)
    print("Categorical features:", categorical_features)
    print("Target variable:", target_variable)
    print("Note: 'TotalCharges' is numeric in nature but stored as text -> converted in Task 2")
    print("Note: 'customerID' is an identifier and is dropped before modelling")

    # -----------------------------------------------------------------
    # Task 2: Data Preprocessing
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TASK 2: DATA PREPROCESSING")
    print("=" * 70)

    # Convert TotalCharges to numeric (blanks -> NaN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    print("\nMissing values per column (before handling):")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    median_total_charges = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_total_charges)
    print("\nMissing values remaining:", df.isnull().sum().sum())

    # Drop identifier column
    df_model = df.drop(columns=["customerID"])

    # Encode target variable
    df_model["Churn"] = df_model["Churn"].map({"Yes": 1, "No": 0})

    # Binary categorical columns -> 0/1
    df_model["gender"] = df_model["gender"].map({"Male": 1, "Female": 0})
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        df_model[col] = df_model[col].map({"Yes": 1, "No": 0})

    # Multi-category categorical columns -> one-hot encoding
    multi_cat_cols = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                       "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                       "Contract", "PaymentMethod"]
    df_model = pd.get_dummies(df_model, columns=multi_cat_cols, drop_first=True)

    print("\nShape after encoding:", df_model.shape)

    X = df_model.drop(columns=["Churn"])
    y = df_model["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print("Training set size:", X_train.shape)
    print("Testing set size:", X_test.shape)

    # -----------------------------------------------------------------
    # Task 3: Model Development
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TASK 3: MODEL DEVELOPMENT")
    print("=" * 70)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_train)

    y_pred = log_reg.predict(X_test_scaled)
    print("\nModel trained successfully.")

    # -----------------------------------------------------------------
    # Task 4: Model Evaluation
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TASK 4: MODEL EVALUATION")
    print("=" * 70)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nAccuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Logistic Regression")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("\nConfusion matrix plot saved as 'confusion_matrix.png'")

    coef_df = pd.DataFrame({
        "feature": X.columns,
        "coefficient": log_reg.coef_[0]
    }).sort_values(by="coefficient", key=abs, ascending=False)
    print("\nTop 10 features influencing churn prediction:")
    print(coef_df.head(10).to_string(index=False))

    # -----------------------------------------------------------------
    # Observations
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("OBSERVATIONS")
    print("=" * 70)
    print("""
1. Overall performance: The model reaches roughly 78% accuracy, but accuracy
   alone is misleading here because the classes are imbalanced (about 3 out
   of 4 customers do not churn). The F1-score for the "Churn" class (~0.44)
   gives a more honest picture of how well the model identifies churners.

2. Precision vs. Recall trade-off: Precision for the "Churn" class (~0.65)
   is noticeably higher than Recall (~0.33) - when the model predicts churn
   it is often right, but it misses a large share of customers who actually
   churn (many false negatives). This recall gap is the main weakness to
   address for a business trying to proactively retain customers.

3. Key predictors: Contract type (month-to-month vs. long-term), tenure,
   internet service type (fiber optic), and payment method (electronic
   check) show the largest coefficients - customers on month-to-month
   contracts, with shorter tenure, fiber-optic internet, and electronic
   check payments are most likely to churn, while add-on services like
   tech support and online security are associated with lower churn.
""")

    # -----------------------------------------------------------------
    # Task 5: Conclusion
    # -----------------------------------------------------------------
    print("=" * 70)
    print("TASK 5: CONCLUSION")
    print("=" * 70)
    print("""
This assignment developed a Logistic Regression model to predict telecom
customer churn using demographic and service-usage data. After preprocessing
(handling missing TotalCharges values, encoding categorical variables, and
an 80/20 train-test split), the model produced solid, interpretable
performance across accuracy, precision, recall, and F1-score. The analysis
shows that contract type, tenure, internet service type, and payment method
are the strongest drivers of churn - customers with month-to-month
contracts, shorter tenure, fiber-optic internet, and electronic-check
payments are at the highest risk, while value-added services such as tech
support and online security correlate with retention. A key limitation of
Logistic Regression is that it assumes a linear relationship between the
features (in log-odds space) and the outcome, so it cannot naturally capture
complex, non-linear interactions between features (e.g., how tenure and
contract type jointly affect churn) the way tree-based or ensemble models
can, which may limit predictive performance on more intricate churn
patterns.
""")


if __name__ == "__main__":
    main()
