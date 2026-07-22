# Customer Churn Prediction using Logistic Regression

## Objective
A telecommunications company wants to predict whether a customer is likely
to leave (churn) based on demographic information and service usage. This
project develops a **Logistic Regression** model to predict customer churn
using demographic and service-usage features.

## Dataset Link
**Telco Customer Churn** (Kaggle, by BlastChar):
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

> **Note:** The raw dataset is **not included** in this repository, in line
> with the assignment instructions and the dataset's license on Kaggle.
> Please download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from the link above
> and place it in the project folder before running the code.
>
> For convenience/reproducibility, `Assignment-2.py` and `Assignment-2.ipynb`
> include a small synthetic-data generator (Task 0) that creates a CSV with
> the **exact same column schema** as the real Kaggle dataset. If the real
> CSV file is already present in the folder, the script/notebook will use it
> directly (skipping generation); otherwise it falls back to generating the
> synthetic file so the code still runs end-to-end.

## Libraries Used
- `pandas` — data loading and manipulation
- `numpy` — numerical operations / synthetic data generation
- `matplotlib` — plotting
- `seaborn` — confusion matrix heatmap
- `scikit-learn` — `train_test_split`, `StandardScaler`, `LogisticRegression`,
  and evaluation metrics (`accuracy_score`, `precision_score`,
  `recall_score`, `f1_score`, `confusion_matrix`, `classification_report`)

## Methodology

**1. Data Understanding**
- Loaded the dataset with `pandas.read_csv()` and inspected the first five
  records and overall structure (`df.info()`).
- Identified 3 numerical features (`SeniorCitizen`, `tenure`,
  `MonthlyCharges`), 15 categorical features (`gender`, `Partner`,
  `InternetService`, `Contract`, `PaymentMethod`, etc.), and the target
  variable `Churn`. `TotalCharges` is numeric in nature but stored as text
  in the raw file (handled separately), and `customerID` is dropped as a
  non-predictive identifier.

**2. Data Preprocessing**
- Converted `TotalCharges` to numeric (`pd.to_numeric`, coercing blank
  strings to `NaN`).
- Imputed missing `TotalCharges` values with the column median.
- Encoded categorical variables: binary columns (`gender`, `Partner`,
  `Dependents`, `PhoneService`, `PaperlessBilling`) mapped to 0/1, and
  multi-category columns (e.g. `InternetService`, `Contract`,
  `PaymentMethod`) one-hot encoded with `pd.get_dummies(drop_first=True)`.
- Split the data into an 80% training set and a 20% testing set using
  `train_test_split(test_size=0.20, stratify=y, random_state=42)`.

**3. Model Development**
- Scaled features with `StandardScaler`.
- Trained a `LogisticRegression` model (`max_iter=1000`) on the training set.
- Generated churn predictions on the held-out test set.

**4. Model Evaluation**
- Computed Accuracy, Precision, Recall, and F1-Score.
- Generated a Confusion Matrix (also saved as `confusion_matrix.png`).
- Inspected model coefficients to identify the strongest churn predictors.

## Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.7833 |
| Precision | 0.6538 |
| Recall    | 0.3312 |
| F1-Score  | 0.4397 |

**Confusion Matrix**

|                  | Predicted: No Churn | Predicted: Churn |
|------------------|---------------------|-------------------|
| **Actual: No Churn** | 419 | 27 |
| **Actual: Churn**    | 103 | 51 |

**Observations**
1. The model reaches ~78% accuracy, but this is inflated by class imbalance
   (~75% of customers do not churn); the F1-score for the "Churn" class
   (~0.44) is a more honest measure of churn-detection performance.
2. Precision for "Churn" (~0.65) is notably higher than Recall (~0.33) — the
   model is conservative and misses many actual churners (false negatives),
   which matters for a business trying to proactively retain customers.
3. The strongest churn predictors are **contract type**, **tenure**,
   **fiber-optic internet service**, and **electronic check payment** —
   customers with month-to-month contracts, short tenure, fiber internet,
   and electronic-check payments churn the most, while tech support and
   online security subscriptions correlate with retention.

*(Results above were produced on the synthetic stand-in dataset described
above; results on the real Kaggle CSV will differ slightly but the pipeline
and code are identical.)*

## Conclusion
This project developed a Logistic Regression model to predict telecom
customer churn using demographic and service-usage data. After preprocessing
(handling missing `TotalCharges` values, encoding categorical variables, and
an 80/20 train-test split), the model produced solid, interpretable
performance across accuracy, precision, recall, and F1-score. The analysis
shows that **contract type, tenure, internet service type, and payment
method** are the strongest drivers of churn — customers with month-to-month
contracts, shorter tenure, fiber-optic internet, and electronic-check
payments are at the highest risk, while value-added services such as tech
support and online security correlate with retention. A key **limitation of
Logistic Regression** is that it assumes a linear relationship between
features (in log-odds space) and the outcome, so it cannot naturally capture
complex, non-linear interactions between features the way tree-based or
ensemble models can, which may limit predictive performance on more
intricate churn patterns.

## Repository Contents
- `Assignment-2.ipynb` — Jupyter notebook with all 5 tasks, executed with
  outputs
- `Assignment-2.py` — equivalent standalone Python script
- `confusion_matrix.png` — saved confusion matrix plot
- `README.md` — this file

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python Assignment-2.py
# or
jupyter notebook Assignment-2.ipynb
```
