# Breast Cancer Wisconsin (Diagnostic) — KNN Classification

## Objective
Develop a K-Nearest Neighbors (KNN) classification model to predict whether a breast tumor is
**Malignant (M)** or **Benign (B)** based on diagnostic measurements derived from digitized images
of fine needle aspirates (FNA) of breast mass.

## Dataset Link
[Breast Cancer Wisconsin (Diagnostic) Data Set — Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

> **Note on the dataset used in this notebook:** The notebook loads the data via
> `sklearn.datasets.load_breast_cancer()`, which contains the same 30 diagnostic features and the
> same underlying UCI source data as the Kaggle CSV, so the notebook runs end-to-end without a
> Kaggle API key. A commented-out line in the notebook shows how to load the raw `data.csv`
> instead if you download it from Kaggle. As instructed, the raw dataset file itself is **not**
> uploaded to this repository — only the link above is provided.

## Libraries Used
- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `matplotlib` / `seaborn` — visualization (class distribution, confusion matrix, K vs. accuracy)
- `scikit-learn` — dataset loading, preprocessing (`LabelEncoder`, `StandardScaler`), model
  selection (`train_test_split`), the `KNeighborsClassifier` model, and evaluation metrics

## Methodology
1. **Data Understanding** — Loaded the dataset, inspected the first five records, identified the
   30 numerical diagnostic features and the categorical target variable (`diagnosis`), and
   reviewed dataset info and summary statistics.
2. **Data Preprocessing**
   - Checked for missing values (none present).
   - Dropped unnecessary columns (`id`, `Unnamed: 32`) where applicable.
   - Encoded the target variable (`diagnosis`) using `LabelEncoder` (B → 0, M → 1).
   - Standardized all feature values using `StandardScaler` (zero mean, unit variance), since KNN
     relies on distance calculations.
   - Split the data into 80% training and 20% testing sets using stratified sampling to preserve
     class balance.
3. **Model Development** — Trained a `KNeighborsClassifier` with **K = 5** on the scaled training
   data and generated predictions on the held-out test set.
4. **Model Evaluation** — Evaluated the model using Accuracy, Precision, Recall, and F1-Score, and
   visualized performance with a confusion matrix. Additionally explored how accuracy varies with
   different values of K (1–20) as supporting analysis.

## Results

| Metric | Score |
|---|---|
| Accuracy | 0.9561 |
| Precision | 0.9744 |
| Recall | 0.9048 |
| F1-Score | 0.9383 |

**Confusion Matrix (K = 5):**

|  | Predicted B | Predicted M |
|---|---|---|
| **Actual B** | 71 | 1 |
| **Actual M** | 4 | 38 |

### Observations
1. The KNN model with K = 5 achieves strong overall accuracy (~95.6%) on the test set, showing
   that the 30 diagnostic features provide good separability between malignant and benign tumors
   once standardized.
2. The confusion matrix shows very few misclassifications; in particular, 4 malignant cases were
   predicted as benign (false negatives) — the most clinically important error type to minimize,
   since it could delay treatment.
3. Testing K values from 1–20 showed accuracy is relatively stable across a range of K, with K = 5
   performing best on this split — very small K risks overfitting to noise, while very large K
   risks oversmoothing the decision boundary.

## Conclusion
This project developed a K-Nearest Neighbors (K = 5) classifier to distinguish malignant from
benign breast tumors using 30 numerical diagnostic features from the Breast Cancer Wisconsin
(Diagnostic) dataset. After standardizing the features and splitting the data 80/20, the model
achieved strong accuracy (95.6%), precision (97.4%), recall (90.5%), and F1-score (93.8%), with the
confusion matrix confirming very few misclassifications.

**Feature scaling is critical for KNN** because the algorithm classifies points based on Euclidean
distance; without standardization, features with larger numeric ranges (e.g., *area*) would
dominate the distance calculation and drown out smaller-scale but equally important features
(e.g., *smoothness*), biasing predictions.

**A key limitation of KNN** is that it is computationally expensive at prediction time — it must
compute distances to all training points for every new prediction — making it slow and
memory-intensive on large datasets, and it is also sensitive to irrelevant/redundant features and
to the choice of K.

## Repository Structure
```
.
├── Assignment-4.ipynb   # Full notebook with code, outputs, and visualizations
├── README.md            # This file
└── requirements.txt     # Python dependencies
```

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook Assignment-4.ipynb
```
