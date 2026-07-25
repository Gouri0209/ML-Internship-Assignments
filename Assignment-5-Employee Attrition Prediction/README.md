# Employee Attrition Prediction — Decision Tree vs Random Forest

Assignment 5: A comparison of Decision Tree and Random Forest classifiers for predicting employee attrition using the IBM HR Analytics dataset.

## Objective

To identify employees likely to leave the organization based on demographic, professional, and work-related attributes, by building and comparing two classification models: a Decision Tree Classifier and a Random Forest Classifier (100 estimators).

## Dataset

**IBM HR Analytics Employee Attrition & Performance**
Kaggle link: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

> The dataset is **not included in this repository**. Download `WA_Fn-UseC_-HR-Employee-Attrition.csv` from the Kaggle link above and place it inside the `data/` folder before running the notebook:
> ```
> data/WA_Fn-UseC_-HR-Employee-Attrition.csv
> ```

## Repository Structure

```
.
├── Assignment-5.ipynb   # Main notebook: all 5 tasks
├── README.md
├── requirements.txt
├── data/                # Place the downloaded dataset CSV here (not tracked in git)
└── .gitignore
```

## Libraries Used

- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — preprocessing (`LabelEncoder`, `train_test_split`), models (`DecisionTreeClassifier`, `RandomForestClassifier`), and evaluation metrics

Install everything with:
```bash
pip install -r requirements.txt
```

## Methodology

1. **Data Understanding** — Loaded the dataset with Pandas, inspected the first five records, identified numerical vs. categorical features and the target variable (`Attrition`), and reviewed dataset info and summary statistics.
2. **Data Preprocessing** — Checked for missing values, dropped non-informative/constant columns (`EmployeeCount`, `StandardHours`, `Over18`, `EmployeeNumber`), label-encoded all categorical variables, and split the data 80/20 into training and test sets (stratified on the target).
3. **Model Development** — Trained a `DecisionTreeClassifier` and a `RandomForestClassifier` (`n_estimators=100`) on the same training set, using a fixed `random_state` for reproducibility.
4. **Model Evaluation** — Compared both models on Accuracy, Precision, Recall, and F1-Score; generated confusion matrices for both models; and plotted the top 15 feature importances from the Random Forest model.
5. **Conclusion** — Summarized which model performed better and why.

## Results

Run `Assignment-5.ipynb` end-to-end against the real dataset to generate:
- A metrics comparison table (Accuracy / Precision / Recall / F1-Score) for both models
- Side-by-side confusion matrices (`confusion_matrices.png`)
- A Random Forest feature importance plot (`feature_importance.png`)
- Written observations comparing the two models

*(Fill in your actual numbers here after running the notebook, e.g.:)*

| Model          | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| Decision Tree  |    —     |     —     |   —    |    —     |
| Random Forest  |    —     |     —     |   —    |    —     |

## Model Comparison

- Random Forest generally achieves higher accuracy and a better F1-score than a single Decision Tree, because it aggregates predictions from many trees trained on bootstrapped samples with random feature subsets (bagging), reducing variance and overfitting.
- The Decision Tree is more interpretable (you can visualize and trace its exact splits) but is more prone to overfitting the training data.
- The Random Forest's feature importance plot highlights which attributes (e.g., `OverTime`, `MonthlyIncome`, `Age`) most strongly drive its predictions, even though the model itself isn't as directly interpretable as a single tree.

## Conclusion

See the final markdown cell in `Assignment-5.ipynb` for the full 150–200 word conclusion, covering which model performed better, why Random Forest often outperforms a single Decision Tree, and one limitation of each model.

## How to Run

```bash
git clone <this-repo-url>
cd <this-repo>
pip install -r requirements.txt
# Download the dataset from Kaggle and place it at data/WA_Fn-UseC_-HR-Employee-Attrition.csv
jupyter notebook Assignment-5.ipynb
```
