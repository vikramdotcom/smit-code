# Assignment 08 - Classification Assignment

This folder contains my solution for the **Classification Assignment**. The objective of this assignment is to perform proper data preprocessing and develop a classification model for three different datasets.

## Folder Structure

```
ASSIGNMENT 08/
│
├── 1. Framingham Heart Disease Dataset/
│   ├── framingham.csv
│   └── Framingham_Heart_Disease_Classification.ipynb
│
├── 2. Heart Failure Dataset/
│   ├── heart_failure_clinical_records_dataset.csv
│   └── Heart_Failure_Classification.ipynb
│
└── 3. California Housing Dataset/
    ├── housing.csv
    └── Housing_Price_Classification.ipynb
```

## Datasets

| Dataset | Target Variable | Type |
|---------|----------------|------|
| Framingham Heart Disease | `TenYearCHD` (10 year risk of heart disease) | Binary Classification |
| Heart Failure Clinical Records | `DEATH_EVENT` (death or survival) | Binary Classification |
| California Housing | `price_category` (expensive / affordable house) | Binary Classification (created from `median_house_value`) |

## Methodology

Each notebook follows the same complete machine learning pipeline:

1. **Dataset Analysis** - loading, shape, info, statistical summary and exploratory data analysis.
2. **Handling Missing Values** - imputation using mode / median where required.
3. **Encoding** - one-hot encoding for categorical columns.
4. **Feature Scaling** - `StandardScaler` applied to the features.
5. **Balancing the Dataset** - `SMOTE` applied where the dataset was imbalanced.
6. **Training Classification Models** - 7 different classifiers trained and compared:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - Support Vector Machine (SVM)
   - Naive Bayes
7. **Evaluation** - Accuracy, Precision, Recall, F1-Score, Classification Report and Confusion Matrix generated for every model.
8. **Best Model Selection** - best model chosen based on performance.
9. **Hyperparameter Tuning** - `GridSearchCV` used to tune the best model and improve accuracy further.
10. **Final Comparison** - model performance compared before and after tuning.

## Results

| Dataset | Best Model | Accuracy (Before Tuning) | Accuracy (After Tuning) |
|---------|-----------|--------------------------|-------------------------|
| Framingham Heart Disease | Random Forest | 0.8054 | 0.8007 (CV: 0.9092) |
| Heart Failure | Random Forest | 0.8000 | 0.8500 |
| California Housing | Random Forest | 0.8939 | 0.8944 |

**Notes:**

- For the **Framingham** dataset the test accuracy of the tuned model stayed around 0.80, but the cross-validation score improved to 0.91, which shows the tuned model generalizes better.
- For the **Heart Failure** dataset, Logistic Regression had the highest initial accuracy (0.8167), but the tuned Random Forest surpassed it and reached 0.85.
- For the **California Housing** dataset, the improvement after tuning was small because Random Forest was already performing very well (~0.89).

## How to Run

1. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

2. Open any notebook with Jupyter:

```bash
jupyter notebook
```

3. Run all the cells (Kernel -> Restart & Run All).

## Requirements

- Python 3.8+
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- imbalanced-learn (for SMOTE)
- Jupyter Notebook

## Submission

- **Source Code:** Included in the notebooks
- **Jupyter Notebook (.ipynb):** One notebook per dataset
- **Dataset:** CSV files included in their respective folders
- **README file:** This file
