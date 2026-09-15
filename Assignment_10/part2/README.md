# Assignment 10B: Spotting Credit Card Fraud with a Neural Network

## About the Project

An end-to-end fraud-detection pipeline built around an Artificial Neural Network. It walks through downloading the dataset, exploring it, preprocessing, training with class weights to handle imbalance, running several model variants, and judging the results.

## Environment & Libraries

- Python 3
- Jupyter Notebook
- Pandas / NumPy
- Matplotlib / Seaborn
- scikit-learn
- TensorFlow / Keras
- KaggleHub

## Workflow Highlights

- Pull the Kaggle dataset with KaggleHub
- Remove duplicates and run quality checks
- Stratified train/test split that keeps class balance intact
- StandardScaler fit on the training split only
- Class weights to counter heavy imbalance
- Dense ANN layouts with varied activations and optimizers
- Confusion matrix and full classification report
- Accuracy, precision, recall, and F1 scoring

## Key Takeaways

- Build a complete fraud-detection learning pipeline
- Keep preprocessing free of data leakage
- Work with severely skewed classification data
- Treat recall as the priority for catching fraud
- Compare and interpret the ANN experiments
