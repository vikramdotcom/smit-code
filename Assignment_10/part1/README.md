# Assignment 10A: Predicting Student Final Grades with Neural Networks

## About the Project

A deep-learning experiment that forecasts the `final_grade` column from the 300K student-performance dataset. Dense feed-forward neural networks are built and tuned, comparing several activation functions and optimizers to find the strongest combination.

## Environment & Libraries

- Python 3
- Jupyter Notebook
- Pandas / NumPy
- Matplotlib / Seaborn
- scikit-learn
- TensorFlow / Keras

## Workflow Highlights

- Inspect the data and strip out duplicate rows
- Encode all categorical columns
- Split features from the target label
- Subsample the large dataset to speed up testing
- Split into train/test sets
- Standardize numeric inputs with StandardScaler
- Construct dense neural-net architectures
- Evaluate ReLU and Tanh activations
- Benchmark Adam, SGD, and RMSprop optimizers
- Report MSE loss alongside MAE, RMSE, and R²

## Key Takeaways

- Get tabular data ready for a neural network
- Experiment with sequential model layouts
- Compare how different settings train
- Read loss curves and prediction metrics
- Reason about optimizer and activation choices
