import random
import pandas as pd
import numpy as np

### Training the ML model using Python -

data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
data = data.sample(frac=1).reset_index(drop=True)   # jumbling the samples
X = data[['age', 'BMI', 'BP', 'blood_sugar', 'Gender']].values
y = data['disease_score']

# Splitting the training dataset (70% - training set, 30% - test set) -
random_seed = 999
training_data=len(X)
test_data=int(training_data - 0.30)
indices = list(range(training_data))
random.shuffle(indices)     # to shuffle the data - to avoid the bias
train_indices = indices[test_data:]
test_indices = indices[:test_data]
X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

# Scaling the data (Z-score normalization) -
Mean = np.mean(X_train, axis=0)
standard_deviation = np.std(X_train, axis=0)
X_train_scaled = (X_train - Mean) / standard_deviation
X_test_scaled = (X_test - Mean) / standard_deviation
