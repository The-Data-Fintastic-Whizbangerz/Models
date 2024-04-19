# Import libraries & Modules
import pandas as pd
import numpy as np
import sklearn as sk
import xgboost as xgb
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm

import pickle
from xgboost import XGBRegressor

# Import dataset
X = pd.read_csv('german_x_20240418.csv')
y = pd.read_csv('german_y_20240418.csv')
X.info()

X = X.to_numpy()
y = y.to_numpy()

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = 0.2)

# Final model with tuned hyperparameters
bstr = XGBRegressor(
    learning_rate= 0.13242674454922954,
    max_depth= 24,
    max_leaves= 10,
    subsample= 0.8783082904441846,
    colsample_bytree= 0.8104855152582308,
    min_child_weight= 2,
    min_split_loss= 0.01761025238301419)

# Export to pkl file
filename = 'XGBR.pkl'
pickle.dump(bstr, open(filename, 'wb'))


