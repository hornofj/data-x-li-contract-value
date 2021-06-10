import pandas as pd 
import numpy as np
from statistics import mean
import math
from datetime import datetime
import statistics as st
import datetime
import time
import pickle
import random
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split

#Load the file containing variables [X_train, y_train, X_test, y_test]
with open(r"../data-x-li-data/df_merged_train_test_005p.pickle", "rb") as input_file:
    X_train, y_train, X_test, y_test = pickle.load(input_file)

#%reset -f
SEED = 333

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Instantiate a lgb.LGBMRegressor
#lgbm0 = lgb.LGBMRegressor(seed=SEED)
lgbm0 = lgb.LGBMRegressor(n_estimators = 14000, max_depth = 8, learning_rate = 0.283, min_data_in_leaf = 20, seed=SEED)
print(lgbm0)

# Fit with SciKit
print(datetime.datetime.now())
lgbm0.fit(X_train, y_train)
print(datetime.datetime.now())

def predict_and_evaluate(x_t, y_t):
    # Predict the test set labels 'y_pred0'
    y_pred0 = lgbm0.predict(x_t)
    # Evaluate the test set RMSE
    MAPE_t0 = mean_absolute_percentage_error(y_t, y_pred0)
    print(MAPE_t0)
    print(datetime.datetime.now())

print(datetime.datetime.now())
print('test MAPE:')
predict_and_evaluate(X_test, y_test)
print('train MAPE:')
predict_and_evaluate(X_train, y_train)