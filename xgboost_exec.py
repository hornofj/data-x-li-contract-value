import xgboost as xgb
import pickle
import numpy as np
import time

#import pandas as pd 
#from statistics import mean
#import math
#from datetime import datetime
#import statistics as st
#import datetime
#import random
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import make_scorer
#from sklearn.model_selection import train_test_split


# Load data from pickle
with open(r"../data-x-li-data/df_merged_train_test_05p.pickle", "rb") as input_file:
    X_train, y_train, X_test, y_test = pickle.load(input_file)

# Instantiate a xgb.XGBRegressor 
gbm0 = xgb.XGBRegressor(n_estimators = 50, learning_rate = 0.1, objective='reg:squarederror', seed = SEED)

# Fit XGBoost with SciKit
gbm0.fit(X_train, y_train)

# Predict the test set labels 'y_pred0'
y_pred0 = gbm0.predict(X_test)

# Evaluate the test set RMSE
#rmse_test0 = mean_squared_error(y_test, y_pred0, squared=False)
#print(rmse_test0)

# Evaluate using MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Evaluate the test set RMSE
MAPE_test0 = mean_absolute_percentage_error(y_test, y_pred0)
print(MAPE_test0)


##########################
#### Grid optimization ###
##########################

# Setup params grid
param_grid = {'learning_rate': [0.01, 0.1, 0.5], # alias eta, Step size shrinkage used in update to prevents overfitting.  
    'n_estimators': [20, 50, 100],
    'subsample': [0.5, 0.8, 1], # Subsample ratio of the training instances
    'max_depth': [3, 5, 10],
    'colsample_bytree': [0.5, 1] # colsample_bytree is the subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.
    }



# Split the dataset into training_validation and testing part
# 95 : 5 

validation_ratio = 0.05

X_train_valid, X_test_valid, y_train_valid, y_test_valid = train_test_split( 
    X_train, y_train,
    test_size = validation_ratio, 
    random_state = SEED
    )

# Setup params grid
# initial ranges
GRID_SIZE = 2
N_ESTIMATORS_MIN = 200
N_ESTIMATORS_MAX = 2000
MAX_DEPTH_MIN = 5
MAX_DEPTH_MAX = 20
LEARNING_RATE_COEF_MIN = -3
LEARNING_RATE_COEF_MAX = -0.5
MIN_DATA_IN_LEAF_MIN = 20
MIN_DATA_IN_LEAF_MAX = 200
LEARNING_RATE_EXPL = 0 # keep 0 here, otherwise LEARNING_RATE_COEF will be omitted

SEED = 333

#random.seed(SEED) # DEACTIVATED
grid = pd.DataFrame({
    'n_estimators' : [random.randint(N_ESTIMATORS_MIN, N_ESTIMATORS_MAX) for x in range(GRID_SIZE)],
    'max_depth' : [random.randint(MAX_DEPTH_MIN, MAX_DEPTH_MAX) for x in range(GRID_SIZE)],
    'learning_rate' : np.power([10 for x in range(GRID_SIZE)], [random.uniform(LEARNING_RATE_COEF_MIN,
     LEARNING_RATE_COEF_MAX) for x in range(GRID_SIZE)]) if LEARNING_RATE_EXPL==0 else [LEARNING_RATE_EXPL for x in range(GRID_SIZE)],
    'min_data_in_leaf' : [random.randint(MIN_DATA_IN_LEAF_MIN, MIN_DATA_IN_LEAF_MAX) for x in range(GRID_SIZE)]
    })


def fit_regressor(X_train, y_train, X_test, y_test, params):
    # Instantiate a lgb.LGBMRegressor
    lgbm = lgb.LGBMRegressor(seed=SEED,
    n_estimators=int(params['n_estimators']),
    max_depth=int(params['max_depth']),
    learning_rate=params['learning_rate'],
    min_data_in_leaf=int(params['min_data_in_leaf'])
    )
    #Fit with SciKit
    lgbm.fit(X_train, y_train)
    # Predict the test set labels 'y_pred0'
    y_pred0 = lgbm.predict(X_test)
    # Evaluate the test set RMSE
    MAPE_test0 = mean_absolute_percentage_error(y_test, y_pred0)
    return MAPE_test0, str(tuple(lgbm0.feature_importances_)), lgbm


# fit regressor and compute MAPE for each param vector
tic = time.perf_counter() #begin timing
MAPE_list = np.empty(grid.shape[0])
FIMP_list = ['' for x in range(grid.shape[0])]
lgbm_list = ['' for x in range(grid.shape[0])]
for i in range(grid.shape[0]):
    MAPE_list[i], FIMP_list[i], lgbm_list[i]  = fit_regressor(
        X_train_valid, y_train_valid,
        X_test_valid, y_test_valid,
         grid.iloc[i]
        )
time_fit_cv = time.perf_counter() - tic #save timer
grid['MAPE'] = MAPE_list #add MAPE to grid  
grid['FIMP'] = FIMP_list #add FIMP to grid  


best_fit_order = np.argpartition(grid['MAPE'], 1)
best_fit_no = np.argmin(grid['MAPE'])
sec_best_fit_no = best_fit_order[1]

low_MAPE = grid.iloc[best_fit_no]['MAPE']
best_params = grid.iloc[best_fit_no].drop('MAPE').drop('FIMP').astype(str).to_dict()
train_size = X_train_valid.shape[0]
columns = str(X_train_valid.columns.values)
fimp = grid.iloc[best_fit_no]['FIMP']

y_pred = lgbm_list[best_fit_no].predict(X_test)
MAPE_test_set = mean_absolute_percentage_error(y_test, y_pred)

sec_best_params = grid.iloc[sec_best_fit_no].drop('MAPE').drop('FIMP').astype(str).to_dict()

y_pred = lgbm_list[sec_best_fit_no].predict(X_test_valid)
sec_best_MAPE = mean_absolute_percentage_error(y_test_valid, y_pred)

y_pred = lgbm_list[best_fit_no].predict(X_train_valid)
MAPE_train_set = mean_absolute_percentage_error(y_train_valid, y_pred)

print(low_MAPE)
print(best_params)
print(FIMP_list[best_fit_no])
