import xgboost as xgb
import pickle
import numpy as np
import time

from sklearn.metrics import mean_squared_error

SEED = 500

# Load data from pickle
with open(r"../data-x-li-data/df_merged_train_test_05p.pickle", "rb") as input_file:
    X_train, y_train, X_test, y_test = pickle.load(input_file)

# Instantiate a xgb.XGBRegressor
#gbm0 = xgb.XGBRegressor(n_estimators = 50, learning_rate = 0.1, objective='reg:squarederror', seed = SEED)
gbm0 = xgb.XGBRegressor(n_estimators = 14000, learning_rate = 0.28, max_depth = 8, objective='reg:squarederror', seed = SEED)

# Fit XGBoost with SciKit
gbm0.fit(X_train, y_train)

# Predict the test set labels 'y_pred0'
y_pred0 = gbm0.predict(X_test)

# Evaluate the test set RMSE
rmse_test0 = mean_squared_error(y_test, y_pred0, squared=False)
print(rmse_test0)

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
param_grid = {'learning_rate': [0.1, 0.2, 0.3], # alias eta, Step size shrinkage used in update to prevents overfitting.  
    'n_estimators': [10000, 15000, 20000],
    'subsample': [1], # Subsample ratio of the training instances
    'max_depth': [6, 8, 10, 12],
    'colsample_bytree': [1] # colsample_bytree is the subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.
    }

#instantiate XGBRegressor 
gbm = xgb.XGBRegressor(seed=SEED, objective='reg:squarederror')
grid_mse = GridSearchCV(estimator=gbm,
                        param_grid=param_grid,
                        scoring='neg_mean_squared_error', 
                        cv=3, 
                        verbose=1, 
                        n_jobs=-1)

# fit GridSearchCV 
tic = time.perf_counter() # begin timing
grid_mse.fit(X_train, y_train)
time_fit_cv = time.perf_counter() - tic # save timer

print("Best parameters found: ",grid_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))

#extract the estimator best_estimator_ 
gbm_ins = grid_mse.best_estimator_

# Predict the test set labels 'y_pred'
y_pred = gbm_ins.predict(X_test)

# Evaluate the test set RMSE
rmse_test = mean_squared_error(y_test, y_pred, squared=False)
print(rmse_test)


#best_fit_order = np.argpartition(grid['MAPE'], 1)
#best_fit_no = np.argmin(grid['MAPE'])
#sec_best_fit_no = best_fit_order[1]

#low_MAPE = grid.iloc[best_fit_no]['MAPE']
#best_params = grid.iloc[best_fit_no].drop('MAPE').drop('FIMP').astype(str).to_dict()
#train_size = X_train_valid.shape[0]
#columns = str(X_train_valid.columns.values)
#fimp = grid.iloc[best_fit_no]['FIMP']

#y_pred = lgbm_list[best_fit_no].predict(X_test)
#MAPE_test_set = mean_absolute_percentage_error(y_test, y_pred)

#sec_best_params = grid.iloc[sec_best_fit_no].drop('MAPE').drop('FIMP').astype(str).to_dict()

#y_pred = lgbm_list[sec_best_fit_no].predict(X_test_valid)
#sec_best_MAPE = mean_absolute_percentage_error(y_test_valid, y_pred)

#y_pred = lgbm_list[best_fit_no].predict(X_train_valid)
#MAPE_train_set = mean_absolute_percentage_error(y_train_valid, y_pred)

#print(low_MAPE)
#print(best_params)
#print(FIMP_list[best_fit_no])
