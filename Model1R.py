# Import libraries & Modules
import pandas as pd
import numpy as np
import sklearn as sk
import xgboost as xgb
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
import optuna
import os
import shutil
import pickle

# Import dataset
X = pd.read_csv('/Users/coletteford/Desktop/IGP/Code/German_X.csv')
y = pd.read_csv('/Users/coletteford/Desktop/IGP/Code/German_y.csv')
X.info()

X = X.to_numpy()
y = y.to_numpy()

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = 0.2)

# Transform train and test into xgboost 
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)

# Hyperparameter tuning
CV_RESULT_DIR = "./xgboost_cv_results"

def objective(trial):

    # Define hyperparameter Search Space (ie. ranges)
    hyperparameters = {
        'eta': trial.suggest_float('eta', 0.005, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.8, 1.0),
        'max_leaves': trial.suggest_int('max_leaves', 10, 200),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'gamma': trial.suggest_float('gamma', 0.0, 0.02),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 0, 10),
    
    }
    # Cross validate the hyperparameter tuning using training dataset
    xgb_cv_results = xgb.cv(
        params=hyperparameters,
        dtrain=dtrain,
        nfold=10,
        num_boost_round=10000,
        early_stopping_rounds=100,
        seed=42,
        verbose_eval=False,
    )
    
    # Set n_estimators for trials
    trial.set_user_attr('n_estimators', len(xgb_cv_results))

    # Save cv results
    filepath = os.path.join(CV_RESULT_DIR, '{}.csv'.format(trial.number))
    xgb_cv_results.to_csv(filepath, index=False)

    # Extract the best score
    best_score = xgb_cv_results['test-rmse-mean'].values[-1]
    return best_score

# Initiate study
if __name__ == '__main__':
    if not os.path.exists(CV_RESULT_DIR):
        os.mkdir(CV_RESULT_DIR)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, timeout=600)

    print('Number of finished trials: ', len(study.trials))
    print('Best trial: ')
    trial = study.best_trial

    # Create & populate dictionary of best tuned hyperparameters
    tuned_params = {}
    print(' Value: {}'.format(trial.value))
    print(' Tuned Hyperparameters: ')
    
    for key, value in trial.params.items():
        print('  {}: {}'.format(key, value))
        tuned_params[str(key)] = value
        
print('   Number of estimators: {}'.format(trial.user_attrs['n_estimators']))

# Remove cv result directory
shutil.rmtree(CV_RESULT_DIR)


# Final model trained on whole training set and tuned hyperparameters
bstr = xgb.train(tuned_params, dtrain)

# Export to pkl file
filename = 'Model1R.pkl'
pickle.dump(bstr, open(filename, 'wb'))


