import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import sklearn.metrics as skm
import optuna
import pickle


# Import dataset
X = pd.read_csv('/Users/coletteford/Desktop/IGP/Code/German_X.csv')
y = pd.read_csv('/Users/coletteford/Desktop/IGP/Code/German_y.csv')

# Transform to arrays for compatibility
X = X.to_numpy()
y = y.to_numpy() 

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = 0.2)

# Transform train and test into xgboost 
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)

# Hyperparameter tuning
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
    # Define model to be used for tuning
    bst = xgb.train(hyperparameters, dtrain)

    # Evaluate model for each 
    preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    rmse = np.sqrt(np.mean(np.square(preds - y_test)))
    return rmse

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, timeout=600)

    print('Number of finished trials: ', len(study.trials))
    print('Best trial: ')
    trial = study.best_trial

    tuned_params = {}
    print(' Value: {}'.format(trial.value))
    print(' Tuned Hyperparameters: ')
    
    for key, value in trial.params.items():
        print('  {}: {}'.format(key, value))
        tuned_params[str(key)] = value


# Final model
bstr = xgb.train(tuned_params, dtrain)

filename = 'Model1R-Simple.pkl'
pickle.dump(bstr, open(filename, 'wb'))