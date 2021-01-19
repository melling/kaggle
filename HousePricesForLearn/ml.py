from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import pandas as pd
from sklearn.model_selection import train_test_split

## Pipelines scores was best: 16432.94092

# Read the data
# X = pd.read_csv('../input/train.csv', index_col='Id')
# X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

X = pd.read_csv('input/train.csv', index_col='Id')
X_test_full = pd.read_csv('input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice              
X.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# Select numeric columns
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# One-hot encode the data (to shorten the code, we use pandas)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)

# +++++++++++++++

# Step 1: Build model

# Define the model
my_model_1 = XGBRegressor()
# Fit the model
my_model_1.fit(X_train, y_train)

# +++++++++++++++

# Get predictions
predictions_1 = my_model_1.predict(X_valid)
# 17662.736729452055
print("Mean Absolute Error: " + str(mean_absolute_error(predictions_1, y_valid)))

## In[7]

# Calculate MAE
mae_1 = mean_absolute_error(predictions_1, y_valid)  # Your code here

# 17662.736729452055
print("Mean Absolute Error:", mae_1)

## Step 2: Improve the model

## In[9]

# Define the model

my_model_2 = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)

# Fit the model
my_model_2.fit(X_train, y_train,
               early_stopping_rounds=5,
               eval_set=[(X_valid, y_valid)],
               verbose=False)
# Get predictions

predictions_2 = my_model_2.predict(X_valid)


# Calculate MAE
# Mean Absolute Error: 16802.965325342466
mae_2 = mean_absolute_error(predictions_2, y_valid)
print("Mean Absolute Error:", mae_2)

## ================================================
## Improve the Best Model

from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

my_model_3 = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)

gbm_params = {"colsample_bytree": [0.5, 0.7, 1.0],
              "min_child_weight": [1.0, 1.2],
              "learning_rate": [0.01, 0.02, 0.03],
              'max_depth': [4, 6, 8],
              'n_estimators': [1000]}

# best_params_={'colsample_bytree': 1.0, 'learning_rate': 0.03, 'max_depth': 6, 'n_estimators': 1000}
# best_params_={'colsample_bytree': 1.0, 'learning_rate': 0.03, 'max_depth': 6, 'n_estimators': 1000}

# Kaggle Score: 14720.15311
# MAE: 16922.81834599743
# Rank: 3154 out of 57,630 Top 6%
#best_params_={'colsample_bytree': 1.0, 'learning_rate': 0.03, 'max_depth': 6, 'n_estimators': 1000}
gbm_params = {"colsample_bytree": [1.0],
              "learning_rate": [0.01, 0.03],
              'max_depth': [4, 6, 8],
              'n_estimators': [1000, 1500]}

reg_cv = GridSearchCV(my_model_3, gbm_params, verbose=3)
#reg_cv.fit(X_train, y_train)

# Fit the model
reg_cv.fit(X_train, y_train,
               early_stopping_rounds=5,
               eval_set=[(X_valid, y_valid)],
               verbose=False)
# Get predictions

print(f"best_params_={reg_cv.best_params_}")

predictions_3 = reg_cv.predict(X_valid)


# Calculate MAE
# Mean Absolute Error: 16802.965325342466
mae_3 = mean_absolute_error(predictions_3, y_valid)
print(f"MAE Best: {mae_2} MAE New: {mae_3}")
print(f"Compared to Best:{mae_2 - mae_3}")

#X_test_full.head()

predictions = reg_cv.predict(X_test)

output = pd.DataFrame({'Id': X_test.index, 'SalePrice': predictions})
output.to_csv('xgboost_tuned_submission.csv', index=False)
print("Your submission was successfully saved!")
