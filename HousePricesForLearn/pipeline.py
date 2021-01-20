from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import pandas as pd
from sklearn.model_selection import train_test_split

# https://www.kaggle.com/aashita/advanced-pipelines-tutorial
# https://www.kaggle.com/dansbecker/pipelines

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
#low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
#                        X_train_full[cname].dtype == "object"]

categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and
                    X_train_full[cname].dtype == "object"]

# Select numeric columns
numerical_cols = [
    cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
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
## Pipeline

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# ++++++++++++++++++

## ================================================
## Improve the Best Model

from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

my_model_3 = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)

gbm_params = {"colsample_bytree": [0.5, 0.7, 1.0],
              "min_child_weight": [1.0, 1.2],
              "learning_rate": [0.01, 0.02, 0.03],
              'max_depth': [4, 6, 8],
              'n_estimators': [1000]}


# Kaggle Score: 14720.15311
# MAE: 16922.81834599743
# Rank: 3154 out of 57,630 Top 6%
#best_params_={'colsample_bytree': 1.0, 'learning_rate': 0.03, 'max_depth': 6, 'n_estimators': 1000}
gbm_params = {"colsample_bytree": [1.0],
              "learning_rate": [0.01, 0.03],
              'max_depth': [6, 8],
              'n_estimators': [1000]}


#reg_cv.fit(X_train, y_train)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('xgbrg', XGBRegressor())
                      ])

param_grid = {
    "xgbrg__n_estimators": [10, 50, 100, 500],
    "xgbrg__learning_rate": [0.1, 0.5, 1],
}

fit_params = {"xgbrg__eval_set": [(X_valid, y_valid)],
              "xgbrg__early_stopping_rounds": 10,
              "xgbrg__verbose": False}

searchCV = GridSearchCV(my_pipeline, cv=5, param_grid=param_grid)

searchCV.fit(X_train, y_train, **fit_params)


# reg_cv = GridSearchCV(pipeline, gbm_params, verbose=3)
# Fit the model

# reg_cv.fit(X_train, y_train,
#                  early_stopping_rounds=5,
#                  eval_set=[(X_valid, y_valid)],
#                  verbose=False)

# Get predictions

print(f"best_params_={searchCV.best_params_}")

predictions_3 = reg_cv.predict(X_valid)


# Calculate MAE
# Mean Absolute Error: 16802.965325342466
mae_3 = mean_absolute_error(predictions_3, y_valid)
print(f"MAE New: {mae_3}")

print(searchCV.cv_results_['mean_train_score'])

print(searchCV.cv_results_['mean_test_score'])

print(searchCV.cv_results_['mean_train_score'].mean(), searchCV.cv_results_['mean_test_score'].mean())

predictions = reg_cv.predict(X_test)

output = pd.DataFrame({'Id': X_test.index, 'SalePrice': predictions})
output.to_csv('xgboost_tuned_submission.csv', index=False)
print("Your submission was successfully saved!")
