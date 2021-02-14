# scikit learn linear regression example

Given daily sales data.
Predict total sales for every product and store in the **next month**.

Features = []
target = item_cnt_month

## Standard Imports

```python
import numpy as np
import pandas as pd
```

## Linear Regression Imports

```python
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
```

## Other Imports

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor
from xgboost import plot_importance

```

## References

- https://www.kaggle.com/obiaf88/simple-linear-model-for-sales-predictions/
- https://www.kaggle.com/kaveeshashah/linear-regression-sample
