#
import numpy as np
import pandas as pd

## Read Train and Test data

train = pd.read_csv("../input/sales_train.csv")
test = pd.read_csv("../input/test.csv")

items = pd.read_csv('../input/items.csv')
item_cats = pd.read_csv('../input/item_categories.csv')
shops = pd.read_csv('../input/shops.csv')


## Split out target

y = MyTrainDF['item_cnt_month'].values.reshape(-1, 1)

## Split out Training to training and Validation Data
#  
Y_train = X_train.item_cnt_month

X_train, X_valid, y_train, y_valid = train_test_split(
    train, y, test_size=0.4, random_state=0)


regressor = LinearRegression()
regressor.fit(X_train, y_train)  # train the model
