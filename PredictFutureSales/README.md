
# https://www.kaggle.com/c/competitive-data-science-predict-future-sales

Shops with [English Names](shops_eng.csv)

## EDA Notebooks

- https://www.kaggle.com/kmezhoud/ggplot-eda


## Model Notebooks

- https://www.kaggle.com/dlarionov/feature-engineering-xgboost
- https://www.kaggle.com/benamarareda/predit-sales
- https://www.kaggle.com/junota/predict-future-sales-naive-forecasting-xgboost


## Data Processing Checklist

- Remove items with price > 100000 and/or sales > 1001 
- English Shop names
- Merge data into one data frame 
- Shops 0/57,1/58, 10/11 are duplicates 39/40
- Take mean of item_price - apply to missing data
   - There is one item with price below zero. Fill it with median.
- Aggregate daily sales data to monthly revenue by item
- Extract item category
   - Each category contains type and subtype in its name
- Extract city from shop name
   - Each shop_name starts with the city name
- LabelEncoder
- date_block_num - 1..33 data predict - 34
