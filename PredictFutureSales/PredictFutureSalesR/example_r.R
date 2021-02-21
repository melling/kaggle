# https://www.kaggle.com/sywang113/linear-regression-model-sales-prediction/

## Kaggle Score: 1.27009, 8188/10281, Top 80%

# 1.Import all datasets
train <- read.csv("../input/sales_train.csv")
test <- read.csv("../input/test.csv")
item <- read.csv("../input/items.csv")
item_cat <- read.csv("../input/item_categories.csv")
shop <- read.csv("../input/shops.csv")

# And the libraries we need
library(data.table)
library(dplyr)
library(lubridate)

# 2.Data preparation/Cleaning data
# 2.1 Check all related datasets structure
# head(item)
# head(item_cat)
# head(test)
# head(shop)

# 2.2 Merge the above 4 datasets
data1 = inner_join(train, shop, by = "shop_id")
data2 = inner_join(data1,item, by = "item_id")
sales = inner_join(data2, item_cat, by = "item_category_id")
# Now drop data1&data2 and examine this dataset
rm(data1)
rm(data2)

head(sales)
tail(sales)
str(sales)
colnames(sales)
dim(sales)
summary(sales)

# 2.3 Sort data structure
# Let's convert the format of sales$date from factor to date.
sales$date = dmy(sales$date)
# And we may want to know sales in terms of day, month, weekday and year, so create
# four new columns.
sales$day = day(sales$date)
sales$month = month(sales$date)
sales$weekday = wday(sales$date)
sales$year = year(sales$date)
# Transform IDs into categorical variables
sales$item_id = factor(sales$item_id)
sales$item_category_id = factor(sales$item_category_id)
sales$shop_id = factor(sales$shop_id)

# 2.4 Look for missing data
sum(is.na(sales))
# OR
anyNA(sales)

# 2.5 Check the Normality
ks.test(sales$item_cnt_day, "pnorm", mean(sales$item_cnt_day), sd(sales$item_cnt_day))
# Obviously, it is not normal, so we need to transform it in linear regression model.

# 3.Run linear regression model
# 3.1 Create a new data.table with a new variable "item_cnt_month"
sales_data = as.data.table(sales)
sales_month = sales_data[, list(item_cnt_month=(sum(item_cnt_day))/12), by = c("date_block_num", "month","shop_id", "item_category_id", "item_id", "item_price")]
# 2 is added to avoid negative values for log transformation, it will be subtracted later
sales_month$item_cnt_month = sales_month$item_cnt_month + 2

# 3.2 Try first model.
lm1 = lm(formula=log(item_cnt_month) ~ date_block_num + month + 
           shop_id + item_category_id, 
         data = sales_month)
summary(lm1)

# 3.4 Use test data. Assign 11 to the column "month" 
# and 34 to the column "date_block_num" for November.
test$month <- 11
test$date_block_num <- 34
test$shop_id <- as.factor(test$shop_id)
# Add the variable "Item_id" to test data by joining test data and item data.
test.com = inner_join(test, item, by = "item_id")
test.com$item_category_id = as.factor(test.com$item_category_id)

# 3.5 Predict the result
result = predict(lm1, newdata = test.com, type = "response")
# Subtract 2 from pred_mo.test to reverse the addition
result <- exp(result) - 2
submission =  data.frame(ID = test$ID,
                         item_cnt_month = result)
submission %>%
  write.csv(., file="submission_r.csv", row.names = F, quote=F)
