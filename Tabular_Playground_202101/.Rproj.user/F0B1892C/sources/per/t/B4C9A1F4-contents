library(tidyverse)
library(caret)

' http://www.sthda.com/english/articles/37-model-selection-essentials-in-r/153-penalized-regression-essentials-ridge-lasso-elastic-net/#ridge-regression
'

#options(scipen = 999) # Disable scientific notation

train <- read.csv("train.csv")
test <- read.csv("test.csv")

glimpse(train)

training.samples <- train$target %>%
  createDataPartition(p = 0.8, list = FALSE)

train.data <- train[training.samples,]
test.data <- train[-training.samples,]

# Predictor variables
x <- model.matrix(target~ . -id, train.data)[,-1]
# Outcome variable
y <- train.data$target

# Find the best lambda using cross-validation
set.seed(123) 
cv <- cv.glmnet(x, y, alpha = 0)
# Display the best lambda value
cv$lambda.min

# Fit the final model on the training data
model <- glmnet(x, y, alpha = 0, lambda = cv$lambda.min)
# Display regression coefficients
coef(model)

# Make predictions on the test data
x.test <- model.matrix(target ~. -id, test.data)[,-1]
predictions <- model %>% predict(x.test) %>% as.vector()
# Model performance metrics
data.frame(
  RMSE = RMSE(predictions, test.data$target),
  Rsquare = R2(predictions, test.data$target)
)

# +++++++++++++++++++++++++++++++++++++++++++

## LASSO ####

# Find the best lambda using cross-validation
set.seed(123) 
cv <- cv.glmnet(x, y, alpha = 1)
# Display the best lambda value
cv$lambda.min

# Fit the final model on the training data
model <- glmnet(x, y, alpha = 1, lambda = cv$lambda.min)
# Dsiplay regression coefficients
coef(model)

# Make predictions on the test data
x.test <- model.matrix(target ~. -id, test.data)[,-1]
predictions <- model %>% predict(x.test) %>% as.vector()
# Model performance metrics
data.frame(
  RMSE = RMSE(predictions, test.data$target),
  Rsquare = R2(predictions, test.data$target)
)

# +++++++++++++++++++++++++++++++++++++++++++

# Make predictions on the Kaggle test data
x.test <- model.matrix(target ~. -id, test.data)[,-1]
predictions <- model %>% predict(x.test) %>% as.vector()
# Model performance metrics
data.frame(
  RMSE = RMSE(predictions, test.data$target),
  Rsquare = R2(predictions, test.data$target)
)

predict(model, test)
