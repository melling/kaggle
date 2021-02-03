# Kaggle Score: 0.72782
# 0.72782

library(tidyverse)
library(caret)
#getwd()
options(scipen = 999) # Disable scientific notation

train <- read_csv("train.csv")
test <- read_csv("test.csv")

set.seed(1)

# https://towardsdatascience.com/create-predictive-models-in-r-with-caret-12baf9941236

dataset <- train

# Repeated K-fold cross-validation
train_control <- trainControl(method = "repeatedcv",  
                              number = 10, repeats = 3)

model.lm <- train(target ~. -id, data = train,   
               method = "lm",  
               trControl = train_control)

print(model.lm)

predictions <- predict(model.lm, test)
print(predictions)
# model <- train(target ~ . -id, data = dataset,  
#                trControl = train_control, method = "nb") # Naive Bayes

# cor(train)
# glm.fit1 = glm(target ~ . -id -cont14, data = train)
# summary(glm.fit1)              

# glm.probs = predict(glm.fit1, type = "response")

# https://stackoverflow.com/questions/43123462/how-to-obtain-rmse-out-of-lm-result
RSS <- c(crossprod(glm.fit1$residuals))
MSE <- RSS / length(glm.fit1$residuals)
RMSE <- sqrt(MSE) # .72619

# head(glm.probs)
# head(train$target)
# 
# coef(glm.fit1)

# glm.probs_test = predict(glm.fit1, test, type = "response")

out <- data.frame(
  Id=test$id,
  target=predictions,
  row.names=NULL)

print("Writing output")
write.csv(x=out,
          file='caret_lm.csv',
          row.names=FALSE,
          quote=FALSE)
