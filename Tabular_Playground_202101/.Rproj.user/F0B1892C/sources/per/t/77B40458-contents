# Kaggle Score: RMSQ Error=0.72782

library(tidyverse)
#getwd()
options(scipen = 999) # Disable scientific notation

train <- read_csv("train.csv")
test <- read_csv("test.csv")

glm.fit1 = glm(target ~ . -id, data = train)
summary(glm.fit1)              

glm.probs = predict(glm.fit1, type = "response") # Remove "response" for lm?

head(glm.probs)
head(train$target)

coef(glm.fit1)

glm.probs_test = predict(glm.fit1, test, type = "response")

out <- data.frame(
  Id=test$id,
  target=glm.probs_test,
  row.names=NULL)

print("Writing output")
write.csv(x=out,
          file='islr_ch4_glm.csv',
          row.names=FALSE,
          quote=FALSE)
