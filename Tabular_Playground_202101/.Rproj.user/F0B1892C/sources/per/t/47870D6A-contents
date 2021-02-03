options(scipen = 999) # Disable scientific notation

train <- read_csv("train.csv")
test <- read_csv("test.csv")

x=model.matrix(target~.,train)[,-1] # -1 Removes Intercept, column 1 created by model.matrix
y=train$target

library(glmnet)
grid=10^seq(10,-2,length=100) # λ = 1010 to λ = 10^−2
ridge.mod=glmnet(x,y,alpha=0,lambda=grid) # alpha=0 means ridge, alpha=1 means 
coef(ridge.mod)
dim(coef(ridge.mod))

plot(ridge.mod$lambda)
ridge.mod$lambda[5] # Large lambda implies small coefficients

ridge.mod$lambda[50] # Large lambda implies small coefficients
sqrt(sum(coef(ridge.mod)[-1,50]^2)) # l2 norm

ridge.mod$lambda [60] # Small lambda implies larger coefficients
sqrt(sum(coef(ridge.mod)[-1,60]^2)) # l2 norm

train=sample(1:nrow(x), nrow(x)/2)
test=(-train)
y.test=y[test]

ridge.mod=glmnet(x[train,],y[train],alpha=0,lambda=grid, thresh=1e-12)

ridge.pred=predict(ridge.mod,s=4,newx=x[test,])
mean((ridge.pred-y.test)^2)

set.seed(1)
cv.out=cv.glmnet(x[train ,],y[train],alpha=0)
plot(cv.out)
best_lambda=cv.out$lambda.min
best_lambda

ridge.pred=predict(ridge.mod,s=best_lambda ,newx=x[test,])
mean((ridge.pred-y.test)^2)

out=glmnet(x,y,alpha=0)
predict(out,type="coefficients",s=bestlam)[1:20,]
