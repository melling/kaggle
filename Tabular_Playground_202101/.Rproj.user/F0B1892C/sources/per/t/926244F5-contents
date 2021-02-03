
# p264

library(glmnet)
library(leaps)
library(pls)

options(scipen = 999) # Disable scientific notation

Tabular <- read.csv("train.csv")
Tablular_kaggle <- read.csv("test.csv")


n = nrow(Tabular)
train = sample(1:n,n/2)
test = -train

head(Tabular)
Tabular.train=model.matrix(target ~ . -id, Tabular[train,])
Tabular.test=model.matrix(target ~ . -id, Tabular[test,])
model.errors=c()

# Best Subset Selection ####

bestsub.train=regsubsets(target ~ . -id, data=Tabular[train,],method='exhaustive',nvmax = 15)
bestsub.train.summary=summary(bestsub.train)

summary(bestsub.train)

predict.regsubsets=function(object,newdata,id,...) {
  form=as.formula(object$call[[2]])
  mat=model.matrix(form,newdata)
  coefi=coef(object,id=id)
  xvars=names(coefi)
  mat[,xvars]%*%coefi
}

# How did we do on the validation data?
# Minimum with 6 predictors
ers=c()
for(i in 1:13) {
  ers=c(ers, mean((Tabular[test,1] - predict.regsubsets(bestsub.train,newdata = Tabular[test,],i))^2)  )
  
}

plot(ers,type='l',xlab="Number of Predictors",ylab="Test MSE")

par(mfrow=c(2,2))
for(score in c('cp','rss','adjr2','bic')) {
  plot(bestsub.train.summary[[score]],xlab='Number of Predictors',ylab=score,type='l')
}

#dev.off()

bestsub.pred=predict.regsubsets(bestsub.train,newdata = Tabular[test,], which.min(ers))
model.errors=c(model.errors, bestsubset=ers[which.min(ers)])

# Lasso ####

#set.seed(37)
lasso.cv = cv.glmnet(x=Tabular.train,y=Tabular[train,'target'], alpha=1)

lasso.cv.cvm.min=which.min(lasso.cv$cvm)
lasso.cv.cvm.min
par(mfrow=c(1,1))
plot(lasso.cv$lambda,lasso.cv$cvm,type='l',xlab='Lambda',ylab='Cross Validation MSE')
points(lasso.cv$lambda.min,lasso.cv$cvm[lasso.cv.cvm.min],pch=20,col="red",cex=1)

lasso.cv$lambda.min

coef(lasso.cv,s = lasso.cv$lambda.min)

ers=c()
for(l in lasso.cv$lambda){
  lasso.pred=predict(lasso.cv,newx=Tabular.test,s = l)
  ers=c(ers,mean( (Tabular[test,'target']-lasso.pred)^2 ))
}

plot(lasso.cv$lambda,ers,type='l',ylab='Test MSE',xlab='Lambda')
points(lasso.cv$lambda.min,ers[which.min(lasso.cv$cvm)],pch=20,col="red",cex=1)

ers.min=which.min(ers)
points(lasso.cv$lambda[ers.min],ers[ers.min],pch=20,col="green",cex=1)

model.errors=c(model.errors,lasso=ers[ers.min])
lasso.pred=predict(lasso.cv,newx=Tabular.test,s = lasso.cv$lambda.min)


# PCR

pcr.fit=pcr(target~. -id,validation='CV',scale=T,data=Tabular[train,])
validationplot(pcr.fit,val.type="MSEP")

ers=c()
for(i in 1:14) {
  pcr.pred=predict(pcr.fit, newdata = Tabular[test,], ncomp = i)
  ers=c(ers,mean((Tabular[test,'target']-pcr.pred)^2))
}
plot(ers,type='l',ylab='Test MSE',xlab='Number of Components')
points(9,ers[9],col='red',pch=20)
points(which.min(ers),ers[which.min(ers)],col='green',pch=20)


## 11b ####

# Which model is best?
