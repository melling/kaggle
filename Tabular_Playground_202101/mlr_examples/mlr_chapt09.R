###############################################
#          SOURCE CODE FOR CHAPTER 9          #
###############################################

# LOAD PACKAGES ----
library(mlr)

library(tidyverse)

# LOAD DATA ----
data(Ozone, package = "mlbench")

ozoneTib <- as_tibble(Ozone)

names(ozoneTib) <- c("Month", "Date", "Day", "Ozone", "Press_height", 
                     "Wind", "Humid", "Temp_Sand", "Temp_Monte", 
                     "Inv_height", "Press_grad", "Inv_temp", "Visib")

#ozoneTib[is.na(Ozone)]

ozoneClean <- mutate_all(ozoneTib, as.numeric) %>%
  filter(is.na(Ozone) == FALSE)

ozoneClean

# PLOT DATA ----
ozoneUntidy <- gather(ozoneClean, key = "Variable", value = "Value", -Ozone)

ozoneUntidy 

ggplot(ozoneUntidy, aes(Value, Ozone)) +
  facet_wrap(~ Variable, scale = "free_x") +
  geom_point() +
  geom_smooth() +
  geom_smooth(method = "lm", col = "red") +
  theme_bw()

#ggsave("Ozone plots.pdf", width = 10, height = 6)

# IMPUTE MISSING VALUES ----
#?imputations

imputeMethod <- imputeLearner("regr.rpart")

ozoneImp <- impute(as.data.frame(ozoneClean), 
                   classes = list(numeric = imputeMethod))

# MAKE TASK AND LEARNER ----
ozoneTask <- makeRegrTask(data = ozoneImp$data, target = "Ozone")

lin <- makeLearner("regr.lm")

# FEATURE SELECTION FILTER METHOD ----
#install.packages("FSelector")
listFilterMethods()
filterVals <- generateFilterValuesData(ozoneTask, 
                                       method = "linear.correlation")

filterVals$data

plotFilterValues(filterVals) + theme_bw()
#ggsave("Filtervals.pdf", width = 10, height = 6)

#ozoneFiltTask <- filterFeatures(ozoneTask, fval = filterVals, abs = 6) #6 most important features
#ozoneFiltTask <- filterFeatures(ozoneTask, fval = filterVals, per = 0.25) #25% most important features
#ozoneFiltTask <- filterFeatures(ozoneTask, fval = filterVals, threshold = 0.2) #variables with contribution > 2

# FILTER WRAPPER ----
filterWrapper <- makeFilterWrapper(learner = lin, 
                                   fw.method = "linear.correlation")

lmParamSpace <- makeParamSet(
  makeIntegerParam("fw.abs", lower = 1, upper = 12)
)

gridSearch <- makeTuneControlGrid()

kFold <- makeResampleDesc("CV", iters = 10)

tunedFeats <- tuneParams(filterWrapper, task = ozoneTask, resampling = kFold,
                         par.set = lmParamSpace, control = gridSearch)

tunedFeats

# MAKE NEW TASK AND TRAIN MODEL FOR FILTER METHOD ----
filteredTask <- filterFeatures(ozoneTask, fval = filterVals,
                               abs = unlist(tunedFeats$x))

filteredModel <- train(lin, filteredTask)

# FEATURE SELECTION WRAPPER METHOD ----
featSelControl <- makeFeatSelControlSequential(method = "sfbs")

selFeats <- selectFeatures(learner = lin, task = ozoneTask, 
                           resampling = kFold, control = featSelControl)

selFeats

# MAKE NEW TASK AND TRAIN MODEL FOR THE WRAPPER METHOD ----
ozoneSelFeat <- ozoneImp$data[, c("Ozone", selFeats$x)]

ozoneSelFeatTask <- makeRegrTask(data = ozoneSelFeat, target = "Ozone")

wrapperModel <- train(lin, ozoneSelFeatTask)

# MAKE IMPUTATION WRAPPER FOR CROSS-VALIDATION ----
imputeMethod <- imputeLearner("regr.rpart")

imputeWrapper <- makeImputeWrapper(lin, 
                                   classes = list(numeric = imputeMethod))

# MAKE FEATURE SELECTION WRAPPER FOR CROSS-VALIDATION ----
featSelControl <- makeFeatSelControlSequential(method = "sfbs")

featSelWrapper <- makeFeatSelWrapper(learner = imputeWrapper, 
                                     resampling = kFold,
                                     control = featSelControl)

# CROSS-VALIDATE MODEL BUILDING PROCESS ----
library(parallel)
library(parallelMap)

ozoneTaskWithNAs <- makeRegrTask(data = ozoneClean, target = "Ozone")

kFold3 <- makeResampleDesc("CV", iters = 3)

parallelStartSocket(cpus = detectCores())

lmCV <- resample(featSelWrapper, ozoneTaskWithNAs, resampling = kFold3) #~ 1.5 min

parallelStop()

lmCV


# LOOK AT MODEL DIAGNOSTICS ----
wrapperModelData <- getLearnerModel(wrapperModel)
summary(wrapperModelData)

par(mfrow = c(2, 2))
plot(wrapperModelData)
par(mfrow = c(1, 1))

# SOLUTIONS TO EXERCISES ----
# 1
filterValsForest <- generateFilterValuesData(ozoneTask, 
                                             method = "randomForestSRC_importance")

filterValsForest$data

plotFilterValues(filterValsForest) + theme_bw()

# 2
filterWrapperDefault <- makeFilterWrapper(learner = lin)

tunedFeats <- tuneParams(filterWrapperDefault, task = ozoneTask, 
                         resampling = kFold, par.set = lmParamSpace, 
                         control = gridSearch)

tunedFeats

# the defaut filter statistic (randomForestSRC) tends to select fewer 
# predictors in this case, but the linear.correlation statistic was faster

# 3
filterWrapperImp <- makeFilterWrapper(learner = imputeWrapper, 
                                      fw.method = "linear.correlation")
filterParam <- makeParamSet(
  makeIntegerParam("fw.abs", lower = 1, upper = 12)
)

tuneWrapper <- makeTuneWrapper(learner = filterWrapperImp, 
                               resampling = kFold,
                               par.set = filterParam, 
                               control = gridSearch)

filterCV <- resample(tuneWrapper, ozoneTask, resampling = kFold)

filterCV

# we have a similar MSE estimate for the filter method
# but it is considerably faster than the wrapper method. No free lunch!
