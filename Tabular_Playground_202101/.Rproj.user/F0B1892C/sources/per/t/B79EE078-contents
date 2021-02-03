library(mlr)

library(tidyverse)

train <- read.csv("train.csv")
test <- read.csv("test.csv")

ozoneTask <- makeRegrTask(data = train, target = "target")

lin <- makeLearner("regr.lm")

listFilterMethods()
filterVals <- generateFilterValuesData(ozoneTask, 
                                       method = "linear.correlation")

filterVals$data

plotFilterValues(filterVals) + theme_bw()

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
