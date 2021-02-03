# https://www.kaggle.com/siero5335/eda-gbdts-xgboost-catboost-and-lightgbm-in-r

# https://www.kaggle.com/kailex/tabular-playground

library(tidyverse); library(data.table); library(FactoMineR); library(xgboost)
library(factoextra); library(ggpubr); library(DALEX)
library(corrplot); library(skimr); library(Matrix); library(caret); library(ggExtra)
# library(catboost);
library(lightgbm);
library(GGally)

# install.packages("factoextra", "DALEX")
# install.packages("catboost")
# install.packages("lightgbm")
# install.packages("DALEX")

train <- fread("train.csv", data.table = F)
test <- fread("test.csv", data.table = F)

train <- train %>% filter(target > 5.5)

skim(train)

skim(test)

train %>% select(target) %>%
  ggplot() +
  geom_density(aes(x=target), fill = "#000080") +
  theme_minimal()

M_train <- cor(train %>% select(!id))

corrplot(M_train, method = "circle", order = "hclust", addrect = 3)

M_test <- cor(test %>% select(!id))
corrplot(M_test, method = "circle", order = "hclust", addrect = 3)

p <- ggplot(train, aes(x=cont1, y=target)) + geom_point(col="transparent")
# Marginal ...
ggMarginal(p + geom_hex() + scale_fill_continuous(type = "viridis"), type="density", fill = "slateblue")

p <- ggplot(train, aes(x=cont2, y=target)) + geom_point(col="transparent")

ggMarginal(p + geom_hex() + scale_fill_continuous(type = "viridis"), type="density", fill = "slateblue")

p <- ggplot(train, aes(x=cont3, y=target)) + geom_point(col="transparent")

ggMarginal(p + geom_hex() + scale_fill_continuous(type = "viridis"), type="density", fill = "slateblue")

p <- ggplot(train, aes(x=cont4, y=target)) + geom_point(col="transparent")
ggMarginal(p + geom_hex() + scale_fill_continuous(type = "viridis"), type="density", fill = "slateblue")

# In[13]
p <- ggplot(train, aes(x=cont5, y=target)) + geom_point(col="transparent")
ggMarginal(p + geom_hex() + scale_fill_continuous(type = "viridis"), type="density", fill = "slateblue")

# In[14]
p <- ggplot(train, aes(x=cont6, y=target)) + geom_point(col="transparent")
ggMarginal(p + geom_hex() + scale_fill_continuous(type = "viridis"), type="density", fill = "slateblue")

# In[15]

p <- ggplot(train, aes(x=cont7, y=target)) + geom_point(col="transparent")
ggMarginal(p + geom_hex() + scale_fill_continuous(type = "viridis"), type="density", fill = "slateblue")

# In[16]

p <- ggplot(train, aes(x=cont8, y=target)) + geom_point(col="transparent")
ggMarginal(p + geom_hex() + scale_fill_continuous(type = "viridis"), type="density", fill = "slateblue")

# In[17]
p <- ggplot(train, aes(x=cont9, y=target)) + geom_point(col="transparent")
ggMarginal(p + geom_hex() + scale_fill_continuous(type = "viridis"), type="density", fill = "slateblue")


# In[37]
train <- train %>% mutate(data = "train")
test <- test %>% mutate(data = "test")

df <- rbind(train %>% select(!target), test)

# In [39]:

#ggviolin(df, x = "data", y = "cont1", fill = "data", palette = c("#00AFBB", "#E7B800"), add = "boxplot", add.params = list(fill = "white")) + stat_compare_means(label.y = 1.2)

p1 <- ggviolin(df, x = "data", y = "cont1", fill = "data", palette = c("#00AFBB", "#E7B800"), add = "boxplot", add.params = list(fill = "white")) +
    stat_compare_means(label.y = 1.2)


p2 <- ggviolin(df, x = "data", y = "cont2", fill = "data", palette = c("#00AFBB", "#E7B800"), add = "boxplot", add.params = list(fill = "white")) + stat_compare_means(label.y = 1)

p3 <- ggviolin(df, x = "data", y = "cont3", fill = "data",
               palette = c("#00AFBB", "#E7B800"),
               add = "boxplot", add.params = list(fill = "white")) +
  stat_compare_means(label.y = 1.2) 

p4 <- ggviolin(df, x = "data", y = "cont4", fill = "data",
               palette = c("#00AFBB", "#E7B800"),
               add = "boxplot", add.params = list(fill = "white")) +
  stat_compare_means(label.y = 1.2) 

p5 <- ggviolin(df, x = "data", y = "cont5", fill = "data",
               palette = c("#00AFBB", "#E7B800"),
               add = "boxplot", add.params = list(fill = "white")) +
  stat_compare_means(label.y = 1.2) 

p6 <- ggviolin(df, x = "data", y = "cont6", fill = "data",
               palette = c("#00AFBB", "#E7B800"),
               add = "boxplot", add.params = list(fill = "white")) +
  stat_compare_means(label.y = 1.2) 

p7 <- ggviolin(df, x = "data", y = "cont7", fill = "data",
               palette = c("#00AFBB", "#E7B800"),
               add = "boxplot", add.params = list(fill = "white")) +
  stat_compare_means(label.y = 1.2) 

p8 <- ggviolin(df, x = "data", y = "cont8", fill = "data",
               palette = c("#00AFBB", "#E7B800"),
               add = "boxplot", add.params = list(fill = "white")) +
  stat_compare_means(label.y = 1.2) 

p9 <- ggviolin(df, x = "data", y = "cont9", fill = "data",
               palette = c("#00AFBB", "#E7B800"),
               add = "boxplot", add.params = list(fill = "white")) +
  stat_compare_means(label.y = 1.2) 

p10 <- ggviolin(df, x = "data", y = "cont10", fill = "data",
                palette = c("#00AFBB", "#E7B800"),
                add = "boxplot", add.params = list(fill = "white")) +
  stat_compare_means(label.y = 1.4) 

p11 <- ggviolin(df, x = "data", y = "cont11", fill = "data",
                palette = c("#00AFBB", "#E7B800"),
                add = "boxplot", add.params = list(fill = "white")) +
  stat_compare_means(label.y = 1.2) 

p12 <- ggviolin(df, x = "data", y = "cont12", fill = "data",
                palette = c("#00AFBB", "#E7B800"),
                add = "boxplot", add.params = list(fill = "white")) +
  stat_compare_means(label.y = 1.2) 

p13 <- ggviolin(df, x = "data", y = "cont13", fill = "data",
                palette = c("#00AFBB", "#E7B800"),
                add = "boxplot", add.params = list(fill = "white")) +
  stat_compare_means(label.y = 1.2) 

p14 <- ggviolin(df, x = "data", y = "cont14", fill = "data",
                palette = c("#00AFBB", "#E7B800"),
                add = "boxplot", add.params = list(fill = "white")) +
  stat_compare_means(label.y = 1.2)

# In [40]:
  
options(repr.plot.width = 14, repr.plot.height = 7)
ggarrange(p1, p2, p3, p4, p5, p6, p7, 
          p8, p9, p10, p11, p12, p13, p14, ncol = 2, common.legend = TRUE)

# There seems to be no difference in the distribution of features between the train and the test set.

## PCA ####

# In [40]:
res.pca <- PCA(train %>% select(!c(id, data)),  graph = FALSE)

# In [42]:

options(repr.plot.width = 7, repr.plot.height = 7)
fviz_screeplot(res.pca, addlabels = TRUE, ylim = c(0, 50))

# In [43]:
fviz_pca_var(res.pca, col.var="contrib",
             axes = c(1, 2),
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE
)

# In [44]:
fviz_pca_var(res.pca, col.var="contrib",
             axes = c(1, 3),
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE
)
