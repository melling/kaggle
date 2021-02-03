# https://www.kaggle.com/kailex/tabular-playground

library(umap)
library(knitr)
library(GGally)
library(mclust)
library(recipes)
library(rsample)
library(lightgbm)
library(tidyverse)
library(ggcorrplot)

path <- "./"
kfolds <- 5

# 2:25pm started
tr <- read_csv(str_c(path, "train.csv"))
te <- read_csv(str_c(path, "test.csv"))
sub <- read_csv(str_c(path, "sample_submission.csv"))


# head(sub, 5) %>% kable()

# head(tr, 5) %>% kable()

# glimpse(tr)

tr %>% 
  select(starts_with("cont")) %>% 
  mutate(grp = "train") %>%
  bind_rows(
    (te %>% 
       select(starts_with("cont")) %>% 
       mutate(grp = "test"))
  ) %>% 
  pivot_longer(cols = starts_with("cont")) %>% 
  group_by(name, grp) %>% 
  mutate(mean = mean(value)) %>% 
  ggplot(aes(x = value)) + 
  facet_wrap(~name, ncol = 2, scales = "free") +
  geom_density(aes(fill = grp), alpha = 0.3) +
  geom_vline(aes(xintercept = mean), linetype = "dashed", size = 0.2) +
  theme_minimal() + 
  theme(legend.position = "top") +
  labs(fill = "")


## Correlation

tr %>% 
  select(-id) %>%
  # select(cont1,cont2) %>% 
  ggcorr(label = TRUE, label_size = 3, label_round = 2, label_alpha = TRUE)

# It's strange that almost nothing correlates with the target. The pairs plot below show relations between features themselves:

tr %>% 
  select(-id, -target) %>% 
  sample_frac(0.05) %>% 
  ggpairs(lower = list(continuous = wrap(ggally_points, col = "steelblue", size = 0.0025)),
          diag = list(continuous = wrap(ggally_densityDiag, col = "#F8766D")),
          upper = list(continuous = "blank"), 
          axisLabels = "none",
          progress = FALSE) +
  theme_minimal()


# You might have noticed the strange pattern of the `cont2` feature:

tr %>% 
  select(cont2) %>% 
  mutate(mean = mean(cont2)) %>% 
  ggplot(aes(x = cont2)) + 
  geom_histogram(alpha = 0.8, bins = 500) +
  geom_vline(aes(xintercept = mean), linetype = "dashed", size = 0.2) +
  theme_minimal() + 
  theme(legend.position = "top") +
  labs(fill = "", y = "")
