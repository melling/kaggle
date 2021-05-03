library(tidyverse)

# d = 1300:1200:-2
# d = 1300
# ndays = 8 - 1
# recent_days = seq(1300, 1300 - 7 * ndays, by=-7)
# 
# paste0(c("d", recent_days), collapse = "_")
# cols <- c(paste0("d_", recent_days))
# cols

# https://stackoverflow.com/questions/23222069/what-is-the-most-efficient-way-to-sum-all-columns-whose-name-starts-with-a-patte/23223594

d = 1300
ndays = 20 - 1
recent_days = seq(d, d - 7 * ndays, by=-7)
cols <- c(paste0(“d_“, recent_days))

train %>% 
  filter(str_detect(item_id, "HOBBIES_1_001")) %>%
  filter(str_detect(store_id, "CA_1")) %>%
  select(state_id, store_id, cat_id, item_id, all_of(cols))

