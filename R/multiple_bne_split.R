#########Split data when having multiple BNE############
rm(list = ls())
### Load Packages
library(tidyverse)
library(knitr)
library(kableExtra)

options(dplyr.print_max = 200)
options(dplyr.width = 300)
### Read in data
subfolder="Journal"#Journal
experiment = "SplitAward"
payment_rule = "first_price/2players_2units"

tb_full_raw = read_delim(str_c("experiments",subfolder,experiment,payment_rule,"full_results.csv",
                               sep = "/", collapse = NULL), ",")

### Preprocess data
tb_full_raw$tag <- gsub(tb_full_raw$tag, pattern="/", replace = "_")

### Split in different BNE datasets determined by L2###

tb_full_raw %>% 
  filter ()
  