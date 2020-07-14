#TODO: Implement option that for eval the epoch is chosen in each run in which a stop_crit is given.
rm(list = ls())
### Load Packages
library(tidyverse)
library(knitr)
library(kableExtra)

options(dplyr.print_max = 200)
options(dplyr.width = 300)
### Read in data
subfolder="Journal"#Journal
experiment = "single_item"
payment_rule = "first_price/normal/symmetric/risk_neutral/3p"#"first_price/uniform/asymmetric/risk_neutral/2p/overlapping"#"first_price/normal/symmetric/risk_neutral/10p"

#"first_price/2players_2units"

tb_full_raw = read_delim(str_c("experiments",subfolder,experiment,payment_rule,"full_results.csv",
                               sep = "/", collapse = NULL), ",")

### Preprocess data
tb_full_raw$tag <- gsub(tb_full_raw$tag, pattern="/", replace = "_")
### Settings
known_bne = T
multiple_bne = F
stop_criterium_1 = 0.0005
stop_criterium_2 = 0.0001
stop_criterium_interval = 100
results_epoch = 5000
type_names = "."#c("locals", "globals")#"."#c("bidder0","bidder1")#."#c("locals", "global") #"."
#nearest_zero = c(0.13399262726306915, 0.46403446793556213) 
#nearest_bid = c(0.12500184774398804, 0.49999746680259705)
#nearest_vcg = c(0.13316573202610016, 0.4673408269882202)
#utility_in_bne_exact = c(0.13316573202610016, 0.4673408269882202)
# Current stopping criterium: eval_util_loss_ex_ante. Alternatives: eval_util_loss_rel_estimate
# further assumptions:
# - stopping criteria is fullfilled during 3 consecutive periods, each 100 epochs

if (multiple_bne){
  bne_filter <- tb_full_raw %>% 
    filter(epoch == results_epoch,
           str_detect(tag, "eval_L_2")) %>% 
    group_by(run) %>% 
    slice(which.min(value))
  
  x = "1"
  bne_x <- bne_filter %>% 
    filter(str_detect(tag, str_c("bne",x)))
  
  tb_full <- tb_full_raw %>% 
    filter(run %in% bne_x$run) %>% 
    mutate(tag = str_replace_all(tag,str_c("eval_",x),"eval"),
           tag = str_replace_all(tag,str_c("eval_epsilon_relative_bne",x),"eval_epsilon_relative"),
           tag = str_replace_all(tag,str_c("eval_epsilon_relative_bne",x), "eval_epsilon_relative"),
           tag = str_replace_all(tag,str_c("eval_epsilon_absolute_bne",x), "eval_epsilon_absolute" ),
           tag = str_replace_all(tag,str_c("eval_L_2_bne",x), "eval_L_2"),
           tag = str_replace_all(tag,str_c("eval_L_inf_bne",x), "eval_L_inf")) %>% 
    filter(tag %in% c("eval_utilities", "eval_update_norm", "eval_util_loss_ex_ante", "eval_util_loss_ex_interim","eval_overhead_hours",
                      "eval", "eval_epsilon_relative", "eval_epsilon_absolute",
                      "eval_L_2", "eval_L_inf"))
}