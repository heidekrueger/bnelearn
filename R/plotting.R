#TODO: Implement option that for eval the epoch is chosen in each run in which a stop_crit is given.
rm(list = ls())
### Load Packages
library(tidyverse)
library(knitr)
library(kableExtra)

options(dplyr.print_max = 200)
options(dplyr.width = 300)
### Read in data
subfolder="NeurIPS"
experiment = "LLLLGG"
payment_rule = "first_price"
tb_full_raw = read_delim(str_c("experiments",subfolder,experiment,payment_rule,"/full_results.csv",
                               sep = "/", collapse = NULL), ",")

### Preprocess data
tb_full_raw$tag <- gsub(tb_full_raw$tag, pattern="/", replace = "_")
tb_wide <- tb_full_raw %>% 
  pivot_wider(id_cols=c(run, subrun,epoch),names_from=tag,values_from=value) %>% 
  select(-c(hyperparameters_batch_size,hyperparameters_pretrain_iters, eval_overhead_hours))

### Plotting
font_size = 15

tb_wide %>% 
  filter(subrun != ".",
         epoch %% 10 == 0,
         epoch <= 1000) %>% 
  
  select(c(run,subrun,epoch,eval_utilities,eval_util_loss_ex_ante)) %>% 
  pivot_longer(cols=-c(run,subrun,epoch),names_to="tag",values_to="value") %>% 
  
  mutate(tag = factor(tag), 
         tag = recode_factor(tag,eval_utilities = "utility loss",
                             eval_util_loss_ex_ante = "ex ante utility loss")) %>% 
  ggplot() + 
    
  geom_point(aes(x=epoch,y=value, group=tag, color=subrun, shape=as.factor(tag)) ,size=2) +
    
  #geom_point(aes(x=epoch,y=eval_utilities, group=subrun, color=subrun), shape=1 ,size=2) +
  #geom_point(aes(x=epoch,y=eval_util_loss_ex_ante, group=subrun, color=subrun), shape=2, size=2) +
  #geom_smooth(aes(x=epoch,y=eval_util_loss_ex_ante, group=subrun, color=subrun),size=1, level=0.99) +
  theme_bw() + theme_classic() +
  guides(group = guide_legend(reverse=TRUE)) +
  labs(x ="iteration", y = "utility", legend = "bidder type") +
  theme(legend.title = element_blank(),
        axis.title.y = element_text(size=font_size),
        axis.text.y = element_text(size=font_size),
        axis.title.x = element_text(size=font_size),
        axis.text.x = element_text(size=font_size))
  
  
  