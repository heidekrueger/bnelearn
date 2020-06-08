rm(list = ls())
### Load Packages
library(tidyverse)
library(knitr)
library(kableExtra)
library(plotly)

options(dplyr.print_max = 200)
options(dplyr.width = 300)
### Read in data
subfolder="NeurIPS"
experiment = "LLLLGG"
payment_rule = "first_price"
tb_full_raw = read_delim(str_c("experiments",subfolder,experiment,payment_rule,"/full_results.csv",
                               sep = "/", collapse = NULL), ",")
# optional
tb_bid_nearest_zero_local_raw = read_delim(str_c("experiments",subfolder,"LLG","nearest_zero","00 16.45.53 0/bidder_0_export.csv",
                              sep = "/", collapse = NULL), ",", col_names = FALSE)
tb_bid_nearest_zero_global_raw = read_delim(str_c("experiments",subfolder,"LLG","nearest_zero","00 16.45.53 0/bidder_2_export.csv",
                                    sep = "/", collapse = NULL), ",", col_names = FALSE)
### Preprocess data
tb_full_raw$tag <- gsub(tb_full_raw$tag, pattern="/", replace = "_")
tb_wide <- tb_full_raw %>% 
  pivot_wider(id_cols=c(run, subrun,epoch),names_from=tag,values_from=value) %>% 
  select(-c(hyperparameters_batch_size,hyperparameters_pretrain_iters, eval_overhead_hours))

############################### Plotting ###################################
font_size = 30

### Plot utility and util loss
tb_plot_wide <- tb_wide %>% 
  filter(subrun != ".",
         epoch %% 10 == 0,
         epoch <= 1000)

tb_plot_long <- tb_plot_wide %>% 
  select(c(run,subrun,epoch,eval_utilities,eval_util_loss_ex_ante)) %>% 
  pivot_longer(cols=-c(run,subrun,epoch),names_to="tag",values_to="value") %>% 
  mutate(tag = factor(tag), 
         tag = recode_factor(tag,eval_utilities = "ex ante utility",
                             eval_util_loss_ex_ante = "ex ante utility loss"),
         supertag = str_c(tag,subrun, sep = " - ", collapse = NULL)) 

tb_plot_long %>%
  ggplot(aes(x=epoch, y=value)) +
  stat_summary(geom="line", aes(linetype=tag, color=subrun), fun = "mean", alpha=1, size=1.5) +
  stat_summary(geom="ribbon", aes(group=supertag, fill=subrun), fun.min = min, fun.max = max, alpha=0.3) +
  theme_bw() +
  labs(x ="iteration", y = "utility (loss)", legend = "bidder type") +
  theme(legend.title = element_blank(),
        legend.text = element_text(size=font_size),
        legend.position = c(0.8, 0.35),
        #legend.background = element_rect(size=0.5, linetype="solid", colour ="grey"),
        axis.title.y = element_text(size=font_size),
        axis.text.y = element_text(size=font_size),
        axis.title.x = element_text(size=font_size),
        axis.text.x = element_text(size=font_size)) 
  #scale_linetype_discrete(labels = c(expression(tilde(u)), expression(tilde("\u2113"))))
  
### Plot bid function
# BNE bid function
fun.1 <- function(x) pmax(0,1 + log(x * (1.0)) / (1.0))
fun.2 <- function(x) x

# Prepare bids
tb_bid <- tb_bid_nearest_zero_local_raw %>% 
  mutate(type = "local",
         BNE = fun.1(X1),
         valuation = X1,
         NPGA = X2) %>%
  select(-c(X1,X2)) %>% 
  slice(which(row_number() %% 50 == 1))

tb_bid <- tb_bid_nearest_zero_global_raw %>% 
  mutate(type = "global",
         BNE = fun.2(X1),
         valuation = X1,
         NPGA = X2) %>%
  select(-c(X1,X2)) %>% 
  slice(which(row_number() %% 50 == 1)) %>% 
  rbind(tb_bid)

tb_bid_long <- tb_bid %>% 
  pivot_longer(cols=c(NPGA,BNE),names_to = "name", values_to="value")

tb_bid_long %>% 
  ggplot(aes(valuation,value,color = type)) +
  geom_point(data = filter(tb_bid_long,name=="NPGA"), aes(shape=name), size = 4) +
  geom_line(data = filter(tb_bid_long,name=="BNE"), aes(linetype=name), size = 1.2) +
  labs(x ="valuation", y = "bid", legend = "bidder type") +
  theme_bw() +
  theme(legend.title = element_blank(),
        legend.text = element_text(size=font_size),
        legend.position = c(0.15, 0.6),
        #legend.background = element_rect(size=0.5, linetype="solid", colour ="grey"),
        axis.title.y = element_text(size=font_size),
        axis.text.y = element_text(size=font_size),
        axis.title.x = element_text(size=font_size),
        axis.text.x = element_text(size=font_size)) 
