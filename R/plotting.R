source("R/Load_and_prepare_data.R")

# Bid data
tb_bid = read_delim(str_c("experiments",subfolder,experiment,payment_rule,"bidder_0_export.csv",
                                      sep = "/", collapse = NULL), ",", col_names = FALSE)

tb_bid_1 = read_delim(str_c("experiments",subfolder,experiment,payment_rule,"bidder_1_export.csv",
                          sep = "/", collapse = NULL), ",", col_names = FALSE)

tb_wide <- tb_full_raw %>%
  pivot_wider(id_cols=c(run, subrun,epoch),names_from=tag,values_from=value) %>%
  select(-c(contains("hyperparameters_batch_size"),contains("hyperparameters_pretrain_iters"), eval_overhead_hours))

############################### Plotting ###################################
font_size = 30

######## Plot utility and util loss#######
tb_plot_wide <- tb_wide %>% 
  filter(#subrun != ".",
         epoch %% 10 == 0,
         epoch <= 5000)

tb_plot_long <- tb_plot_wide %>% 
  select(c(run,subrun,epoch,eval_utilities,eval_utility_vs_bne)) %>%  #eval_util_loss_ex_ante
  pivot_longer(cols=-c(run,subrun,epoch),names_to="tag",values_to="value") %>% 
  mutate(tag = factor(tag), 
         tag = recode_factor(tag,eval_utilities = "ex-ante utility",
                             eval_util_loss_ex_ante = "ex-ante utility vs BNE"), #loss
         supertag = str_c(tag,subrun, sep = " - ", collapse = NULL)) 

tb_plot_long %>%
  ggplot(aes(x=epoch, y=value)) +
  stat_summary(geom="line", aes(linetype=tag), fun = "mean", alpha=1, size=1.5) + #, color=subrun
  stat_summary(geom="ribbon", aes(group=supertag), fun.min = min, fun.max = max, alpha=0.3) + #, fill=subrun
  theme_bw() +
  labs(x ="iteration", y = "utility", legend = "bidder type") +
  theme(legend.title = element_blank(),
        legend.text = element_text(size=font_size),
        legend.position = c(0.8, 0.35),
        #legend.background = element_rect(size=0.5, linetype="solid", colour ="grey"),
        axis.title.y = element_text(size=font_size),
        axis.text.y = element_text(size=font_size),
        axis.title.x = element_text(size=font_size),
        axis.text.x = element_text(size=font_size)) 
  #scale_linetype_discrete(labels = c(expression(tilde(u)), expression(tilde("\u2113"))))
  
########## Plot bid function##########
### BNE bid function for LLG
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

### BNE bid function for single_item, normal, symmetric, 3p #TODO: WIP!
fun.dnorm_15_10 <- function(x){
  return(pnorm(x,10,5)^3)
}
fun.integ_norm <- function(v){
  return(v-(integrate(fun.dnorm_15_10,0,v)$value/fun.dnorm_15_10(v)))
}
fun.integ_norm_vec <- Vectorize(fun.integ_norm)


tb_bid_plot <- tb_bid %>% 
  mutate(BNE = fun.integ_norm_vec(X1),
         valuation = X1,
         NPGA = X2) %>%
  select(-c(X1,X2)) %>% 
  slice(which(row_number() %% 50 == 1))

tb_bid_long <- tb_bid_plot %>% 
  pivot_longer(cols=c(NPGA,BNE),names_to = "name", values_to="value")

tb_bid_long %>% 
  ggplot(aes(valuation,value)) +
  geom_point(data = filter(tb_bid_long,name=="NPGA"), aes(shape=name), size = 4) +
  geom_line(data = filter(tb_bid_long,name=="BNE"), aes(linetype=name), size = 1.2) +
  labs(x ="valuation", y = "bid", legend = "bidder type") +
  theme_bw() +
  xlim(0,25)+ylim(0,20)+
  theme(legend.title = element_blank(),
        legend.text = element_text(size=font_size),
        legend.position = c(0.15, 0.6),
        #legend.background = element_rect(size=0.5, linetype="solid", colour ="grey"),
        axis.title.y = element_text(size=font_size),
        axis.text.y = element_text(size=font_size),
        axis.title.x = element_text(size=font_size),
        axis.text.x = element_text(size=font_size)) 

### BNE bid function for single_item, uniform, asymmetric, 2p, overlapping #TODO: WIP!
fun.1 <- function(valuation, player){
  c = 1 / (15 - 5) ** 2 - 1 / (25 - 5) ** 2
  factor = 2 * player - 1  # -1 for 0 (weak player), +1 for 1 (strong player)
  denominator = 1.0 + sqrt(1 + factor * c * (valuation - 5) ** 2)
  bid = 5 + (valuation - 5) / denominator
  return(pmax(bid, 0))
}

# Prepare bids
tb_bid_plot <- tb_bid %>% 
  mutate(type = "weak",
         BNE = fun.1(X1,0),
         valuation = X1,
         NPGA = X2) %>%
  select(-c(X1,X2)) %>% 
  slice(which(row_number() %% 50 == 1))

tb_bid_plot <- tb_bid_1 %>% 
  mutate(type = "strong",
         BNE = fun.1(X1,1),
         valuation = X1,
         NPGA = X2) %>%
  select(-c(X1,X2)) %>% 
  slice(which(row_number() %% 50 == 1)) %>% 
  rbind(tb_bid_plot)

tb_bid_long <- tb_bid_plot %>% 
  pivot_longer(cols=c(NPGA,BNE),names_to = "name", values_to="value")

tb_bid_long %>% 
  ggplot(aes(valuation,value,color = type)) +
  geom_point(data = filter(tb_bid_long,name=="NPGA"), aes(shape=name), size = 4) +
  geom_line(data = filter(tb_bid_long,name=="BNE"), aes(linetype=name), size = 1.2) +
  ylim(5,12) +
  labs(x ="valuation", y = "bid", legend = "bidder type") +
  theme_bw() +
  theme(legend.title = element_blank(),
        legend.text = element_text(size=font_size),
        legend.position = c(0.85, 0.4),
        #legend.background = element_rect(size=0.5, linetype="solid", colour ="grey"),
        axis.title.y = element_text(size=font_size),
        axis.text.y = element_text(size=font_size),
        axis.title.x = element_text(size=font_size),
        axis.text.x = element_text(size=font_size)) 






        