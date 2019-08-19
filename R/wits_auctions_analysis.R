# manually read out data from tensorboard
library(tidyverse)

# FPSB Uniform Symmetric ----------------------------------------------------------------------

#### 2p

fp_unif_2p <- tribble(
  ~run,             ~wall_time,  ~overhead, ~util_sp, ~util_bne, ~eps,
  "2019-08-12 Mon 14:52", 10*60+55, 3.974, 1.671,   1.666, 4.897e-4,
  "2019-08-12 Mon 15:08", 11*60+7,  3.922, 1.654,   1.665, 7.551e-4,
  "2019-08-12 Mon 15:24", 11*60+4,  3.921, 1.658,   1.665, 8.609e-4,
  "2019-08-12 Mon 15:37", 11*60+1,  3.924, 1.660,   1.666, 5.869e-4,
  "2019-08-12 Mon 16:01", 10*60+48, 3.918, 1.653,   1.664, 1.4567e-3
) %>% mutate(
  runtime = (wall_time - overhead*60) / 60
) %>% select(runtime, util_sp, util_bne, eps) %>%
  summarize_all(funs(mean,median, sd))

fp_unif_2p


#### 3p 

fp_unif_3p <- tribble(
  ~run,             ~wall_time,  ~overhead, ~util_sp, ~util_bne, ~eps,
  "2019-08-12 Mon 14:52", 12*60+58, 5.155, .8399,   .8324, 1.0241e-3,
  "2019-08-12 Mon 15:08", 12*60+54,  5.149, .8196,  .8321, 1.4317e-3,
  "2019-08-12 Mon 15:24", 13*60+07,  5.102, .8253,  .8303, 3.6343e-3,
  "2019-08-12 Mon 15:37", 12*60+58,  5.098,  .8214, .8317, 2.012e-3,
  "2019-08-12 Mon 16:01", 12*60+47, 5.1, .8291,     .8314, 2.3009e-3
) %>% mutate(
  runtime = (wall_time - overhead*60) / 60
) %>% select(runtime, util_sp, util_bne, eps) %>%
  summarize_all(funs(mean,median, sd))

fp_unif_3p %>% t()


#### 5p 

fp_unif_5p <- tribble(
  ~run,             ~wall_time,  ~overhead, ~util_sp, ~util_bne, ~eps,
  "2019-08-12 Mon 14:32", 24*60+53, 10.97, .3304,  .3325, 2.1913e-3,
  "2019-08-12 Mon 14:59", 24*60+48, 10.99, .3292,  .3321, 3.9971e-3,
  "2019-08-12 Mon 15:27", 24*60+47, 10.87, .3352,  .3322, 3.1973e-3,
  "2019-08-12 Mon 16:03", 24*60+48, 10.87, .3347, .3325,  2.5308e-3,
  "2019-08-12 Mon 16:43", 24*60+34, 10.88, .3445, .3326 , 2.1051e-3
) %>% mutate(
  runtime = (wall_time - overhead*60) / 60
) %>% select(runtime, util_sp, util_bne, eps) %>%
  summarize_all(funs(mean, sd))

fp_unif_5p %>% t()


#### 10 p

fp_unif_10p <- tribble(
  ~run,             ~wall_time,  ~overhead, ~util_sp, ~util_bne, ~eps,
  "2019-08-12 Mon 14:35", 34*60+23, 18.85, .09319,  .0895,  0.01551,
  "2019-08-12 Mon 15:18", 34*60+26, 18.80, .1038,  .08766, 0.0357,
  "2019-08-12 Mon 16:11", 34*60+03, 18.72, .1038,  .08932, 0.01746
) %>% mutate(
  runtime = (wall_time - overhead*60) / 60
) %>% select(runtime, util_sp, util_bne, eps) %>%
  summarize_all(funs(mean,median, sd))

fp_unif_10p %>% t()



# FP Assymmetric ------------------------------------------------------------------------------

fp_asymm <- tribble(
  ~run, ~success, ~wall_time, ~overhead,       ~util_sp_strong, ~util_bne_strong, ~eps_strong, ~util_sp_weak,~util_bne_weak,  ~eps_weak,
  "2019-08-12 Mon 13:27", T, 38*60+33, 20.54,     5.046, 5.027, 8.2045e-3,     0.9096,.9585,0.01071,
  "2019-08-12 Mon 14:18", T, 38*60+50, 20.28,     5.158, 5.046, 4.47687e-3,    0.8584,.9499,0.02026,
  "2019-08-12 Mon 15:06", F, NA,NA,NA,NA,NA,NA,NA,NA,
  "2019-08-12 Mon 15:35", T, 38*60+29, 20.27,     5.068, 5.03, 7.7451e-3,      0.9233,.9583,0.01147,
  "2019-08-12 Mon 16:44", T, 38*60+10, 20.3,      5.107, 5.034, 6.8063e-3,     0.9038,.9608,8.8645e-3,
  "2019-08-12 Mon 17:25", T, 38*60+02, 20.29,     5.132, 5.03,  7.7166e-3,     0.9086,.9629,6.6875e-3
) %>% mutate(
  runtime = (wall_time - overhead*60) / 60
) %>% filter(success) %>% 
  select(runtime, util_sp_strong, util_bne_strong, eps_strong, util_sp_weak, util_bne_weak, eps_weak) %>%
  summarize_all(funs(mean, sd))


fp_asymm %>% t()



# Normal Distributions ------------------------------------------------------------------------


#### 2p

fp_norm_2p <- tribble(
  ~run,             ~wall_time,  ~overhead, ~util_sp, ~util_bne, ~eps,
  "2019-08-12 Mon 21:36", 11*60+34, .4398, 2.692,   2.767, 4.0303e-3,
  "2019-08-12 Mon 23:30", 10*60+53, .4226, 2.732,   2.774, 7.6127e-4,
  "2019-08-13 Tue 01:07", 12*60+3,  .4462, 2.549,   2.74 , .01494,
  "2019-08-13 Tue 01:27", 9*60+18,  .3951, 2.633,   2.759, 8.1476e-3,
  "2019-08-13 Tue 10:33", 12*60+6,  .477 , 2.588,   2.751, .01103,
  
) %>% mutate(
  runtime = (wall_time - overhead*60) / 60
) %>% select(runtime, util_sp, util_bne, eps) %>%
  summarize_all(funs(mean, sd))

fp_norm_2p %>% t()


fp_norm_3p <- tribble(
  ~run,             ~wall_time,  ~overhead, ~util_sp, ~util_bne, ~eps,
  "2019-08-13 Tue 10:28", 32*60+52, 0.6048, 1.38, 1.401, 6.60976e-3,
  "2019-08-13 Tue 11:23", 23*60+27, 0.489, 1.294, 1.365, 0.03188,
  "2019-08-13 Tue 12:27", 21*60+59, .4739, 1.417, 1.409, 5.9748e-4,
  "2019-08-13 Tue 12:53", 20*60+37, .4548, 1.429, 1.408, 1.312e-3,
  "2019-08-13 Tue 13:26", 26*60+5,  .5064, 1.43,  1.405, 3.38365e-3
) %>% mutate(
  runtime = (wall_time - overhead*60) / 60
) %>% select(runtime, util_sp, util_bne, eps) %>%
  summarize_all(funs(mean, sd))

fp_norm_3p %>% t()

fp_norm_5p <- tribble(
  ~run,             ~wall_time,  ~overhead, ~util_sp, ~util_bne, ~eps,
  "2019-08-13 Tue 10:58", 71*60+14, 1.582, 0.6886, 0.6686, -9.7253e-4,
  "2019-08-13 Tue 12:52", 78*60+20, 1.778, 0.6881, 0.666,  2.8716e-3,
  "2019-08-19 Mon 10:04", 68*60+15, 1.516, 0.6573, 0.6677, 4.1515e-4,
  "2019-08-19 Mon 13:14", 63*60+55, 1.486, 0.67,   0.6674, 7.661e-4,
  "2019-08-19 Mon 13:27", 63*60+54, 1.497, 0.678,  0.6666, 2.0536e-3
  

  
) %>% mutate(
  runtime = (wall_time - overhead*60) / 60
) %>% select(runtime, util_sp, util_bne, eps) %>%
  summarize_all(funs(mean, sd))

fp_norm_5p %>% t()



fp_norm_10p <- tribble(
  ~run,             ~wall_time,  ~overhead, ~util_sp, ~util_bne, ~eps,
  "2019-08-12 Mon 21:13", 58*60+9, 3.448, 0.2765, 0.2681, 2.3311e-3,
  "2019-08-13 Tue 00:31", 60*60+48, 3.48, 0.2610, 0.2651, 0.01328,
  "2019-08-13 Tue 11:42", 60*60+57, 3.463, 0.2864, 0.2666, 0.01022

) %>% mutate(
  runtime = (wall_time - overhead*60) / 60
) %>% select(runtime, util_sp, util_bne, eps) %>%
  summarize_all(funs(mean, sd))

fp_norm_10p %>% t()



# gradient function plot ----------------------------------------------------------------------

library(latex2exp)

xd = 5

left <- function(x){
  if_else(x<=xd, 0, NULL)
}

right <- function(x){
  if_else(x>=xd, 10-x, NULL)
}

full <- function(x){
  10-x
}

df <- tibble(x=seq(0,12,.0001))

g <- ggplot(df,aes(x)) + 
  stat_function(fun=full, col='lightgrey', linetype='dashed') +
  stat_function(fun=left) + 
  stat_function(fun=right) + 
  #geom_vline(xintercept = xd, col='lightgrey', linetype='dashed') +
  scale_y_continuous(breaks = c(0,10), minor_breaks = NULL, limits = c(0,10), labels=c('0', TeX('v_i'))) +
  scale_x_continuous(breaks=c(0, xd, 10), minor_breaks = NULL, limits = c(0,10.1), labels = c('0', TeX('$b^{(1)}_{-i}$'), TeX('v_i'))) +
  theme_minimal() +
  labs(x='bid of player i', y = 'utility of player i') +
  theme(axis.text = element_text(size=14, face='bold'), axis.title = element_text(size=14, face='bold'))

ggsave('R/figures/utility_discontinuity.eps', device = 'eps', width=8, height = 6, units='in', dpi=300)
  

